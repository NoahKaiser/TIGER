#!/usr/bin/env python3
"""Count active speakers per segment for ECHI train and validation splits.

Example:
    python analyze_echi_active_speakers.py configs/tiger_tse2.yml
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

import look2hear.datas


MAX_ACTIVE_REPORTED = 4


@dataclass
class SplitSummary:
    name: str
    total_segments: int
    counts_0_to_4: list[int]
    overflow_segments: int

    @property
    def mean_active(self) -> float:
        if self.total_segments == 0:
            return 0.0
        weighted = sum(i * c for i, c in enumerate(self.counts_0_to_4))
        if self.overflow_segments > 0:
            weighted += MAX_ACTIVE_REPORTED * self.overflow_segments
        return weighted / float(self.total_segments)

    @property
    def most_common_active(self) -> int:
        if not self.counts_0_to_4:
            return 0
        return int(max(range(len(self.counts_0_to_4)), key=lambda i: self.counts_0_to_4[i]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ECHI segment distribution by number of active speakers "
            "for TRAIN and VALID splits."
        )
    )
    parser.add_argument(
        "conf",
        type=str,
        help="Path to YAML config (e.g. configs/tiger_tse2.yml).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for analysis DataLoader (default: 64).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader num_workers (default: use config).",
    )
    parser.add_argument(
        "--activity_tau",
        type=float,
        default=None,
        help=(
            "Activity threshold tau. Default is training.pit_activity_tau from config "
            "(fallback 1e-6)."
        ),
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=0,
        help="Optional cap per split for quick checks (0 = full split).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path for per-split counts.",
    )
    return parser.parse_args()


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a dict in YAML config.")
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _require_dict(cfg, "root")


def build_datamodule(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any], float]:
    dm_root = _require_dict(cfg.get("datamodule", {}), "datamodule")
    training_root = _require_dict(cfg.get("training", {}), "training")

    data_name = dm_root.get("data_name", None)
    if data_name != "ECHIDataModule":
        raise ValueError(
            f"This script expects datamodule.data_name == 'ECHIDataModule', got: {data_name}"
        )

    data_cfg = _require_dict(dm_root.get("data_config", {}), "datamodule.data_config")
    datamodule_cls = getattr(look2hear.datas, str(data_name), None)
    if datamodule_cls is None:
        raise ValueError(f"Could not find datamodule class '{data_name}' in look2hear.datas.")

    activity_tau = float(training_root.get("pit_activity_tau", 1e-6))
    datamodule = datamodule_cls(**data_cfg)
    datamodule.setup()
    return datamodule, data_cfg, activity_tau


def _count_active_speakers(
    dataset: Any,
    split_name: str,
    activity_tau: float,
    batch_size: int,
    num_workers: int,
    max_segments: int,
) -> SplitSummary:
    if dataset is None:
        raise RuntimeError(f"Split '{split_name}' is None.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )

    counts = torch.zeros(MAX_ACTIVE_REPORTED + 1, dtype=torch.long)
    overflow = 0
    seen = 0
    target_total = len(dataset) if max_segments <= 0 else min(len(dataset), max_segments)

    print(f"[{split_name}] Processing {target_total} / {len(dataset)} segments...")
    for batch in loader:
        if seen >= target_total:
            break

        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise RuntimeError(
                f"Unexpected batch format for split '{split_name}'. Expected (mix, targets, ...)."
            )

        targets = batch[1]
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        if targets.ndim != 3:
            raise RuntimeError(
                f"Expected targets with shape [B, n_src, T], got {tuple(targets.shape)}."
            )

        if seen + targets.shape[0] > target_total:
            keep = target_total - seen
            targets = targets[:keep]

        target_energy = (targets ** 2).mean(dim=-1)  # [B, n_src]
        n_active = (target_energy > activity_tau).sum(dim=-1).to(torch.long)  # [B]

        overflow += int((n_active > MAX_ACTIVE_REPORTED).sum().item())
        capped = n_active.clamp(max=MAX_ACTIVE_REPORTED)
        bincounts = torch.bincount(capped, minlength=MAX_ACTIVE_REPORTED + 1)
        counts += bincounts[: MAX_ACTIVE_REPORTED + 1]

        seen += int(targets.shape[0])

    return SplitSummary(
        name=split_name,
        total_segments=int(seen),
        counts_0_to_4=[int(x) for x in counts.tolist()],
        overflow_segments=int(overflow),
    )


def _format_percent(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(100.0 * float(count) / float(total)):.2f}%"


def print_report(
    train_summary: SplitSummary,
    valid_summary: SplitSummary,
    activity_tau: float,
    data_cfg: dict[str, Any],
) -> None:
    n_src = int(data_cfg.get("n_src", 4))
    print("")
    print("ECHI Active-Speaker Distribution")
    print("--------------------------------")
    print(f"train_dir   : {data_cfg.get('train_dir')}")
    print(f"valid_dir   : {data_cfg.get('valid_dir')}")
    print(f"segment [s] : {data_cfg.get('segment')}")
    print(f"hop [s]     : {data_cfg.get('hop', None)}")
    print(f"sample_rate : {data_cfg.get('sample_rate')}")
    print(f"n_src       : {n_src}")
    print(f"activity_tau: {activity_tau}")
    print("")

    header = (
        f"{'n_active':>8} | {'train_count':>12} {'train_%':>8} | "
        f"{'valid_count':>12} {'valid_%':>8}"
    )
    print(header)
    print("-" * len(header))
    for n_active in range(MAX_ACTIVE_REPORTED + 1):
        t = train_summary.counts_0_to_4[n_active]
        v = valid_summary.counts_0_to_4[n_active]
        print(
            f"{n_active:>8} | {t:>12} {_format_percent(t, train_summary.total_segments):>8} | "
            f"{v:>12} {_format_percent(v, valid_summary.total_segments):>8}"
        )

    print("-" * len(header))
    print(
        f"{'total':>8} | {train_summary.total_segments:>12} {'100.00%':>8} | "
        f"{valid_summary.total_segments:>12} {'100.00%':>8}"
    )
    print("")
    print(
        f"train: mean_active={train_summary.mean_active:.3f}, "
        f"mode={train_summary.most_common_active}"
    )
    print(
        f"valid: mean_active={valid_summary.mean_active:.3f}, "
        f"mode={valid_summary.most_common_active}"
    )
    if train_summary.overflow_segments or valid_summary.overflow_segments:
        print(
            "warning: Found segments with n_active > 4 "
            f"(train={train_summary.overflow_segments}, valid={valid_summary.overflow_segments})."
        )


def write_csv(
    out_path: Path,
    train_summary: SplitSummary,
    valid_summary: SplitSummary,
    activity_tau: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "total_segments",
        "n_active",
        "count",
        "fraction",
        "activity_tau",
    ]
    rows = []
    for summary in (train_summary, valid_summary):
        for n_active, count in enumerate(summary.counts_0_to_4):
            rows.append(
                {
                    "split": summary.name,
                    "total_segments": summary.total_segments,
                    "n_active": n_active,
                    "count": count,
                    "fraction": 0.0
                    if summary.total_segments <= 0
                    else float(count) / float(summary.total_segments),
                    "activity_tau": activity_tau,
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote CSV summary to: {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.conf))
    datamodule, data_cfg, tau_from_cfg = build_datamodule(cfg)
    activity_tau = float(args.activity_tau) if args.activity_tau is not None else float(tau_from_cfg)

    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(data_cfg.get("num_workers", 0))
    )

    train_summary = _count_active_speakers(
        dataset=getattr(datamodule, "data_train", None),
        split_name="train",
        activity_tau=activity_tau,
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        max_segments=int(args.max_segments),
    )
    valid_summary = _count_active_speakers(
        dataset=getattr(datamodule, "data_val", None),
        split_name="valid",
        activity_tau=activity_tau,
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        max_segments=int(args.max_segments),
    )

    print_report(
        train_summary=train_summary,
        valid_summary=valid_summary,
        activity_tau=activity_tau,
        data_cfg=data_cfg,
    )

    if args.out_csv is not None:
        write_csv(
            out_path=Path(args.out_csv),
            train_summary=train_summary,
            valid_summary=valid_summary,
            activity_tau=activity_tau,
        )


if __name__ == "__main__":
    main()
