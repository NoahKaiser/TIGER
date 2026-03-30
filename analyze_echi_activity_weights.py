#!/usr/bin/env python3
"""Analyze activity weights used by PairwiseNegSISDRSilenceAware on ECHI targets.

This script mirrors the gate in look2hear.losses.matrix.PairwiseNegSISDRSilenceAware:

    E = mean((target - mean(target))^2, dim=time)
    w = sigmoid(beta * (log(clamp(E, EPS)) - log(max(tau, EPS))))

It loads dataset settings and hyperparameters from a training config (default:
configs/tiger_tse2.yml), iterates the chosen ECHI split, and writes compact
statistics to JSON/CSV so you can verify whether activity gating behaves as
expected for silent-heavy targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from look2hear.datas.echidatamodule import ECHIDataset


EPS = 1e-8


@dataclass
class AnalysisConfig:
    split: str
    json_dir: str
    n_src: int
    sample_rate: int
    segment: float
    hop: float | None
    mixture_mode: str
    tau: float
    beta: float


class RunningStats:
    """Collect scalar samples and expose summary stats."""

    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, arr: torch.Tensor) -> None:
        if arr.numel() == 0:
            return
        self.values.extend(arr.detach().cpu().reshape(-1).to(torch.float64).tolist())

    @property
    def count(self) -> int:
        return len(self.values)

    def as_dict(self, quantiles: Sequence[float]) -> Dict[str, Any]:
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "quantiles": {},
            }
        arr = np.asarray(self.values, dtype=np.float64)
        qvals = np.quantile(arr, quantiles)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "quantiles": {
                f"q{int(round(q * 100)):02d}": float(v) for q, v in zip(quantiles, qvals)
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze activity weights from ECHIDataset targets using the same "
            "formula as PairwiseNegSISDRSilenceAware."
        )
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="configs/tiger_tse2.yml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default=None,
        help="Override split directory from config (must contain mix.json and target_pos*.json).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Override pit_activity_tau from config.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Override pit_activity_beta from config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Num workers for DataLoader.",
    )
    parser.add_argument(
        "--max_segments",
        type=int,
        default=0,
        help="Max number of segments to process (0 means full split).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/activity_weight_analysis",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed.",
    )
    parser.add_argument(
        "--synthetic_check",
        action="store_true",
        help="Run a synthetic sanity check table before dataset analysis.",
    )
    parser.add_argument(
        "--synthetic_only",
        action="store_true",
        help="Run only the synthetic sanity check and skip dataset analysis.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_split_dir(data_cfg: Dict[str, Any], split: str) -> str:
    key = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}[split]
    value = data_cfg.get(key)
    if not value:
        raise ValueError(f"Missing datamodule.data_config.{key} in config.")
    return str(value)


def build_analysis_config(
    cfg: Dict[str, Any],
    split: str,
    json_dir_override: str | None = None,
    tau_override: float | None = None,
    beta_override: float | None = None,
) -> AnalysisConfig:
    data_cfg = cfg.get("datamodule", {}).get("data_config", {})
    train_cfg = cfg.get("training", {})
    json_dir = str(json_dir_override) if json_dir_override else _resolve_split_dir(data_cfg, split=split)
    tau = float(tau_override) if tau_override is not None else float(train_cfg.get("pit_activity_tau", 1e-6))
    beta = float(beta_override) if beta_override is not None else float(train_cfg.get("pit_activity_beta", 8.0))
    return AnalysisConfig(
        split=split,
        json_dir=json_dir,
        n_src=int(data_cfg.get("n_src", 4)),
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        segment=float(data_cfg.get("segment", 3.0)),
        hop=None if data_cfg.get("hop", None) is None else float(data_cfg["hop"]),
        mixture_mode=str(data_cfg.get("mixture_mode", "manifest")),
        tau=tau,
        beta=beta,
    )


def compute_energy_and_activity(
    targets: torch.Tensor, tau: float, beta: float, eps: float = EPS
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicate PairwiseNegSISDRSilenceAware target-energy gate.

    Args:
        targets: [B, K, T]
        tau: activity threshold
        beta: gate slope
        eps: numerical epsilon

    Returns:
        energy: [B, K] mean-square energy after zero-mean
        activity: [B, K] sigmoid gate
    """
    if targets.ndim != 3:
        raise ValueError(f"Expected targets [B,K,T], got {tuple(targets.shape)}")
    tau_eff = max(float(tau), eps)
    targets_zm = targets - targets.mean(dim=2, keepdim=True)
    energy = targets_zm.pow(2).mean(dim=2).clamp(min=eps)
    logits = float(beta) * (torch.log(energy) - math.log(tau_eff))
    activity = torch.sigmoid(logits)
    return energy, activity


def maybe_run_synthetic_check(tau: float, beta: float) -> List[Dict[str, float]]:
    energies = [1e-12, 1e-10, 1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4]
    rows: List[Dict[str, float]] = []
    tau_eff = max(float(tau), EPS)
    for e in energies:
        e_eff = max(float(e), EPS)
        logit = float(beta) * (math.log(e_eff) - math.log(tau_eff))
        w = 1.0 / (1.0 + math.exp(-logit))
        rows.append({"energy": e, "energy_eff": e_eff, "activity_weight": w})
    return rows


def _format_ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def _histogram_dict(values: List[float], bins: Sequence[float]) -> List[Dict[str, Any]]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=np.asarray(bins, dtype=np.float64))
    out: List[Dict[str, Any]] = []
    for i in range(len(counts)):
        out.append(
            {
                "left": float(edges[i]),
                "right": float(edges[i + 1]),
                "count": int(counts[i]),
                "fraction": float(_format_ratio(int(counts[i]), int(arr.size))),
            }
        )
    return out


def _collect_extremes(
    per_example_rows: List[Dict[str, Any]], key: str, n: int = 10
) -> Dict[str, List[Dict[str, Any]]]:
    if not per_example_rows:
        return {"lowest": [], "highest": []}
    sorted_rows = sorted(per_example_rows, key=lambda row: row[key])
    return {"lowest": sorted_rows[:n], "highest": sorted_rows[-n:]}


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_dataset_analysis(
    analysis_cfg: AnalysisConfig,
    batch_size: int,
    num_workers: int,
    max_segments: int,
) -> Dict[str, Any]:
    json_dir = Path(analysis_cfg.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(
            f"json_dir does not exist: {json_dir}. "
            "Provide --json_dir to point to your local processed ECHI split directory."
        )
    mix_manifest = json_dir / "mix.json"
    if not mix_manifest.is_file():
        raise FileNotFoundError(
            f"Missing manifest: {mix_manifest}. "
            "Expected ECHI manifest files produced by preprocessing."
        )

    dataset = ECHIDataset(
        json_dir=json_dir,
        n_src=analysis_cfg.n_src,
        sample_rate=analysis_cfg.sample_rate,
        segment=analysis_cfg.segment,
        hop=analysis_cfg.hop,
        pad_last=False,
        return_path=True,
        mixture_mode=analysis_cfg.mixture_mode,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset has zero segments for the selected split/config.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
    )

    n_total = len(dataset)
    n_target = n_total if max_segments <= 0 else min(max_segments, n_total)

    energy_stats = RunningStats()
    activity_stats = RunningStats()
    energy_by_pos = [RunningStats() for _ in range(analysis_cfg.n_src)]
    activity_by_pos = [RunningStats() for _ in range(analysis_cfg.n_src)]

    n_processed = 0
    n_very_low = 0
    n_middle = 0
    n_very_high = 0
    n_items = 0
    per_example_rows: List[Dict[str, Any]] = []

    for batch in loader:
        _, targets, utt_ids = batch
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        bs = targets.shape[0]
        if n_processed >= n_target:
            break
        keep = min(bs, n_target - n_processed)
        targets = targets[:keep]
        utt_ids = list(utt_ids[:keep])

        energy, activity = compute_energy_and_activity(
            targets=targets, tau=analysis_cfg.tau, beta=analysis_cfg.beta
        )

        energy_stats.add(energy)
        activity_stats.add(activity)
        for pos in range(analysis_cfg.n_src):
            energy_by_pos[pos].add(energy[:, pos])
            activity_by_pos[pos].add(activity[:, pos])

        flat_activity = activity.reshape(-1)
        n_very_low += int((flat_activity < 1e-3).sum().item())
        n_middle += int(((flat_activity >= 0.1) & (flat_activity <= 0.9)).sum().item())
        n_very_high += int((flat_activity > 0.999).sum().item())
        n_items += int(flat_activity.numel())

        mean_energy = energy.mean(dim=1)
        mean_activity = activity.mean(dim=1)
        for i in range(keep):
            per_example_rows.append(
                {
                    "dataset_idx": n_processed + i,
                    "utt_id": str(utt_ids[i]),
                    "mean_energy": float(mean_energy[i].item()),
                    "mean_log10_energy": float(math.log10(max(mean_energy[i].item(), EPS))),
                    "mean_activity": float(mean_activity[i].item()),
                    "min_activity": float(activity[i].min().item()),
                    "max_activity": float(activity[i].max().item()),
                }
            )

        n_processed += keep

    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    summary: Dict[str, Any] = {
        "config": {
            "split": analysis_cfg.split,
            "json_dir": analysis_cfg.json_dir,
            "n_src": analysis_cfg.n_src,
            "sample_rate": analysis_cfg.sample_rate,
            "segment": analysis_cfg.segment,
            "hop": analysis_cfg.hop,
            "mixture_mode": analysis_cfg.mixture_mode,
            "activity_tau": analysis_cfg.tau,
            "activity_beta": analysis_cfg.beta,
            "eps": EPS,
        },
        "dataset": {
            "segments_total_split": n_total,
            "segments_processed": n_processed,
            "pairs_processed": n_items,
        },
        "activity_buckets": {
            "very_low_lt_1e-3": {
                "count": n_very_low,
                "fraction": _format_ratio(n_very_low, n_items),
            },
            "transition_0p1_to_0p9": {
                "count": n_middle,
                "fraction": _format_ratio(n_middle, n_items),
            },
            "very_high_gt_0p999": {
                "count": n_very_high,
                "fraction": _format_ratio(n_very_high, n_items),
            },
        },
        "energy_global": energy_stats.as_dict(quantiles),
        "activity_global": activity_stats.as_dict(quantiles),
        "per_position": [],
        "histograms": {
            "activity_weight": _histogram_dict(
                activity_stats.values, bins=[0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
            ),
            "log10_energy": _histogram_dict(
                [math.log10(max(v, EPS)) for v in energy_stats.values],
                bins=[-8, -7, -6.5, -6, -5.5, -5, -4, -3, -2, -1, 0, 1],
            ),
        },
        "extremes": {
            "by_mean_activity": _collect_extremes(per_example_rows, key="mean_activity", n=10),
            "by_mean_log10_energy": _collect_extremes(
                per_example_rows, key="mean_log10_energy", n=10
            ),
        },
    }

    for pos in range(analysis_cfg.n_src):
        summary["per_position"].append(
            {
                "position": f"pos{pos + 1}",
                "energy": energy_by_pos[pos].as_dict(quantiles),
                "activity": activity_by_pos[pos].as_dict(quantiles),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    conf_path = Path(args.conf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(conf_path)
    analysis_cfg = build_analysis_config(
        cfg=cfg,
        split=args.split,
        json_dir_override=args.json_dir,
        tau_override=args.tau,
        beta_override=args.beta,
    )

    synthetic_rows: List[Dict[str, float]] = []
    if args.synthetic_check or args.synthetic_only:
        synthetic_rows = maybe_run_synthetic_check(
            tau=analysis_cfg.tau, beta=analysis_cfg.beta
        )
        _write_csv(
            out_dir / "synthetic_sanity.csv",
            synthetic_rows,
            fieldnames=["energy", "energy_eff", "activity_weight"],
        )

    if args.synthetic_only:
        payload = {
            "config": {
                "activity_tau": analysis_cfg.tau,
                "activity_beta": analysis_cfg.beta,
                "eps": EPS,
            },
            "synthetic_sanity": synthetic_rows,
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote synthetic-only summary to {out_dir / 'summary.json'}")
        return

    try:
        summary = run_dataset_analysis(
            analysis_cfg=analysis_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_segments=args.max_segments,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    summary["synthetic_sanity"] = synthetic_rows

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    flat_per_pos_rows: List[Dict[str, Any]] = []
    for row in summary["per_position"]:
        flat_per_pos_rows.append(
            {
                "position": row["position"],
                "activity_mean": row["activity"]["mean"],
                "activity_q01": row["activity"]["quantiles"].get("q01"),
                "activity_q50": row["activity"]["quantiles"].get("q50"),
                "activity_q99": row["activity"]["quantiles"].get("q99"),
                "energy_mean": row["energy"]["mean"],
                "energy_q50": row["energy"]["quantiles"].get("q50"),
            }
        )
    _write_csv(
        out_dir / "per_position_summary.csv",
        flat_per_pos_rows,
        fieldnames=[
            "position",
            "activity_mean",
            "activity_q01",
            "activity_q50",
            "activity_q99",
            "energy_mean",
            "energy_q50",
        ],
    )

    hist_rows = []
    for row in summary["histograms"]["activity_weight"]:
        hist_rows.append(
            {
                "metric": "activity_weight",
                "left": row["left"],
                "right": row["right"],
                "count": row["count"],
                "fraction": row["fraction"],
            }
        )
    for row in summary["histograms"]["log10_energy"]:
        hist_rows.append(
            {
                "metric": "log10_energy",
                "left": row["left"],
                "right": row["right"],
                "count": row["count"],
                "fraction": row["fraction"],
            }
        )
    _write_csv(
        out_dir / "histograms.csv",
        hist_rows,
        fieldnames=["metric", "left", "right", "count", "fraction"],
    )

    extreme_rows = []
    for category, side_dict in summary["extremes"].items():
        for side, items in side_dict.items():
            for item in items:
                row = {"category": category, "side": side}
                row.update(item)
                extreme_rows.append(row)
    _write_csv(
        out_dir / "extreme_examples.csv",
        extreme_rows,
        fieldnames=[
            "category",
            "side",
            "dataset_idx",
            "utt_id",
            "mean_energy",
            "mean_log10_energy",
            "mean_activity",
            "min_activity",
            "max_activity",
        ],
    )

    print(f"Wrote summary to {summary_path}")
    print(f"Processed {summary['dataset']['segments_processed']} segments from split='{args.split}'.")
    print(
        "Activity fractions: "
        f"low={summary['activity_buckets']['very_low_lt_1e-3']['fraction']:.4f}, "
        f"mid={summary['activity_buckets']['transition_0p1_to_0p9']['fraction']:.4f}, "
        f"high={summary['activity_buckets']['very_high_gt_0p999']['fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
