#!/usr/bin/env python3
"""
Verify speaker-id alignment between TSE target manifests and ECAPA embeddings.

Checks:
  - target_ids: unique speaker IDs found in target_pos*.json
  - emb_ids: unique keys found in the embedding .pt dict
  - missing_raw: IDs present in target manifests but missing in embedding keys
  - missing_after_canonicalization: same check after normalizing IDs to P###
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Set, Tuple

import torch


def canon(s: str) -> str:
    """Normalize speaker IDs like p24/P24/P024 -> P024."""
    s = str(s).strip()
    m = re.fullmatch(r"(?i)p0*(\d+)", s)
    return f"P{int(m.group(1)):03d}" if m else s


def load_target_ids(split_dirs: Iterable[Path]) -> Tuple[Set[str], int]:
    target_ids: Set[str] = set()
    file_count = 0

    for split_dir in split_dirs:
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Split directory not found: {split_dir}")

        json_files = sorted(split_dir.glob("target_pos*.json"))
        if not json_files:
            print(f"[WARN] No target_pos*.json found in {split_dir}")

        for jpath in json_files:
            rows = json.loads(jpath.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"{jpath} must contain a JSON list.")

            for row in rows:
                if not (isinstance(row, list) and len(row) >= 2):
                    raise ValueError(
                        f"Invalid row in {jpath}: expected at least [path, spk_id, ...], got {row}"
                    )
                target_ids.add(str(row[1]))

            file_count += 1

    return target_ids, file_count


def load_embedding_ids(emb_pt: Path) -> Set[str]:
    if not emb_pt.is_file():
        raise FileNotFoundError(f"Embedding .pt not found: {emb_pt}")

    obj = torch.load(str(emb_pt), map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise ValueError(
            f"{emb_pt} must contain a non-empty dict mapping spk_id -> embedding."
        )
    return {str(k) for k in obj.keys()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_dir",
        action="append",
        default=[
            "/no_backups/s1495/Processed_TSE_ECHI/ha/train",
            "/no_backups/s1495/Processed_TSE_ECHI/ha/dev",
            "/no_backups/s1495/Processed_TSE_ECHI/ha/eval",
        ],
        help=(
            "Split directory containing target_pos*.json. "
            "Repeat flag to provide multiple dirs."
        ),
    )
    parser.add_argument(
        "--emb_pt",
        default="/no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings/ecapa_embeddings.pt",
        help="Path to ECAPA embedding .pt file (dict spk_id -> embedding).",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many IDs to print from each mismatch list.",
    )
    args = parser.parse_args()

    split_dirs = [Path(p) for p in args.split_dir]
    emb_pt = Path(args.emb_pt)

    target_ids, n_manifest_files = load_target_ids(split_dirs)
    emb_ids = load_embedding_ids(emb_pt)

    missing_raw = sorted(target_ids - emb_ids)
    missing_norm = sorted({canon(x) for x in target_ids} - {canon(x) for x in emb_ids})

    print(f"split_dirs: {[str(p) for p in split_dirs]}")
    print(f"emb_pt: {emb_pt}")
    print(f"manifest_files_scanned: {n_manifest_files}")
    print(f"target_ids: {len(target_ids)}")
    print(f"emb_ids: {len(emb_ids)}")
    print(f"missing_raw: {len(missing_raw)} {missing_raw[: args.show]}")
    print(
        "missing_after_canonicalization: "
        f"{len(missing_norm)} {missing_norm[: args.show]}"
    )
    print(f"sample_emb_keys: {sorted(list(emb_ids))[: args.show]}")


if __name__ == "__main__":
    main()
