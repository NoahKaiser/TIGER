#!/usr/bin/env python3
"""
Patch missing speaker embeddings using target wavs from TSE manifests.

Workflow:
1) Read target_pos*.json from split dirs and collect (spk_id -> target wav paths).
2) Compare target speaker IDs against keys in base embedding .pt.
3) For missing IDs, build a temporary per-speaker folder with symlinks.
4) Run compute_spk_embeddings_ecapa.py in --mode speakers on that folder.
5) Merge new embeddings into the base dict and save:
   - default: <base_stem>_merged.pt
   - --inplace: overwrite base .pt after writing a backup.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set

import torch


def load_target_map(split_dirs: Iterable[Path]) -> tuple[DefaultDict[str, Set[Path]], int]:
    """Return spk_id -> set(target_wav_paths), and number of target manifest files scanned."""
    out: DefaultDict[str, Set[Path]] = defaultdict(set)
    manifest_files = 0

    for split_dir in split_dirs:
        if not split_dir.is_dir():
            raise NotADirectoryError(f"Split directory not found: {split_dir}")

        json_files = sorted(split_dir.glob("target_pos*.json"))
        if not json_files:
            print(f"[WARN] No target_pos*.json in {split_dir}")

        for jpath in json_files:
            rows = json.loads(jpath.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"{jpath} must contain a JSON list.")

            for row in rows:
                if not (isinstance(row, list) and len(row) >= 2):
                    raise ValueError(
                        f"Invalid row in {jpath}: expected at least [wav_path, spk_id, ...], got {row}"
                    )
                wav_path = Path(str(row[0]))
                spk_id = str(row[1]).strip()
                out[spk_id].add(wav_path)

            manifest_files += 1

    return out, manifest_files


def load_emb_dict(path: Path) -> Dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{path} must contain a non-empty dict mapping spk_id -> embedding.")
    # Normalize keys to str for stable set operations and merge behavior.
    return {str(k): v for k, v in obj.items()}


def build_missing_ref_tree(
    missing_ids: List[str],
    target_map: DefaultDict[str, Set[Path]],
    ref_root: Path,
    max_refs_per_speaker: int,
) -> tuple[Dict[str, int], Dict[str, int]]:
    """
    Creates:
      ref_root/P###/*.wav (symlinks to target wavs)
    Returns:
      linked_count_per_id, skipped_missing_path_per_id
    """
    if ref_root.exists():
        shutil.rmtree(ref_root)
    ref_root.mkdir(parents=True, exist_ok=True)

    linked_count: Dict[str, int] = {}
    skipped_missing: Dict[str, int] = {}

    for sid in missing_ids:
        sid_dir = ref_root / sid
        sid_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(target_map.get(sid, set()))
        if max_refs_per_speaker > 0:
            wavs = wavs[:max_refs_per_speaker]

        linked = 0
        skipped = 0
        for i, wav in enumerate(wavs, start=1):
            if not wav.exists():
                skipped += 1
                continue
            # Keep deterministic unique names for repeated basenames.
            dst = sid_dir / f"{i:04d}__{wav.parent.name}__{wav.name}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(str(wav), str(dst))
            linked += 1

        linked_count[sid] = linked
        skipped_missing[sid] = skipped

    return linked_count, skipped_missing


def run_compute_missing_embeddings(
    compute_script: Path,
    in_dir: Path,
    out_dir: Path,
    out_name: str,
    device: str,
    model_source: str,
    whole_utt: bool,
    chunk_sec: float,
    hop_sec: float,
    normalize_chunks: bool,
) -> Path:
    if not compute_script.is_file():
        raise FileNotFoundError(f"compute_spk_embeddings_ecapa.py not found: {compute_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(compute_script),
        "--mode",
        "speakers",
        "--in_dir",
        str(in_dir),
        "--out_dir",
        str(out_dir),
        "--out_name",
        out_name,
        "--device",
        device,
        "--model_source",
        model_source,
        "--chunk_sec",
        str(chunk_sec),
        "--hop_sec",
        str(hop_sec),
    ]
    if whole_utt:
        cmd.append("--whole_utt")
    if normalize_chunks:
        cmd.append("--normalize_chunks")

    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir / out_name


def save_merged_embeddings(
    base_emb: Dict[str, torch.Tensor],
    add_emb: Dict[str, torch.Tensor],
    base_path: Path,
    merged_out: Path | None,
    inplace: bool,
    backup_suffix: str,
) -> Path:
    overlap = sorted(set(base_emb).intersection(add_emb))
    if overlap:
        raise RuntimeError(
            f"Refusing to overwrite existing speaker IDs in base embeddings. Overlap: {overlap[:20]}"
        )

    merged = dict(base_emb)
    merged.update(add_emb)

    if inplace:
        backup = base_path.with_name(base_path.name + backup_suffix)
        shutil.copy2(base_path, backup)
        torch.save(merged, base_path)
        print(f"[OK] Backup written: {backup}")
        print(f"[OK] Updated in place: {base_path}")
        return base_path

    assert merged_out is not None
    torch.save(merged, merged_out)
    print(f"[OK] Merged file written: {merged_out}")
    return merged_out


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
        help="Split directory containing target_pos*.json. Repeat flag for multiple dirs.",
    )
    parser.add_argument(
        "--emb_pt",
        default="/no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings/ecapa_embeddings.pt",
        help="Base embedding .pt path.",
    )
    parser.add_argument(
        "--compute_script",
        default=str((Path(__file__).resolve().parent / "compute_spk_embeddings_ecapa.py")),
        help="Path to compute_spk_embeddings_ecapa.py.",
    )
    parser.add_argument(
        "--work_dir",
        default="/tmp/echi_missing_spk_patch",
        help="Temporary working directory for symlink refs and missing embeddings.",
    )
    parser.add_argument(
        "--missing_out_name",
        default="ecapa_embeddings_missing_from_targets.pt",
        help="Output filename for missing-only embeddings.",
    )
    parser.add_argument(
        "--merged_out",
        default=None,
        help="Output path for merged embeddings when not using --inplace. Default: <emb_pt stem>_merged.pt",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite --emb_pt after writing a backup file.",
    )
    parser.add_argument(
        "--backup_suffix",
        default=".bak",
        help="Suffix for backup when using --inplace (e.g., .bak).",
    )
    parser.add_argument(
        "--max_refs_per_speaker",
        type=int,
        default=0,
        help="Cap number of target refs per missing speaker (0 = all).",
    )
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device passed to compute_spk_embeddings_ecapa.py.",
    )
    parser.add_argument(
        "--model_source",
        default="speechbrain/spkrec-ecapa-voxceleb",
        help="SpeechBrain model source.",
    )
    parser.add_argument("--whole_utt", action="store_true", help="Pass through to ECAPA compute script.")
    parser.add_argument("--chunk_sec", type=float, default=2.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--normalize_chunks", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report missing IDs and available target refs; do not compute or write outputs.",
    )
    args = parser.parse_args()

    split_dirs = [Path(p) for p in args.split_dir]
    emb_pt = Path(args.emb_pt)
    compute_script = Path(args.compute_script)
    work_dir = Path(args.work_dir)
    ref_root = work_dir / "refs_by_speaker"
    out_dir = work_dir / "missing_embeddings"

    if args.merged_out is None:
        merged_out = emb_pt.with_name(f"{emb_pt.stem}_merged{emb_pt.suffix}")
    else:
        merged_out = Path(args.merged_out)

    target_map, n_manifest_files = load_target_map(split_dirs)
    base_emb = load_emb_dict(emb_pt)

    target_ids = set(target_map.keys())
    emb_ids = set(base_emb.keys())
    missing_ids = sorted(target_ids - emb_ids)

    print(f"split_dirs: {[str(p) for p in split_dirs]}")
    print(f"emb_pt: {emb_pt}")
    print(f"manifest_files_scanned: {n_manifest_files}")
    print(f"target_ids: {len(target_ids)}")
    print(f"emb_ids: {len(emb_ids)}")
    print(f"missing_ids: {len(missing_ids)} {missing_ids}")

    if not missing_ids:
        print("[OK] No missing speaker IDs. Nothing to patch.")
        return

    linked_count, skipped_missing = build_missing_ref_tree(
        missing_ids=missing_ids,
        target_map=target_map,
        ref_root=ref_root,
        max_refs_per_speaker=int(args.max_refs_per_speaker),
    )

    print("\nPer-missing-speaker target refs:")
    for sid in missing_ids:
        print(
            f"  {sid}: linked={linked_count.get(sid, 0)} "
            f"missing_path={skipped_missing.get(sid, 0)}"
        )

    no_ref = [sid for sid in missing_ids if linked_count.get(sid, 0) == 0]
    if no_ref:
        raise RuntimeError(
            f"No usable target wavs found for missing speakers: {no_ref}. "
            "Cannot compute embeddings for these IDs."
        )

    if args.dry_run:
        print("\n[DRY RUN] Stopping before embedding computation and merge.")
        return

    missing_emb_pt = run_compute_missing_embeddings(
        compute_script=compute_script,
        in_dir=ref_root,
        out_dir=out_dir,
        out_name=args.missing_out_name,
        device=args.device,
        model_source=args.model_source,
        whole_utt=bool(args.whole_utt),
        chunk_sec=float(args.chunk_sec),
        hop_sec=float(args.hop_sec),
        normalize_chunks=bool(args.normalize_chunks),
    )

    add_emb = load_emb_dict(missing_emb_pt)
    add_ids = set(add_emb.keys())
    unresolved = sorted(set(missing_ids) - add_ids)
    if unresolved:
        raise RuntimeError(
            f"Missing IDs still unresolved after compute: {unresolved}. "
            f"Computed IDs: {sorted(add_ids)}"
        )

    final_path = save_merged_embeddings(
        base_emb=base_emb,
        add_emb=add_emb,
        base_path=emb_pt,
        merged_out=merged_out,
        inplace=bool(args.inplace),
        backup_suffix=str(args.backup_suffix),
    )

    print("\nSummary:")
    print(f"  missing_ids_patched: {len(missing_ids)}")
    print(f"  missing_emb_file: {missing_emb_pt}")
    print(f"  final_emb_file: {final_path}")
    print(f"  final_num_speakers: {len(base_emb) + len(add_emb)}")


if __name__ == "__main__":
    main()
