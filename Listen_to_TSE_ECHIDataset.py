#!/usr/bin/env python3
"""
Dump a small batch of TSE_ECHI segments to WAV so you can listen to them.

Creates N samples (segments) from TSE_ECHIDataset and writes:
  out_dir/
    sample_0000_mix.wav
    sample_0000_tgt.wav
    ...

It also writes metadata for the dumped samples:
  out_dir/metadata.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from look2hear.datas.tse_echidatamodule import TSE_ECHIDataset
from look2hear.utils.speaker_embedding_utils import build_spk_table_from_pt


def to_1d_numpy(x: torch.Tensor) -> np.ndarray:
    """Ensure audio is 1D float32 numpy."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(type(x))

    x = x.detach().cpu()
    if x.ndim == 0:
        x = x.view(1)
    elif x.ndim == 2:
        # If something ended up [T, C], take channel 0.
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {tuple(x.shape)}")

    return x.to(torch.float32).numpy()


def parse_pos_seg(utt_id: str):
    """Parse '|posX|segY' suffix produced by TSE_ECHIDataset."""
    m = re.search(r"\|pos(?P<pos>\d+)\|seg(?P<seg>\d+)$", str(utt_id))
    if not m:
        return None, None
    return int(m.group("pos")), int(m.group("seg"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", required=True, type=str, help="Directory containing mix.json and target_pos*.json")
    p.add_argument("--spk_emb_path", required=True, type=str, help="Path to speaker embedding .pt used to build spk2idx")
    p.add_argument("--out_dir", required=True, type=str, help="Where to write WAV files")
    p.add_argument("--n", default=10, type=int, help="Number of segments to dump")
    p.add_argument("--n_src", default=4, type=int, help="Number of target positions available (pos1..pos4)")
    p.add_argument("--sr", default=16000, type=int, help="Sample rate for saving WAVs")
    p.add_argument("--segment", default=3.0, type=float, help="Segment length in seconds")
    p.add_argument("--hop", default=None, type=float, help="Hop in seconds (default: segment)")
    p.add_argument("--pad_last", action="store_true", help="If set, allow tail padding")
    p.add_argument("--start_idx", default=0, type=int, help="Dataset index to start from")
    p.add_argument("--utt_id_mode", default="path", choices=["path", "session"], help="How utt_id base is formed")
    p.add_argument("--unknown_speaker", default="error", choices=["error", "use_unk"], help="How to handle unknown speaker ids")
    p.add_argument("--unk_idx", default=0, type=int, help="Fallback speaker index when --unknown_speaker use_unk")
    p.add_argument(
        "--only_valid_speech_region",
        action="store_true",
        help="If set, sample only from metadata valid-speech regions",
    )
    p.add_argument(
        "--valid_speech_metadata_root",
        default=None,
        type=str,
        help="Metadata root (contains train/dev/eval CSVs); required with --only_valid_speech_region",
    )
    args = p.parse_args()

    if args.only_valid_speech_region and not args.valid_speech_metadata_root:
        raise ValueError(
            "--valid_speech_metadata_root must be provided when --only_valid_speech_region is set."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spk2idx, spk_ids, _ = build_spk_table_from_pt(args.spk_emb_path, sort_ids=True)
    idx2spk = {idx: sid for sid, idx in spk2idx.items()}

    ds = TSE_ECHIDataset(
        json_dir=args.json_dir,
        spk2idx=spk2idx,
        n_src=args.n_src,
        sample_rate=args.sr,
        segment=args.segment,
        hop=args.hop,
        pad_last=args.pad_last,
        utt_id_mode=args.utt_id_mode,
        unknown_speaker=args.unknown_speaker,
        unk_idx=args.unk_idx,
        only_valid_speech_region=args.only_valid_speech_region,
        valid_speech_metadata_root=args.valid_speech_metadata_root,
    )

    n = min(args.n, len(ds) - args.start_idx)
    if n <= 0:
        raise RuntimeError(f"Nothing to dump: len(ds)={len(ds)}, start_idx={args.start_idx}")

    print(f"Loaded speaker table: {len(spk_ids)} ids from {args.spk_emb_path}")
    print(f"Dataset len={len(ds)} | dumping n={n} starting at idx={args.start_idx}")
    print(f"Writing WAVs to: {out_dir}")

    metadata_rows = []

    for i in range(n):
        idx = args.start_idx + i
        mixture, target, spk_idx_t, utt_id = ds[idx]

        mix_np = to_1d_numpy(mixture)
        tgt_np = to_1d_numpy(target)

        sf.write(out_dir / f"sample_{i:04d}_mix.wav", mix_np, args.sr)
        sf.write(out_dir / f"sample_{i:04d}_tgt.wav", tgt_np, args.sr)

        spk_idx = int(spk_idx_t.item()) if isinstance(spk_idx_t, torch.Tensor) else int(spk_idx_t)
        spk_id = idx2spk.get(spk_idx, f"UNK_IDX_{spk_idx}")
        pos, seg = parse_pos_seg(utt_id)

        row = {
            "sample": i,
            "dataset_idx": idx,
            "utt_id": str(utt_id),
            "pos": pos,
            "seg": seg,
            "spk_idx": spk_idx,
            "spk_id": spk_id,
            "mix_wav": f"sample_{i:04d}_mix.wav",
            "tgt_wav": f"sample_{i:04d}_tgt.wav",
        }
        metadata_rows.append(row)

        print(
            f"[{i:04d}] idx={idx} | pos={pos} seg={seg} | "
            f"spk_idx={spk_idx} spk_id={spk_id} | utt_id={utt_id}"
        )

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_rows, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
