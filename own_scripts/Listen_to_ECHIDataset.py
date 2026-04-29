#!/usr/bin/env python3
"""
Dump a small batch of ECHI segments to WAV so you can listen to them.

Creates N samples (segments) from ECHIDataset and writes:
  out_dir/
    sample_0000_mix.wav
    sample_0000_tgt_pos1.wav
    ...
    sample_0000_tgt_pos4.wav
    sample_0001_mix.wav
    ...

Usage example:
  python dump_echi_batch.py \
    --json_dir /no_backups/s1495/Processed_ECHI/ha/train \
    --out_dir  /no_backups/s1495/debug_audio/batch01 \
    --n 10 --n_src 4 --sr 16000 --segment 3.0 --start_idx 0
"""

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from look2hear.datas.echidatamodule import ECHIDataset


def to_1d_numpy(x: torch.Tensor) -> np.ndarray:
    """Ensure audio is 1D float32 numpy."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(type(x))
    x = x.detach().cpu()
    if x.ndim == 2:
        # If something ended up [T, C], take channel 0 (better: downmix).
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"Expected 1D audio, got shape {tuple(x.shape)}")
    return x.to(torch.float32).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir", required=True, type=str, help="Directory containing mix.json and target_pos*.json")
    p.add_argument("--out_dir", required=True, type=str, help="Where to write WAV files")
    p.add_argument("--n", default=10, type=int, help="Number of segments to dump")
    p.add_argument("--n_src", default=4, type=int, help="Number of targets (pos1..pos4)")
    p.add_argument("--sr", default=16000, type=int, help="Sample rate for saving WAVs")
    p.add_argument("--segment", default=3.0, type=float, help="Segment length in seconds")
    p.add_argument("--hop", default=None, type=float, help="Hop in seconds (default: segment)")
    p.add_argument("--pad_last", action="store_true", help="If set, allow tail padding")
    p.add_argument("--start_idx", default=0, type=int, help="Dataset index to start from")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate dataset
    #ECHIDataset = getattr(look2hear.datas, "ECHIDataset")

    ds = ECHIDataset(
        json_dir=args.json_dir,
        n_src=args.n_src,
        sample_rate=args.sr,
        segment=args.segment,
        hop=args.hop,
        pad_last=args.pad_last,
        return_path=True,
    )

    n = min(args.n, len(ds) - args.start_idx)
    if n <= 0:
        raise RuntimeError(f"Nothing to dump: len(ds)={len(ds)}, start_idx={args.start_idx}")

    print(f"Dataset len={len(ds)} | dumping n={n} starting at idx={args.start_idx}")
    print(f"Writing WAVs to: {out_dir}")

    for i in range(n):
        idx = args.start_idx + i
        mixture, targets, utt_id = ds[idx]

        mix_np = to_1d_numpy(mixture)
        sf.write(out_dir / f"sample_{i:04d}_mix.wav", mix_np, args.sr)

        if targets.numel() == 0:
            # inference-only split
            print(f"[{i:04d}] idx={idx} NO TARGETS | utt_id={utt_id}")
            continue

        if targets.ndim != 2 or targets.shape[0] != args.n_src:
            raise RuntimeError(f"Unexpected targets shape at idx={idx}: {tuple(targets.shape)}")

        for k in range(args.n_src):
            tgt_np = to_1d_numpy(targets[k])
            sf.write(out_dir / f"sample_{i:04d}_tgt_pos{k+1}.wav", tgt_np, args.sr)

        # Optional: also save a quick "sum of targets" for sanity listening
        sum_np = np.sum([to_1d_numpy(targets[k]) for k in range(args.n_src)], axis=0)
        sf.write(out_dir / f"sample_{i:04d}_tgt_sum.wav", sum_np.astype(np.float32), args.sr)

        print(f"[{i:04d}] idx={idx} | utt_id={utt_id}")

    print("Done.")


if __name__ == "__main__":
    main()
