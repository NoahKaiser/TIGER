#!/usr/bin/env python3
"""
STFT -> iSTFT roundtrip for a .wav file using PyTorch (causal configuration).
Reads input.wav, computes STFT, reconstructs with iSTFT, and writes output.wav.

Notes on causality:
- The script uses center=False for both STFT and iSTFT, meaning frames do not
  access future samples (no look-ahead). Each frame depends only on the current
  sample and its past.

Usage:
  python stft_Pytorch_Test.py --in input.wav --out output_recon.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="Path to input .wav")
    p.add_argument("--out", dest="out", required=True, help="Path to output .wav")
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=None, help="Default: n_fft//4")
    p.add_argument("--win_length", type=int, default=None, help="Default: n_fft")
    p.add_argument("--mono", action="store_true", help="Convert to mono by averaging channels")
    args = p.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Load WAV: waveform shape (channels, time)
    waveform, sr = torchaudio.load(str(in_path))

    if args.mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Use float32
    waveform = waveform.to(torch.float32)

    n_fft = args.n_fft
    hop_length = args.hop_length if args.hop_length is not None else n_fft // 4
    win_length = args.win_length if args.win_length is not None else n_fft

    # basic safety checks
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")
    if hop_length > win_length:
        print(f"[Warning] hop_length ({hop_length}) > win_length ({win_length}); overlap-add may be sparse.")
    # Enforce causal framing: no look-ahead
    center = False

    # Right-pad with zeros to ensure full coverage at the tail (causal-safe):
    # With center=False, STFT produces only frames that fit fully into the signal.
    # If the tail isn't covered up to the original length, iSTFT with a forced length
    # would fail due to zero overlap-add energy at the very end.
    T = waveform.shape[-1]
    a = T - win_length
    if a > 0:
        pad_right = (hop_length - (a % hop_length)) % hop_length
    else:
        pad_right = 0
    if pad_right > 0:
        waveform_proc = F.pad(waveform, (0, pad_right))
    else:
        waveform_proc = waveform

    # Window must be on same device/dtype as waveform
    window = torch.hann_window(win_length, device=waveform_proc.device, dtype=waveform_proc.dtype)

    # STFT per channel: input for torch.stft is (time) or (batch, time)
    # We'll treat channels as batch: (C, T)
    X = torch.stft(
        waveform_proc,          # (C, T') possibly right-padded
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,          # causal: no future samples
        normalized=False,
        onesided=True,
        return_complex=True
    )

    # iSTFT reconstruction
    recon = torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        # Reconstruct to the padded length (fully covered), then crop back
        length=waveform_proc.shape[-1]
    )  # shape (C, T)

    # Crop to original (pre-pad) length to preserve exact duration
    recon = recon[..., :T]

    # Optional: avoid clipping when saving PCM wav
    recon = recon.clamp(-1.0, 1.0)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), recon.cpu(), sample_rate=sr)

    print("Saved:", out_path)
    print("Input shape:", tuple(waveform.shape), "SR:", sr)
    if pad_right > 0:
        print(f"Applied right zero-padding for causality-safe coverage: {pad_right} samples")
        print("Padded input shape:", tuple(waveform_proc.shape))
    print("STFT shape:", tuple(X.shape), "(C, F, Frames)")
    print("Recon shape:", tuple(recon.shape))


if __name__ == "__main__":
    main()
