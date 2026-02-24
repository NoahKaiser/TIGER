#!/usr/bin/env python3
"""
compute_spk_embeddings_ecapa.py

Compute SpeechBrain ECAPA-TDNN speaker embeddings from raw .wav references.

Two input modes:
  1) --mode files:
       - If --in_dir contains wav files directly: embed each wav (non-recursive).
       - If --in_dir contains subfolders (e.g., train/dev/eval): auto-scan recursively for *.wav.
       Output: dict[speaker_id -> embedding]
       (speaker_id is derived from filename stem, e.g., P005 from P005.wav)

  2) --mode speakers:
       --in_dir contains subfolders (one per speaker), each with wav files
       Output: dict[speaker_id -> prototype_embedding]

Embeddings can be computed either from whole utterances (--whole_utt) or by
chunking and averaging chunk embeddings (default).

Example (your participant folder with train/dev/eval):
  uv run --extra=cu118 DataPreProcess/compute_spk_embeddings_ecapa.py \
    --mode files \
    --in_dir /data/public/CHiME9/participant \
    --out_dir /no_backups/s1495/ECHI_spk_embeddings/ECAPA_embeddings \
    --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, DefaultDict
from collections import defaultdict

import torch
import torchaudio
import soundfile as sf

TARGET_SR = 16000


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def load_wav_mono(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Returns waveform as torch.Tensor shape [1, T] and sample_rate.
    Uses torchaudio if possible; falls back to soundfile for WAV.
    """
    try:
        wav, sr = torchaudio.load(str(path))  # [C, T]
        if wav.ndim != 2:
            raise RuntimeError("Unexpected torchaudio waveform shape.")
        wav = wav.mean(dim=0, keepdim=True)  # mono -> [1, T]
        return wav, sr
    except Exception:
        x, sr = sf.read(str(path), always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        wav = torch.from_numpy(x).float().unsqueeze(0)  # [1, T]
        return wav, sr


def resample_if_needed(
    wav: torch.Tensor, sr: int, target_sr: int, resamplers: dict
) -> torch.Tensor:
    """
    wav: [1, T]
    """
    if sr == target_sr:
        return wav
    key = (sr, target_sr)
    if key not in resamplers:
        resamplers[key] = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    return resamplers[key](wav)


def chunk_waveform(wav: torch.Tensor, sr: int, chunk_sec: float, hop_sec: float) -> torch.Tensor:
    """
    wav: [1, T]
    Returns chunks stacked as [N, chunk_T]
    """
    assert wav.ndim == 2 and wav.shape[0] == 1
    T = wav.shape[1]
    chunk_T = int(round(chunk_sec * sr))
    hop_T = int(round(hop_sec * sr))

    if T <= 0:
        raise RuntimeError("Empty waveform.")
    if chunk_T <= 0 or hop_T <= 0:
        raise ValueError("chunk_sec and hop_sec must be > 0.")

    # pad short refs to exactly one chunk
    if T < chunk_T:
        pad = torch.zeros((1, chunk_T - T), dtype=wav.dtype)
        wav = torch.cat([wav, pad], dim=1)
        return wav  # [1, chunk_T] -> treated as one chunk

    chunks = []
    for start in range(0, T - chunk_T + 1, hop_T):
        chunks.append(wav[:, start : start + chunk_T])  # [1, chunk_T]
    if not chunks:
        chunks = [wav[:, :chunk_T]]
    return torch.cat(chunks, dim=0)  # [N, chunk_T]


@torch.inference_mode()
def embed_from_wavs(
    wav_paths: List[Path],
    classifier,
    device: str,
    whole_utt: bool,
    chunk_sec: float,
    hop_sec: float,
    normalize_chunks: bool,
    max_chunk_batch: int,
) -> torch.Tensor:
    """
    Returns a single embedding (prototype) for the provided wav_paths.
    """
    resamplers = {}
    embs = []

    for p in wav_paths:
        wav, sr = load_wav_mono(p)
        wav = resample_if_needed(wav, sr, TARGET_SR, resamplers)

        if whole_utt:
            batches = [wav]  # [1, T]
        else:
            chunks = chunk_waveform(wav, TARGET_SR, chunk_sec, hop_sec)  # [N, chunk_T] or [1, chunk_T]
            if max_chunk_batch > 0 and chunks.shape[0] > max_chunk_batch:
                batches = [
                    chunks[i : i + max_chunk_batch]
                    for i in range(0, chunks.shape[0], max_chunk_batch)
                ]
            else:
                batches = [chunks]

        for batch in batches:
            batch = batch.to(device)
            emb = classifier.encode_batch(batch, wav_lens=None, normalize=False)

            # SpeechBrain versions may return [B, 1, D] or [B, D]; make it [B, D]
            if emb.ndim == 3:
                emb = emb.squeeze(1)
            emb = emb.float().detach().cpu()  # [B, D]

            if normalize_chunks:
                emb = l2_normalize(emb)

            embs.append(emb)

    embs = torch.cat(embs, dim=0)  # [N_total, D]
    proto = embs.mean(dim=0)       # [D]
    proto = l2_normalize(proto)    # final normalization
    return proto


def list_wavs_nonrecursive(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])


def list_wavs_recursive(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.rglob("*.wav") if p.is_file()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["files", "speakers"],
        required=True,
        help="files: embed each wav (auto-recursive if needed). speakers: embed per subfolder speaker_id/",
    )
    ap.add_argument("--in_dir", type=str, required=True, help="Input directory.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument(
        "--wav_list",
        type=str,
        default=None,
        help="Optional text file listing wav paths (one per line). Only used in --mode files.",
    )
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--model_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--whole_utt", action="store_true", help="If set, embed whole utterance (no chunking).")
    ap.add_argument("--chunk_sec", type=float, default=2.0)
    ap.add_argument("--hop_sec", type=float, default=1.0)
    ap.add_argument(
        "--normalize_chunks",
        action="store_true",
        help="If set, L2-normalize each chunk embedding before averaging.",
    )
    ap.add_argument(
        "--max_chunk_batch",
        type=int,
        default=128,
        help="Maximum number of chunks per encoder forward call in chunk mode (0 = all at once).",
    )
    ap.add_argument(
        "--out_name",
        type=str,
        default="ecapa_embeddings.pt",
        help="Output filename for embeddings (default: ecapa_embeddings.pt).",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    # Load pretrained ECAPA model
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(out_dir / "pretrained_ecapa"),
        run_opts={"device": args.device},
    )

    # We may see the same PXXX across splits; accumulate then average.
    accum: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    provenance: DefaultDict[str, List[str]] = defaultdict(list)

    if args.mode == "files":
        if args.wav_list is not None:
            wav_paths: List[Path] = []
            with open(args.wav_list, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        wav_paths.append(Path(line))
            wav_paths = [p for p in wav_paths if p.exists()]
        else:
            # Try non-recursive first; if empty, fallback to recursive (train/dev/eval layout)
            wav_paths = list_wavs_nonrecursive(in_dir)
            if not wav_paths:
                wav_paths = list_wavs_recursive(in_dir)

        if not wav_paths:
            raise RuntimeError(f"No wav files found in: {in_dir} (mode=files).")

        for p in wav_paths:
            spk_id = p.stem  # expects PXXX.wav
            emb = embed_from_wavs(
                [p],
                classifier,
                args.device,
                whole_utt=args.whole_utt,
                chunk_sec=args.chunk_sec,
                hop_sec=args.hop_sec,
                normalize_chunks=args.normalize_chunks,
                max_chunk_batch=args.max_chunk_batch,
            )
            accum[spk_id].append(emb)
            provenance[spk_id].append(str(p))
            print(f"[OK] {spk_id}: D={emb.numel()} from {p}")

    else:  # speakers
        spk_dirs = sorted([p for p in in_dir.iterdir() if p.is_dir()])
        if not spk_dirs:
            raise RuntimeError("No speaker subfolders found (mode=speakers).")

        for spk_dir in spk_dirs:
            wavs = sorted(list(spk_dir.glob("*.wav")))
            if not wavs:
                print(f"[WARN] No wavs in {spk_dir.name}, skipping.")
                continue

            emb = embed_from_wavs(
                wavs,
                classifier,
                args.device,
                whole_utt=args.whole_utt,
                chunk_sec=args.chunk_sec,
                hop_sec=args.hop_sec,
                normalize_chunks=args.normalize_chunks,
                max_chunk_batch=args.max_chunk_batch,
            )
            accum[spk_dir.name].append(emb)
            provenance[spk_dir.name].extend([str(w) for w in wavs])
            print(f"[OK] {spk_dir.name}: D={emb.numel()} from {len(wavs)} wav(s)")

        if not accum:
            raise RuntimeError("No speaker embeddings computed.")

    # Finalize: average duplicates and L2-normalize
    out: Dict[str, torch.Tensor] = {}
    num_dupe = 0
    for spk_id, embs in accum.items():
        if len(embs) > 1:
            num_dupe += 1
        proto = torch.stack(embs, dim=0).mean(dim=0)
        out[spk_id] = l2_normalize(proto)

    # Save outputs
    out_path = out_dir / args.out_name
    torch.save(out, out_path)

    # Metadata
    any_emb = next(iter(out.values()))
    meta = {
        "backend": "speechbrain",
        "model_source": args.model_source,
        "target_sr": TARGET_SR,
        "mode": args.mode,
        "whole_utt": bool(args.whole_utt),
        "chunk_sec": float(args.chunk_sec),
        "hop_sec": float(args.hop_sec),
        "normalize_chunks": bool(args.normalize_chunks),
        "input_root": str(in_dir.resolve()),
        "num_speakers": int(len(out)),
        "num_speakers_with_multiple_refs": int(num_dupe),
        "num_reference_files_total": int(sum(len(v) for v in provenance.values())),
        "embedding_dim": int(any_emb.numel()),
        # Do NOT dump full provenance mapping here (could be large). Add if you really want it.
    }
    meta_path = out_dir / "ecapa_embeddings_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Saved: {meta_path}")
    print(f"Speakers: {len(out)} | Speakers with multiple refs: {num_dupe}")


if __name__ == "__main__":
    main()
