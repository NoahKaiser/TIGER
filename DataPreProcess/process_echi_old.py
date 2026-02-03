import argparse
import subprocess
from pathlib import Path


import json
import os
import soundfile as sf
import resampy
from tqdm import tqdm
from rich import print

'''
This skript preprocesses the ECHI dataset to prepare it for the use with TIGER.
ECHI should have the following structure:

├── aria
│   ├── train
│   ├── dev
│   └── eval
├── ha
│   ├── train
│   ├── dev
│   └── eval
├── ct
│   ├── train
│   ├── dev
│   └── eval
├── ref
│   ├── train
│   ├── dev
│   └── eval
├── tracker
│   ├── train
│   ├── dev
│   └── eval
├── participant
│   ├── train
│   ├── dev
│   └── eval
└─── metadata
    ├── train/ref
    ├── dev/ref
    └── eval/ref


In a first iteration of this skript only ha/train and ref/train of the Dataset is used
Multi-Channel recordings ha/train are converted to mono, ref/train are already mono

ToDo:

- from multichannel mixture recordings (ha) to mono by adding the four channels
- from 48kHz to 16kHz downsampeling + adjusting bit depth to 16-Bit PCM
- create json files mix.json, with full path

example of using it: 

- converting(16kH, mono) + create json file: uv run DataPreProcess/process_echi_old.py --source_in_dir /data/public/CHiME9/ha/train/ --source_out_dir /no_backups/s1495/Resampled_ECHI/ha/train
- creating only json files if converted files already in out_dir: uv run --ectra=cpu process_echi_old.py --source_out_dir /path/out --json_only

data/public/CHiME9/ref/train
'''


import argparse
import json
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Tuple
import re


def sox_4ch_to_mono_16k(in_wav_path: Path, out_wav_path: Path) -> None:
    """Convert wav to mono 16 kHz, 16-bit PCM, normalize with headroom of 3 dB."""
    in_wav_path = Path(in_wav_path)
    out_wav_path = Path(out_wav_path)

    if not in_wav_path.is_file():
        raise FileNotFoundError(f"Input wav not found: {in_wav_path}")

    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "sox",
        str(in_wav_path),
        "-c",
        "1",
        "-b",
        "16",
        str(out_wav_path),
        "gain",
        "-3",
        "rate",
        "-v",
        "16000",
    ]
    subprocess.run(cmd, check=True)


def build_out_path(in_file: Path, out_root: Path) -> Path:
    """file01.wav -> file01.16kHz.mono.wav (written into out_root)"""
    new_name = f"{in_file.stem}.16kHz.mono{in_file.suffix}"
    return out_root / new_name


def get_num_samples_wav(wav_path: Path) -> int:
    """Return number of frames in a WAV file (fast: reads header)."""
    with wave.open(str(wav_path), "rb") as wf:
        return int(wf.getnframes())


def session_key_from_mix_path(mix_path: Path) -> str:
    """
    Extract session key like 'train_01' from a mix filename.
    Works for:
      train_01.wav
      train_01.16kHz.mono.wav
      train_01.anything.with.dots.wav
    """
    m = re.search(r"(train_\d+)", mix_path.name)
    if not m:
        raise ValueError(f"Could not extract session key 'train_XX' from: {mix_path.name}")
    return m.group(1)


def write_manifest_json(source_out_dir: Path, json_path: Path) -> List[Tuple[str, int]]:
    """
    Create mix JSON:
    [
      ["/abs/path/file.wav", 96000],
      ...
    ]
    Only scans *.wav directly inside source_out_dir (non-recursive).

    Returns the manifest as a list of (path, num_samples).
    """
    source_out_dir = Path(source_out_dir)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(source_out_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {source_out_dir}")

    manifest: List[Tuple[str, int]] = []
    for f in wav_files:
        n = get_num_samples_wav(f)
        manifest.append((str(f.resolve()), n))

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump([list(x) for x in manifest], fp, indent=4, ensure_ascii=False)

    print(f"Wrote JSON manifest with {len(manifest)} entries: {json_path}")
    return manifest


def index_targets_by_session_and_pos(targets_in_dir: Path) -> Dict[Tuple[str, int], Path]:
    """
    Build an index:
      (session_key, pos) -> Path to sessionXX.ha.pos{pos}.wav

    Accepts only files containing '.ha.pos{pos}.' and ending with '.wav'.
    Ignores aria automatically since it won't match '.ha.'.
    """
    targets_in_dir = Path(targets_in_dir)
    if not targets_in_dir.is_dir():
        raise NotADirectoryError(f"targets_in_dir not found: {targets_in_dir}")

    index: Dict[Tuple[str, int], Path] = {}
    wav_files = sorted(targets_in_dir.glob("*.wav"))

    for f in wav_files:
        name = f.name  # e.g. session01.ha.pos3.wav
        if ".ha.pos" not in name:
            continue
        # Parse: <session> . ha . pos<k> . wav
        parts = name.split(".")
        if len(parts) != 4:
            # Not exactly session.ha.posX.wav
            continue
        session, ha_tag, pos_tag, ext = parts
        if ha_tag != "ha":
            continue
        if not pos_tag.startswith("pos"):
            continue
        try:
            pos = int(pos_tag.replace("pos", ""))
        except ValueError:
            continue
        if pos not in (1, 2, 3, 4):
            continue

        key = (session, pos)
        if key in index:
            raise ValueError(f"Duplicate target for {key}: {index[key]} and {f}")
        index[key] = f

    return index


def write_target_pos_jsons(
    targets_in_dir: Path,
    mix_manifest: List[Tuple[str, int]],
    out_json_dir: Path,
) -> None:
    """
    Creates:
      target_pos1.json ... target_pos4.json

    Each list is aligned by index with mix_manifest.
    We enforce this by iterating sessions in mix order and picking the matching target file.
    """
    out_json_dir = Path(out_json_dir)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    target_index = index_targets_by_session_and_pos(targets_in_dir)

    # Derive mix session order from mix_manifest paths
    mix_paths = [Path(p) for (p, _) in mix_manifest]
    mix_sessions = [session_key_from_mix_path(p) for p in mix_paths]

    for pos in (1, 2, 3, 4):
        entries: List[List[object]] = []

        for i, session in enumerate(mix_sessions):
            key = (session, pos)
            if key not in target_index:
                raise FileNotFoundError(
                    f"Missing target for session='{session}', pos{pos}. "
                    f"Expected file like: {session}.ha.pos{pos}.wav in {targets_in_dir}"
                )

            tgt_path = target_index[key]
            n = get_num_samples_wav(tgt_path)
            entries.append([str(tgt_path.resolve()), n])

        json_path = out_json_dir / f"target_pos{pos}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(entries, fp, indent=4, ensure_ascii=False)

        print(f"Wrote target JSON ({len(entries)} entries): {json_path}")


def preprocess_directory(source_in_dir: Path, source_out_dir: Path, overwrite: bool) -> None:
    source_in_dir = Path(source_in_dir)
    source_out_dir = Path(source_out_dir)

    if not source_in_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {source_in_dir}")

    wav_files = sorted(source_in_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in: {source_in_dir}")
        return

    print(f"Found {len(wav_files)} wav files.")
    for i, in_file in enumerate(wav_files, start=1):
        out_file = build_out_path(in_file, source_out_dir)

        if out_file.exists() and not overwrite:
            print(f"[{i}/{len(wav_files)}] skip (exists): {out_file}")
            continue

        try:
            sox_4ch_to_mono_16k(in_file, out_file)
            print(f"[{i}/{len(wav_files)}] OK: {in_file.name} -> {out_file}")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(wav_files)}] ERROR: sox failed for {in_file}\n  {e}")


def main() -> None:
    parser = argparse.ArgumentParser("ECHI audio preprocessing + JSON manifests")

    # MIX/SOURCES
    parser.add_argument(
        "--source_in_dir",
        type=str,
        help="Input directory of raw source wav files (needed if not --json_only)",
    )
    parser.add_argument(
        "--source_out_dir",
        type=str,
        required=True,
        help="Output directory containing processed mix wav files (flat directory)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite processed mix files if they exist")
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Skip mix processing and only create mix.json from existing wav files in source_out_dir",
    )
    parser.add_argument(
        "--mix_json_out",
        type=str,
        default=None,
        help="Path for mix JSON (default: <source_out_dir>/mix.json)",
    )

    # TARGETS
    parser.add_argument(
        "--targets_in_dir",
        type=str,
        required=True,
        help="Directory containing target wav files (flat directory)",
    )
    parser.add_argument(
        "--targets_json_out_dir",
        type=str,
        default=None,
        help="Directory to write target_pos*.json files (default: <targets_in_dir>)",
    )

    args = parser.parse_args()

    source_out_dir = Path(args.source_out_dir)
    mix_json_path = Path(args.mix_json_out) if args.mix_json_out else (source_out_dir / "mix.json")

    # 1) Create/refresh processed mix files unless json_only
    if args.json_only:
        if not source_out_dir.is_dir():
            raise NotADirectoryError(f"source_out_dir not found: {source_out_dir}")
    else:
        if args.source_in_dir is None:
            parser.error("--source_in_dir is required unless --json_only is set")
        preprocess_directory(
            source_in_dir=Path(args.source_in_dir),
            source_out_dir=source_out_dir,
            overwrite=args.overwrite,
        )

    # 2) Write mix.json and keep manifest in memory for alignment
    mix_manifest = write_manifest_json(source_out_dir=source_out_dir, json_path=mix_json_path)

    # 3) Write target_pos1..4.json aligned to mix order
    targets_json_out_dir = Path(args.targets_json_out_dir) if args.targets_json_out_dir else Path(args.source_out_dir)
    write_target_pos_jsons(
        targets_in_dir=Path(args.targets_in_dir),
        mix_manifest=mix_manifest,
        out_json_dir=targets_json_out_dir,
    )


if __name__ == "__main__":
    main()
