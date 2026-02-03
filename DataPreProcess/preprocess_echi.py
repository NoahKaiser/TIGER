import argparse
import json
import re
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
"""
ECHI preprocessing + JSON manifest generation (device-specific).

What this script does:
- Takes an ECHI dataset root and processes ONE device (--device ha or aria) split-wise (train/dev/eval).
- For each split:
  1) Converts the device mixture/source WAVs (multi-channel sessions) to mono, 16 kHz, 16-bit PCM using SoX
     and stores them in: <output_root>/Processed_ECHI/<device>/<split>/ (flat directory).
     (Skips recomputation if --json_only is set, and skips existing files unless --overwrite is used.)
  2) Optionally fixes small resampling length mismatches (default enabled):
     If ref/<split> exists, each processed mix is cropped/padded at the END to match the sample length of
     <session>.<device>.pos1.wav (only allowed up to --max_len_diff samples, default: 1).
  3) Writes mix.json in the split output directory:
     A list of [absolute_path_to_processed_mix_wav, num_samples].
  4) If ref/<split> exists, writes target_pos1.json ... target_pos4.json (aligned by index to mix.json):
     Each contains [absolute_path_to_ref_target_wav, num_samples] for the selected device and the given position.
     If ref/<split> is missing, target JSONs (and length-fix) are skipped for that split.
     
- Normal run (processing + fix length + json):
   uv run --extra=cpu DataPreProcess/preprocess_echi.py --echi_root /data/public/CHiME9 --output_root /no_backups/s1495 --device ha
- Only rebuild json (still fixes length if ref exists, because it can): --json_only
- Disable the length-fix patch: --no_fix_len_to_ref
- If you ever see larger mismatches and want to allow up to 2 samples difference: --max_len_diff 2
"""

# -------------------------
# Audio processing helpers
# -------------------------
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
    """
    Keep filename but append processing tag:
    train_01.ha.wav -> train_01.ha.16kHz.mono.wav
    dev_02.aria.wav -> dev_02.aria.16kHz.mono.wav
    """
    new_name = f"{in_file.stem}.16kHz.mono{in_file.suffix}"
    return Path(out_root) / new_name


def get_num_samples_wav(wav_path: Path) -> int:
    """Return number of frames in a WAV file (fast: reads header)."""
    with wave.open(str(wav_path), "rb") as wf:
        return int(wf.getnframes())


# -------------------------
# Naming / alignment helpers
# -------------------------
def session_key_from_mix_path(mix_path: Path) -> str:
    """
    Extract session key like 'train_01', 'dev_02', 'eval_03' from a mix filename.
    Works for:
      train_01.ha.wav
      train_01.ha.16kHz.mono.wav
      dev_02.aria.anything.wav
    """
    m = re.search(r"\b(train|dev|eval)_\d+\b", mix_path.name)
    if not m:
        raise ValueError(
            f"Could not extract session key '(train|dev|eval)_XX' from: {mix_path.name}"
        )
    return m.group(0)


# -------------------------
# Length fixing patch
# -------------------------
def force_length_to_reference(
    processed_mix_path: Path,
    ref_path: Path,
    max_len_diff: int = 1,
) -> None:
    """
    Ensure processed mix has exactly the same number of samples as ref.
    For small diffs (<= max_len_diff), fix by cropping/padding at the END.
    Otherwise, raise an error.

    Writes corrected audio back to processed_mix_path as PCM_16.
    """
    processed_mix_path = Path(processed_mix_path)
    ref_path = Path(ref_path)

    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference wav not found: {ref_path}")
    if not processed_mix_path.is_file():
        raise FileNotFoundError(f"Processed mix wav not found: {processed_mix_path}")

    ref_info = sf.info(str(ref_path))
    mix_info = sf.info(str(processed_mix_path))

    ref_len = int(ref_info.frames)
    mix_len = int(mix_info.frames)
    diff = mix_len - ref_len

    if diff == 0:
        return

    if mix_info.samplerate != ref_info.samplerate:
        raise ValueError(
            f"Samplerate mismatch: mix={mix_info.samplerate}, ref={ref_info.samplerate} "
            f"({processed_mix_path.name} vs {ref_path.name})"
        )

    if abs(diff) > max_len_diff:
        raise ValueError(
            f"Length mismatch too large ({diff} samples): "
            f"mix={mix_len}, ref={ref_len} "
            f"for {processed_mix_path.name} vs {ref_path.name}"
        )

    # Load and fix length (should already be mono after sox)
    audio, sr = sf.read(str(processed_mix_path), dtype="float32", always_2d=False)
    if audio.ndim != 1:
        audio = audio[:, 0]

    if mix_len > ref_len:
        audio = audio[:ref_len]  # crop at end
    else:
        pad = ref_len - mix_len
        audio = np.pad(audio, (0, pad), mode="constant", constant_values=0.0)  # pad at end

    sf.write(str(processed_mix_path), audio, sr, subtype="PCM_16")


def fix_all_processed_mixes_to_ref(
    out_split_dir: Path,
    ref_split_dir: Path,
    device: str,
    max_len_diff: int,
) -> None:
    """
    For every processed mix wav in out_split_dir, force its length to the
    reference length of <session>.<device>.pos1.wav found in ref_split_dir.
    """
    out_split_dir = Path(out_split_dir)
    ref_split_dir = Path(ref_split_dir)

    wav_files = sorted(out_split_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No processed mix wav files found in: {out_split_dir}")

    for f in wav_files:
        session = session_key_from_mix_path(f)
        ref_pos1 = ref_split_dir / f"{session}.{device}.pos1.wav"
        force_length_to_reference(f, ref_pos1, max_len_diff=max_len_diff)


# -------------------------
# JSON writers
# -------------------------
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


def index_targets_by_session_and_pos(ref_split_dir: Path, device: str) -> Dict[Tuple[str, int], Path]:
    """
    Build an index:
      (session_key, pos) -> Path to <session>.<device>.pos{pos}.wav
    Example file: train_01.ha.pos3.wav or eval_02.aria.pos1.wav

    We only index targets matching the chosen device and pos1..pos4.
    """
    ref_split_dir = Path(ref_split_dir)
    if not ref_split_dir.is_dir():
        raise NotADirectoryError(f"ref split directory not found: {ref_split_dir}")

    if device not in ("ha", "aria"):
        raise ValueError(f"device must be 'ha' or 'aria', got: {device}")

    index: Dict[Tuple[str, int], Path] = {}
    wav_files = sorted(ref_split_dir.glob("*.wav"))

    pattern = re.compile(
        rf"^(?P<session>(train|dev|eval)_\d+)\.{device}\.pos(?P<pos>[1-4])\.wav$"
    )

    for f in wav_files:
        m = pattern.match(f.name)
        if not m:
            continue

        session = m.group("session")
        pos = int(m.group("pos"))
        key = (session, pos)

        if key in index:
            raise ValueError(f"Duplicate target for {key}: {index[key]} and {f}")

        index[key] = f

    return index


def write_target_pos_jsons(
    ref_split_dir: Path,
    device: str,
    mix_manifest: List[Tuple[str, int]],
    out_json_dir: Path,
) -> None:
    """
    Creates (in out_json_dir):
      target_pos1.json ... target_pos4.json

    Each list is aligned by index with mix_manifest.
    """
    out_json_dir = Path(out_json_dir)
    out_json_dir.mkdir(parents=True, exist_ok=True)

    target_index = index_targets_by_session_and_pos(ref_split_dir=ref_split_dir, device=device)

    mix_paths = [Path(p) for (p, _) in mix_manifest]
    mix_sessions = [session_key_from_mix_path(p) for p in mix_paths]

    for pos in (1, 2, 3, 4):
        entries: List[List[object]] = []

        for session in mix_sessions:
            key = (session, pos)
            if key not in target_index:
                raise FileNotFoundError(
                    f"Missing target for session='{session}', pos{pos}, device='{device}'. "
                    f"Expected file like: {session}.{device}.pos{pos}.wav in {ref_split_dir}"
                )

            tgt_path = target_index[key]
            n = get_num_samples_wav(tgt_path)
            entries.append([str(tgt_path.resolve()), n])

        json_path = out_json_dir / f"target_pos{pos}.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(entries, fp, indent=4, ensure_ascii=False)

        print(f"Wrote target JSON ({len(entries)} entries): {json_path}")


# -------------------------
# Preprocessing
# -------------------------
def preprocess_directory(source_in_dir: Path, source_out_dir: Path, overwrite: bool) -> None:
    """
    Convert all wav files in source_in_dir into source_out_dir (flat directory).
    """
    source_in_dir = Path(source_in_dir)
    source_out_dir = Path(source_out_dir)

    if not source_in_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {source_in_dir}")

    wav_files = sorted(source_in_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in: {source_in_dir}")
        return

    print(f"Found {len(wav_files)} wav files in {source_in_dir}.")
    for i, in_file in enumerate(wav_files, start=1):
        out_file = build_out_path(in_file, source_out_dir)

        if out_file.exists() and not overwrite:
            print(f"[{i}/{len(wav_files)}] skip (exists): {out_file.name}")
            continue

        try:
            sox_4ch_to_mono_16k(in_file, out_file)
            print(f"[{i}/{len(wav_files)}] OK: {in_file.name} -> {out_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(wav_files)}] ERROR: sox failed for {in_file}\n  {e}")


# -------------------------
# Main pipeline for full ECHI root
# -------------------------
def run_for_split(
    echi_root: Path,
    processed_root: Path,
    device: str,
    split: str,
    overwrite: bool,
    json_only: bool,
    fix_len_to_ref: bool,
    max_len_diff: int,
) -> None:
    """
    Process one split (train/dev/eval) for one device.
    Writes:
      processed wavs into: processed_root/device/split/
      JSONs into:          processed_root/device/split/
        - mix.json
        - target_pos1..4.json   (only if ref/<split> exists)
    """
    if split not in ("train", "dev", "eval"):
        raise ValueError(f"split must be one of train/dev/eval, got: {split}")

    source_in_dir = echi_root / device / split
    ref_split_dir = echi_root / "ref" / split

    out_split_dir = processed_root / device / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    mix_json_path = out_split_dir / "mix.json"

    # 1) Create/refresh processed mix files unless json_only
    if json_only:
        if not out_split_dir.is_dir():
            raise NotADirectoryError(f"Processed output directory not found: {out_split_dir}")
    else:
        preprocess_directory(
            source_in_dir=source_in_dir,
            source_out_dir=out_split_dir,
            overwrite=overwrite,
        )

    # 2) Optionally fix processed mix lengths to reference (pos1) BEFORE writing mix.json
    if fix_len_to_ref:
        if not ref_split_dir.is_dir():
            print(
                f"[WARN] Cannot fix mix length to ref for split='{split}' because ref directory is missing: {ref_split_dir}"
            )
        else:
            fix_all_processed_mixes_to_ref(
                out_split_dir=out_split_dir,
                ref_split_dir=ref_split_dir,
                device=device,
                max_len_diff=max_len_diff,
            )

    # 3) Always write mix.json
    mix_manifest = write_manifest_json(source_out_dir=out_split_dir, json_path=mix_json_path)

    # 4) Write targets only if ref/<split> exists; otherwise skip split's targets
    if not ref_split_dir.is_dir():
        print(
            f"[WARN] Skipping target JSONs for split='{split}' because ref directory is missing: {ref_split_dir}"
        )
        return

    write_target_pos_jsons(
        ref_split_dir=ref_split_dir,
        device=device,
        mix_manifest=mix_manifest,
        out_json_dir=out_split_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser("ECHI full-dataset preprocessing + JSON manifests")

    parser.add_argument(
        "--echi_root",
        type=str,
        required=True,
        help="Path to ECHI dataset root directory (contains ha/, aria/, ref/, ...)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Directory where 'Processed_ECHI' will be created/written",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["ha", "aria"],
        help="Which device to process (ha or aria)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite processed mix files if they exist")
    parser.add_argument(
        "--json_only",
        action="store_true",
        help="Skip SoX processing and only (re)create JSONs from already processed mix wavs",
    )

    # Patch flags
    parser.add_argument(
        "--no_fix_len_to_ref",
        action="store_true",
        help="Disable forcing processed mix length to match reference (pos1). Default is enabled.",
    )
    parser.add_argument(
        "--max_len_diff",
        type=int,
        default=1,
        help="Max allowed sample length difference between processed mix and reference before error (default: 1)",
    )

    args = parser.parse_args()

    echi_root = Path(args.echi_root)
    if not echi_root.is_dir():
        raise NotADirectoryError(f"ECHI root directory not found: {echi_root}")

    processed_root = Path(args.output_root) / "Processed_ECHI"
    processed_root.mkdir(parents=True, exist_ok=True)

    fix_len_to_ref = not args.no_fix_len_to_ref

    for split in ("train", "dev", "eval"):
        print(f"\n=== Processing split='{split}' device='{args.device}' ===")
        run_for_split(
            echi_root=echi_root,
            processed_root=processed_root,
            device=args.device,
            split=split,
            overwrite=args.overwrite,
            json_only=args.json_only,
            fix_len_to_ref=fix_len_to_ref,
            max_len_diff=args.max_len_diff,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
