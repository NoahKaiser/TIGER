# TIGER for TSE (Target Speaker Extraction)

## Goal
Implement a supervised **Target Speaker Extraction (TSE)** pipeline in this repository.

## ECHI Dataset Description (Partial, TSE-Relevant)
This is a partial description of ECHI components that are relevant for TSE training.

### 1) Mixture recordings
- `ha`: multi-channel hearing-aid recordings (4 channels; two per ear), multiple sessions, each approx. 30 minutes.
- `aria`: multi-channel smart-glasses recordings (7 channels), multiple sessions, each approx. 30 minutes.
- Each mixture session contains four target speakers, interfering speakers, and cafeteria background noise.

### 2) Clean references
- `ref`: clean mono reference recordings of the target speakers.
- Four target speakers per session.
- Reference duration matches the corresponding mixture session duration.

### 3) Speaker references for embeddings
- `participant`: reference recordings of each speaker reading the "Rainbow" passage.
- Used to compute speaker embeddings with `compute_spk_embeddings_ecapa.py`.

### Split structure
`ha`, `ref`, and `participant` are split into `train`, `dev`, and `eval`:

```text
├── ha
│   ├── train
│   ├── dev
│   └── eval
├── ref
│   ├── train
│   ├── dev
│   └── eval
├── participant
│   ├── train
│   ├── dev
│   └── eval
```

### Naming conventions

#### 1) Mixture names (example: `ha/train/`)
```text
train_01.ha.wav  train_04.ha.wav  train_07.ha.wav  train_10.ha.wav  train_13.ha.wav  train_16.ha.wav  train_19.ha.wav  train_22.ha.wav  train_25.ha.wav  train_28.ha.wav
train_02.ha.wav  train_05.ha.wav  train_08.ha.wav  train_11.ha.wav  train_14.ha.wav  train_17.ha.wav  train_20.ha.wav  train_23.ha.wav  train_26.ha.wav  train_29.ha.wav
train_03.ha.wav  train_06.ha.wav  train_09.ha.wav  train_12.ha.wav  train_15.ha.wav  train_18.ha.wav  train_21.ha.wav  train_24.ha.wav  train_27.ha.wav  train_30.ha.wav
```

#### 2) Reference names (for `ha` and `aria`, example: `ref/train/`)
Speaker-ID links (`PXXX`) point to fixed seating positions (`posX`) per session.

```text
train_01.aria.P005.wav -> train_01.aria.pos1.wav
train_01.aria.P006.wav -> train_01.aria.pos4.wav
train_01.aria.P007.wav -> train_01.aria.pos2.wav
train_01.aria.P008.wav -> train_01.aria.pos3.wav
train_01.aria.pos1.wav
train_01.aria.pos2.wav
train_01.aria.pos3.wav
train_01.aria.pos4.wav

train_01.ha.P005.wav -> train_01.ha.pos1.wav
train_01.ha.P006.wav -> train_01.ha.pos4.wav
train_01.ha.P007.wav -> train_01.ha.pos2.wav
train_01.ha.P008.wav -> train_01.ha.pos3.wav
train_01.ha.pos1.wav
train_01.ha.pos2.wav
train_01.ha.pos3.wav
train_01.ha.pos4.wav
```

#### 3) Participant names (example: `participant/train/`)
```text
P001.wav  P007.wav  P013.wav  P020.wav  P027.wav  P033.wav  P039.wav  P045.wav  P051.wav  P057.wav  P063.wav  P069.wav  P075.wav  P086.wav  P128.wav  P158.wav  P168.wav  P194.wav  P200.wav
P002.wav  P008.wav  P014.wav  P021.wav  P028.wav  P034.wav  P040.wav  P046.wav  P052.wav  P058.wav  P064.wav  P070.wav  P076.wav  P088.wav  P129.wav  P163.wav  P173.wav  P195.wav  P201.wav
P003.wav  P009.wav  P015.wav  P022.wav  P029.wav
```

Reference: https://www.chimechallenge.org/current/task2/data

## Implemented Description: `preprocess_tse_echi.py`
`DataPreProcess/preprocess_tse_echi.py` currently implements the following behavior.

### Scope and CLI behavior
- Processes one selected device per run (`--device ha` or `--device aria`), not only `ha`.
- Iterates all three splits automatically: `train`, `dev`, `eval`.
- Main required arguments:
  - `--echi_root`: ECHI root containing `ha/`, `aria/`, `ref/`, ...
  - `--output_root`: base output path
  - `--device`: `ha` or `aria`
- Optional flags:
  - `--overwrite`: overwrite already processed mix wav files
  - `--json_only`: skip SoX conversion and only regenerate JSON manifests from existing processed wavs
  - `--no_fix_len_to_ref`: disable mix-length fixing to reference
  - `--max_len_diff`: allowed sample mismatch before error during length fixing (default: `1`)

### Output layout
- The script writes into:
  - `<output_root>/Processed_TSE_ECHI/<device>/<split>/`
- Per split, output includes:
  - processed mix wav files (`*.16kHz.mono.wav`)
  - `mix.json`
  - `target_pos1.json` ... `target_pos4.json` (if `ref/<split>` exists)

### Mix preprocessing
- Input mixes are read from `<echi_root>/<device>/<split>/`.
- Each wav is converted with SoX to:
  - mono (`-c 1`)
  - 16-bit PCM (`-b 16`)
  - 16 kHz (`rate -v 16000`)
  - gain normalization with 3 dB headroom (`gain -3`)
- Output filename pattern:
  - `train_01.ha.wav -> train_01.ha.16kHz.mono.wav`

### Length-fix step (enabled by default)
- After conversion and before writing `mix.json`, processed mix length is aligned to:
  - `<echi_root>/ref/<split>/<session>.<device>.pos1.wav`
- If mismatch is within `max_len_diff`, fix is applied by end-cropping or end-padding with zeros.
- If mismatch exceeds `max_len_diff`, the script raises an error.
- This step can be disabled with `--no_fix_len_to_ref`.

### Manifest writing
- `mix.json` is always written from processed wavs in the split directory.
- Entry format:
  - `[abs_mix_path, num_samples]`
- Paths are absolute (`resolve()`), files are sorted, scan is non-recursive.

### Target manifests with speaker IDs
- Target and speaker mappings are built from `<echi_root>/ref/<split>/`.
- Target file index:
  - `<session>.<device>.pos{1..4}.wav -> (session, pos) -> target_path`
- Speaker index from symlinks:
  - `<session>.<device>.P###.wav -> <session>.<device>.posX.wav`
  - parsed as `(session, pos) -> "P###"`
- For each mix session and each position 1..4, the script writes:
  - `[abs_target_path, speaker_id, num_samples]`
- `target_pos1..4.json` are aligned with `mix.json` session order.

### Validation and failure behavior
- Fails on duplicate real target files for the same `(session, pos)`.
- Fails on conflicting speaker IDs for the same `(session, pos)`.
- Fails if any required target or speaker mapping for `(session, pos)` is missing when writing `target_pos*.json`.
- If `ref/<split>` is missing, target JSONs for that split are skipped with a warning, while `mix.json` is still written.
- Symlinks that do not match the expected naming pattern are ignored.

### Compatibility note
- `look2hear/datas/tse_echidatamodule.py` is expected to consume 3-field target entries:
  - `[abs_target_path, spk_id, num_samples]`

## Implemented Description: `compute_spk_embeddings_ecapa.py`
`DataPreProcess/compute_spk_embeddings_ecapa.py` computes speaker embeddings using SpeechBrain ECAPA-TDNN and writes a `.pt` dictionary used by the TSE pipeline.

### Purpose
- Build a mapping:
  - `speaker_id -> embedding_tensor`
- Intended input for:
  - `look2hear/utils/speaker_embedding_utils.py`
  - `look2hear/datas/tse_echidatamodule.py`
  - `look2hear/system/audio_litmodule_tse_echi.py`

### Model backend
- Uses SpeechBrain `EncoderClassifier` from:
  - `speechbrain/spkrec-ecapa-voxceleb` (default, configurable via `--model_source`)
- Pretrained model cache is written to:
  - `<out_dir>/pretrained_ecapa`

### Input modes
- `--mode files`:
  - If wavs exist directly in `--in_dir`, they are read non-recursively.
  - Otherwise, it falls back to recursive scan (`rglob("*.wav")`), which supports `train/dev/eval` folder layouts.
  - Speaker ID is derived from filename stem (`P005.wav -> "P005"`).
  - If the same speaker ID appears multiple times (for example across splits), embeddings are accumulated and averaged.
- `--mode speakers`:
  - Expects subfolders in `--in_dir`, one folder per speaker ID.
  - All `*.wav` in each speaker folder are embedded and averaged to one prototype.

### Audio handling and embedding strategy
- Loads wav via `torchaudio`, with fallback to `soundfile`.
- Converts multi-channel to mono by channel averaging.
- Resamples to 16 kHz if needed.
- Two embedding modes:
  - `--whole_utt`: embed full utterance directly.
  - default chunk mode: split into overlapping chunks (`--chunk_sec`, `--hop_sec`) and average chunk embeddings.
- Optional `--normalize_chunks`:
  - L2-normalize each chunk embedding before averaging.
- Final speaker embedding is always L2-normalized.

### Output files
- Embeddings dictionary (`torch.save`) at:
  - `<out_dir>/<out_name>` (default: `ecapa_embeddings.pt`)
- Metadata JSON at:
  - `<out_dir>/ecapa_embeddings_meta.json`
- Metadata includes:
  - backend/model, target sample rate, mode/chunk settings, number of speakers, embedding dimension, and reference-file counts.

### Output format expected by TSE code
- The `.pt` file contains:
  - `{ "P001": tensor([D]), "P002": tensor([D]), ... }`
- This matches `build_spk_table_from_pt(...)`, which stacks vectors into `[N, D]` and builds `spk2idx`.

### CLI summary
- Required:
  - `--mode {files,speakers}`
  - `--in_dir <path>`
  - `--out_dir <path>`
- Common optional:
  - `--device`, `--model_source`, `--whole_utt`, `--chunk_sec`, `--hop_sec`, `--normalize_chunks`, `--out_name`
  - `--wav_list` (only in `files` mode; one wav path per line)

## Speaker Embedding Flow (Current TSE Runtime)

This section describes how speaker embeddings are made available to the model when using:
- `look2hear/datas/tse_echidatamodule.py`
- `look2hear/system/audio_litmodule_tse_echi.py`
- `look2hear/utils/speaker_embedding_utils.py`

### 1) Embedding file structure
- The speaker embedding file (configured as `spk_emb_path`) is a `.pt` file containing a dictionary:
  - `{ "P005": emb_vector, "P006": emb_vector, ... }`
- `build_spk_table_from_pt(...)` validates this dict, sorts speaker IDs (when `sort_ids=True`), and returns:
  - `spk2idx`: `speaker_id -> integer index`
  - `spk_ids`: ordered list of speaker IDs
  - `spk_table`: stacked tensor `[N_speakers, emb_dim]`

### 2) What the DataModule uses
- `TSE_ECHIDataModule.setup()` loads the embedding file to get `spk2idx`.
- It passes `spk2idx` into each `TSE_ECHIDataset` (train/val/test).
- The dataset reads `target_pos*.json` entries in this format:
  - `[target_path, spk_id, num_samples]`
- For each row, `spk_id` is converted to `spk_idx` using `spk2idx`.
- `__getitem__` returns:
  - `(mixture[T], target[T], spk_idx (LongTensor scalar), utt_id)`
- After collation, batch shape is effectively:
  - `mixtures[B,T], target[B,T], spk_idx[B], utt_id[list]`

### 3) What the Lightning system uses
- `AudioLightningModuleTSE_ECHI` loads the same `spk_emb_path` (explicit arg or config fallback).
- It rebuilds `spk_table` via `build_spk_table_from_pt(..., sort_ids=True)` and stores it as a registered buffer.
- In `forward(wav, spk_idx)`:
  - `spk_emb = F.embedding(spk_idx, self.spk_table)` gives `[B, emb_dim]`
  - then calls `self.audio_model(wav, spk_emb=spk_emb)`

### 4) Why the index mapping stays consistent
- Both data and system build mappings from the same `.pt` file.
- Both use `sort_ids=True`.
- Therefore, the integer `spk_idx` produced by the dataset addresses the same row in `spk_table` inside the Lightning module.

### 5) Current model-consumption status
- The current `look2hear/models/tiger_tse.py` forward signature accepts `spk_emb`:
  - `forward(self, input, spk_emb=None)`
- In the current implementation, `spk_emb` is not yet used internally for conditioning.
- So embeddings are available at the system-model interface, but not yet consumed inside the separator blocks.
