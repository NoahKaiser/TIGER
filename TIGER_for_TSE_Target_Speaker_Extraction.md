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

## Modification Description: `preprocess_tse_echi.py`
`DataPreProcess/preprocess_tse_echi.py` should be adapted from the current copy of `preprocess_echi.py` with the following behavior.

### Target behavior
- Keep preprocessing flow identical to `preprocess_echi.py` for the `ha` device:
  - same mixture conversion (SoX -> mono 16 kHz),
  - same optional length-fix logic,
  - same `mix.json` format and content.
- Ignore `aria` for TSE preprocessing.
  - Use only `ha` data exactly as in current `preprocess_echi.py` when run with `--device ha`.
- Extend target manifests with target speaker IDs from symbolic links (`PXXX -> posX` mapping for `ha`):
  - `target_pos1.json`, `target_pos2.json`, `target_pos3.json`, `target_pos4.json`
  - each entry must contain: absolute target path, target speaker ID (`PXXX`), number of samples.

### Required output format
- `mix.json` (unchanged):
  - `[abs_mix_path, num_samples]`
- `target_pos*.json` (extended):
  - `[abs_target_path, target_speaker_id, num_samples]`
  - where `target_speaker_id` is a string like `"P005"`.

Example (`target_pos1.json`):

```json
["/misc/data/public/CHiME9/ref/train/train_01.ha.pos1.wav", "P005", 35036500]
```

### How to extract `PXXX` from symlinks
- For each split (`train`, `dev`, `eval`), build a mapping for `ha` only:
  - parse symbolic links like:
    - `train_01.ha.P005.wav -> train_01.ha.pos1.wav`
  - derive mapping key/value:
    - key: `(session="train_01", pos=1)`
    - value: `"P005"`
- Then, when writing `target_pos{1..4}.json`, attach the mapped speaker ID for each `(session, pos)`.

### Suggested script changes
- Keep existing core functions for audio processing and `mix.json`.
- Modify/extend target-json writing path only:
  1. Add helper to parse `ha` symlink mappings into `(session, pos) -> PXXX`.
  2. Update target manifest writer to emit 3-field entries:
     - `[target_path, speaker_id, num_samples]`.
  3. Add strict checks:
     - fail if `(session, pos)` has no speaker ID mapping,
     - fail on duplicate mappings,
     - fail if symlink target does not match expected `*.ha.pos{1..4}.wav`.

### Compatibility note
- `tse_echidatamodule.py` must be updated accordingly to read the new 3-field target entries.
- Existing non-TSE pipelines can continue using `preprocess_echi.py` and the old 2-field target format.
