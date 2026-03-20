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
`DataPreProcess/compute_spk_embeddings_ecapa.py` computes speaker embeddings using SpeechBrain ECAPA-TDNN (supervised deep speaker-verification embedding extraction (x-vector family)) and writes a `.pt` dictionary used by the TSE pipeline.

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
  - `{ "P001": tensor(...), "P002": tensor(...), ... }`, where each embedding vector has shape $[D]$
- This matches `build_spk_table_from_pt(...)`, which stacks vectors into $[N, D]$ and builds `spk2idx`.

### CLI summary
- Required:
  - `--mode {files,speakers}`
  - `--in_dir <path>`
  - `--out_dir <path>`
- Common optional:
  - `--device`, `--model_source`, `--whole_utt`, `--chunk_sec`, `--hop_sec`, `--normalize_chunks`, `--out_name`
  - `--wav_list` (only in `files` mode; one wav path per line)

## Implemented Description: `verify_tse_spk_id_alignment.py`
`DataPreProcess/verify_tse_spk_id_alignment.py` verifies that speaker IDs used in TSE target manifests are covered by the ECAPA embedding table.

### Purpose
- Detect whether any speaker IDs in `target_pos*.json` are missing from embedding `.pt` keys.
- Distinguish between:
  - true missing IDs
  - formatting-only mismatches (for example `p24` vs `P024`)

### Inputs and parsing
- Reads `target_pos*.json` from one or more `--split_dir` paths (repeatable flag).
- Expects each target row to be at least:
  - `[target_path, spk_id, ...]`
- Loads `--emb_pt` as a non-empty dict:
  - `spk_id -> embedding_tensor`

### Canonicalization rule
- Uses `canon(...)` to normalize IDs that match `p0*(\d+)` (case-insensitive) into:
  - `P###` (3-digit uppercase, zero-padded)
- Example:
  - `p24`, `P24`, `P024` -> `P024`

### Reported checks
- `target_ids`: unique IDs from manifests.
- `emb_ids`: unique keys from `.pt`.
- `missing_raw`: `target_ids - emb_ids` without normalization.
- `missing_after_canonicalization`: same difference after applying `canon(...)`.
- Also prints:
  - number of manifest files scanned
  - sample embedding keys

### Behavior notes
- Read-only utility: no files are modified.
- Fails fast on missing directories/files or malformed manifest rows.

## Implemented Description: `patch_missing_spk_embeddings_from_targets.py`
`DataPreProcess/patch_missing_spk_embeddings_from_targets.py` fills missing speaker entries in an existing embedding `.pt` by re-embedding target wavs referenced by TSE manifests.

### Purpose
- Patch missing speaker IDs after verifying manifest-vs-embedding mismatch.
- Reuses target reference audio already listed in `target_pos*.json`.

### End-to-end workflow
1. Scan `target_pos*.json` from all provided `--split_dir` paths and build:
   - `spk_id -> set(target_wav_paths)`
2. Load base embeddings (`--emb_pt`) and find:
   - `missing_ids = target_ids - emb_ids`
3. Build temporary per-speaker reference tree:
   - `<work_dir>/refs_by_speaker/<spk_id>/*.wav`
   - implemented as symlinks to original target wavs
4. Run `compute_spk_embeddings_ecapa.py` in `--mode speakers` on that tree to produce missing-only embeddings.
5. Validate all missing IDs were generated.
6. Merge missing embeddings into base dict and write output.

### Temporary reference-tree details
- Symlink names are deterministic (`{index}__{parent}__{name}`) to avoid basename collisions.
- Optional `--max_refs_per_speaker` limits references used per missing speaker (`0` means all).
- Missing target wav paths are counted and reported per speaker.

### Output and write modes
- Missing-only embeddings are written to:
  - `<work_dir>/missing_embeddings/<missing_out_name>`
- Final merged output:
  - default: `<emb_pt_stem>_merged.pt`
  - `--merged_out`: custom path
  - `--inplace`: overwrite `--emb_pt` after writing backup (`--backup_suffix`, default `.bak`)

### Safety and validation behavior
- If no IDs are missing, script exits without changes.
- Refuses merge if newly computed IDs overlap with existing base keys.
- Fails when any missing speaker has zero usable target refs.
- Fails if any missing ID remains unresolved after embedding computation.
- `--dry_run` reports missing IDs and ref availability but skips compute/merge writes.

### ECAPA compute passthrough options
- For missing-ID embedding computation, forwards key options to `compute_spk_embeddings_ecapa.py`:
  - `--device`, `--model_source`
  - `--whole_utt` or chunk mode (`--chunk_sec`, `--hop_sec`, `--normalize_chunks`)
  - `--max_chunk_batch`

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
  - `spk_table`: stacked tensor $[N_{\text{speakers}}, \text{emb\_dim}]$

### 2) What the DataModule uses
- `TSE_ECHIDataModule.setup()` loads the embedding file to get `spk2idx`.
- It passes `spk2idx` into each `TSE_ECHIDataset` (train/val/test).
- The dataset reads `target_pos*.json` entries in this format:
  - `[target_path, spk_id, num_samples]`
- For each row, `spk_id` is converted to `spk_idx` using `spk2idx`.
- `__getitem__` returns:
  - $(\text{mixture}[T], \text{target}[T], \text{spk\_idx}\ (\text{LongTensor scalar}), \text{utt\_id})$
- After collation, batch shape is effectively:
  - $\text{mixtures}[B,T], \text{target}[B,T], \text{spk\_idx}[B], \text{utt\_id}[\text{list}]$

### 3) What the Lightning system uses
- `AudioLightningModuleTSE_ECHI` loads the same `spk_emb_path` (explicit arg or config fallback).
- It rebuilds `spk_table` via `build_spk_table_from_pt(..., sort_ids=True)` and stores it as a registered buffer.
- In `forward(wav, spk_idx)`:
  - `spk_emb = F.embedding(spk_idx, self.spk_table)` gives $[B, \text{emb\_dim}]$
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

## FiLM conditioning (speaker conditioning)
### TIGER_TSE_FiLM1
FiLM (Feature-wise Linear Modulation) uses the target speaker embedding (`spk_emb`) to adapt the internal subband features before separation.  
A small MLP predicts per-feature scale (`gamma`) and shift (`beta`) values, and the features are modulated as:

`subband_feature = gamma * subband_feature + beta`

This helps the model emphasize features that match the target speaker and suppress non-target speech/noise.  
The FiLM layer is initialized close to identity ($\gamma \approx 1$, $\beta \approx 0$) for more stable training.

### TIGER_TSE_FiLM2
`TIGER_TSE_FiLM2` moves FiLM conditioning from an early one-shot feature modulation to an **iterative update modulation** inside the separator (`Recurrent.forward`).

In `TSE_TIGER_FiLM2.forward`, the speaker embedding is mapped once to FiLM parameters:
- `film_params = film_mlp(spk_emb)` -> $[B*nch, 2N]$
- split into `gamma, beta` -> $[B*nch, N]$
- scaled as `gamma = 1 + film_scale * gamma`, `beta = film_scale * beta`
- reshaped for separator broadcasting: `gamma_it, beta_it` -> $[B*nch, N, 1, 1]$

Then `Recurrent.forward(x, gamma, beta)` applies FiLM to the **iteration update** (not directly to the input state):
- input separator state is rearranged to $[B, N, nband, T]$
- each iteration computes:
  - `u_i` (current iteration input; first `x`, then `concat_block(mixture + x)`)
  - `y_i = freq_time_process(u_i)`
  - `delta_i = y_i - u_i`
- FiLM is applied on `delta_i` channel-wise:
  - `delta_i <- gamma * delta_i + beta` (with broadcasting over `nband` and `T`)
- state update:
  - `x <- x + delta_i`

So, in FiLM2, speaker information controls **how much each feature channel update is amplified/suppressed at every recurrent step**, instead of only modulating subband features once before separation.

Implementation note: in the current call site, the separator is invoked with `beta=None`, so conditioning is effectively scale-only (`delta <- gamma * delta`) while retaining the full FiLM interface.


## SpeakerPromptTokenizer

`SpeakerPromptTokenizer` converts a fixed-length speaker embedding into a small set of **speaker prompt tokens** that act as a learnable *memory* for cross-attention conditioning.

### Purpose
Instead of conditioning the separator with a single vector (e.g., concatenation or FiLM), we represent the target speaker as a **token sequence** $spk\_tokens \in \mathbb{R}^{B\times M\times d}$. These tokens can be used as **Keys/Values** in cross-attention, allowing the separator to selectively read speaker information at each time-frequency location.

This is conceptually related to prompt/prefix conditioning in attention models and to affine modulation of learned prompts (in `prompt_mod` mode).  
References: Vaswani et al. (Transformer attention) , Li & Liang (Prefix-Tuning) , Perez et al. (FiLM) 

### I/O
- **Input:** `spk_emb` with shape $(B, D)$ (or $(B, 1, D)$; the singleton dim is removed)
- **Output:** `spk_tokens` with shape $(B, M, d)$

Where:
- `B` = batch size  
- `D` = speaker embedding dimension (e.g., 192 for ECAPA-style embeddings)
- `M` = number of prompt tokens (e.g., 8)
- `d` = token dimension (e.g., 128)

### Modes
#### 1) `mode="linear"`
A single linear projection maps the embedding into `M·d` scalars and reshapes:
$$
\text{tok} = \mathrm{reshape}(W\,\text{spk\_emb}+b) \in \mathbb{R}^{B\times M\times d}.
$$

#### 2) `mode="prompt_mod"`
Learned base prompts are **affinely modulated** by per-sample $(\gamma, \beta)$ predicted from the embedding:
$$
\text{tok} = P\odot(1+\gamma) + \beta,
$$
where $P\in\mathbb{R}^{M\times d}$ is a learned prompt table and $\gamma,\beta\in\mathbb{R}^{B\times M\times d}$.

### Notes
- LayerNorm is applied on the token dimension (`d`) and optional dropout is supported.
- The module validates input rank and embedding dimension to avoid silent mismatches.
- These tokens are typically fed into `MultiHeadCrossAttention2D` as `Keys/Values`.

---

## MultiHeadCrossAttention2D (Speaker–Speech Cross-Attention)

`MultiHeadCrossAttention2D` implements **multi-head cross-attention** that conditions a 2D time–frequency representation on speaker prompt tokens.

### Purpose
Given mixture features $x \in \mathbb{R}^{B\times C\times T\times F}$ and speaker tokens $S \in \mathbb{R}^{B\times M\times D_s}$, the module computes:
- **Queries** from mixture features (`x`)
- **Keys/Values** from speaker tokens (`S`)

This allows each time–frequency position to *selectively read* speaker information from a compact token memory.  
Reference: Vaswani et al. (scaled dot-product attention) 

### I/O
- **Input:**
  - `x`: mixture features $(B, C, T, F)$
  - `spk_tokens`: speaker memory $(B, M, D_s)$
- **Output:** speaker-conditioned features with the same shape as `x`

Where:
- `C` = feature channels
- `T` = time frames
- `F` = frequency bins / subband index (depending on path)
- `M` = number of speaker tokens
- `D_s` = speaker token dimension (must match the tokenizer output)
- `h` = number of attention heads

### Mechanism (per head)
Scaled dot-product cross-attention:
$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
with:
- $Q = W_q\,x$
- $K = W_k\,S$
- $V = W_v\,S$

### 2D handling and factorization
The module **factorizes attention over the last axis** (frequency or time depending on `dim`):

1. Optionally swap the last two axes if `dim==4` so that attention always runs over the internal `T` axis.
2. Reshape into sequences per slice:
   $$
   x \rightarrow x_{\text{seq}} \in \mathbb{R}^{(B\cdot F)\times T\times C}
   $$
3. Compute attention from each slice to the same speaker memory `S` (broadcasted across `F`).
4. Reshape back to $(B, C, T, F)$ and add a residual connection.

### Complexity
Let `L` be the attended axis length (typically `T`) and `M` the number of speaker tokens.
- Self-attention scales as $O(L^2)$
- Cross-attention here scales as $O(L\cdot M)$

With small `M` (e.g., 8), cross-attention is significantly cheaper in the attention matrix than full self-attention. 

### Notes
- Uses `Dropout` on attention weights and on the output projection.
- Adds a residual connection: `out = x + CrossAttn(x, spk_tokens)`.
- In `TSE_TIGER_SelfCross`, this module is applied **after** self-attention in each path, implementing serial self + cross conditioning.

### Usage (Speaker embedding → prompt tokens → cross-attention conditioning)

This section shows the standard wiring used in **TSE_TIGER_SelfCross**:
1) convert a fixed speaker embedding to **prompt tokens** with `SpeakerPromptTokenizer`  
2) use these tokens as **Keys/Values** in `MultiHeadCrossAttention2D` to condition mixture features  
3) apply **serial Self-Attention + Cross-Attention** as done inside the `Recurrent` separator  
4) replicate tokens correctly for **multi-channel** internal batching ($B*nch$).

#### 1) Speaker embedding → speaker prompt tokens

```python
import torch

# Example shapes
B = 2                   # batch size
D_spk = 192             # speaker embedding dimension
M = 8                   # number of prompt tokens
d_tok = 128             # token dimension

spk_emb = torch.randn(B, D_spk)  # (B, D_spk)

tokenizer = SpeakerPromptTokenizer(
    spk_emb_dim=D_spk,
    num_tokens=M,
    token_dim=d_tok,
    mode="linear",       # or "prompt_mod"
    dropout=0.0,
    use_ln=True,
)

spk_tokens = tokenizer(spk_emb)  # (B, M, d_tok)
```
#### 2) Mixture features → speaker-conditioned features (cross-attention)
```python
# Mixture feature map from a separator path (typical layout in TIGER-style blocks)
# x: (B, C, T, F)
C = 128     # feature channels
T = 100     # time frames
F = 16      # frequency bins / subband index
x = torch.randn(B, C, T, F)

cross_attn = MultiHeadCrossAttention2D(
    in_chan=C,
    spk_dim=d_tok,       # must match tokenizer token_dim
    n_head=4,
    hid_chan=4,
    dim=4,               # keep consistent with your Recurrent freq/frame paths
    attn_drop=0.0,
    proj_drop=0.0,
)

x_cond = cross_attn(x, spk_tokens)   # (B, C, T, F)
```

### 3) Serial Self + Cross (as used inside Recurrent)
```python
self_attn = MultiHeadSelfAttention2D(
    in_chan=C,
    n_freqs=1,
    n_head=4,
    hid_chan=4,
    act_type="prelu",
    norm_type="LayerNormalization4D",
    dim=4,
)

x = self_attn(x)                  # mixture self-attention
x = cross_attn(x, spk_tokens)     # speaker-conditioned cross-attention
```
---

## Implemented Description: `look2hear/models/tiger_tse_FiLMCross.py`
`TSE_TIGER_FiLMCross` combines:
- **FiLM1-style early feature modulation** on separator input subband features
- **SelfCross speaker-token cross-attention** inside both frequency and frame paths of the recurrent separator

It is implemented as a subclass of `TSE_TIGER_SelfCross` and reuses the full speaker-token cross-attention pipeline while adding stronger FiLM conditioning.

### Core conditioning flow
Given mixture waveform `input` and speaker embedding `spk_emb`:
1. Build STFT subband features (`subband_feature`) from mixture.
2. Build speaker prompt tokens via inherited `self.spk_tokenizer(spk_emb)`.
3. Build FiLM parameters from `spk_emb` using `self.film_mlp`.
4. Apply **per-band, per-channel FiLM** to separator input features.
5. Run recurrent separator with speaker cross-attention (`self.separator(..., spk_tokens)`).
6. Predict complex masks and reconstruct waveform via iSTFT.

### FiLM parameterization in FiLMCross
Unlike global channel-only FiLM, FiLMCross predicts FiLM parameters for each `(band, feature_channel)`:
- `film_mlp(spk_emb)` -> $[B*nch, 2 * nband * N]$
- reshape to $[B*nch, 2, nband, N]$
- split:
  - `gamma_raw = film_params[:, 0]` -> $[B*nch, nband, N]$
  - `beta_raw = film_params[:, 1]` -> $[B*nch, nband, N]$
- scale:
  - `gamma = 1 + film_scale * gamma_raw`
  - `beta  = film_scale * beta_raw`
- apply (broadcast over time):
  - `film_out = gamma[..., None] * subband_feature + beta[..., None]`

This increases speaker selectivity by allowing different modulation across subbands.

### FiLM activation strategy (non-zero init + gate + warmup)
To avoid FiLM being completely inactive at initialization:
- The final FiLM linear layer uses **tiny non-zero init** on the gamma branch (`film_init_std`) and zero init on beta branch.
- A learnable scalar gate `film_gate_logit` is used:
  - `film_gate = sigmoid(film_gate_logit) * film_warmup`
- FiLM is blended with identity using residual gating:
  - `subband_feature = subband_feature + film_gate * (film_out - subband_feature)`

This provides controlled FiLM activation early in training and prevents abrupt over-conditioning.

### Warmup control from Lightning module
`AudioLightningModuleTSE_ECHI.training_step` updates model warmup each step when supported:
- reads `training.film_warmup_steps`
- computes warmup factor:
  - `warmup = min(1, (global_step + 1) / film_warmup_steps)` (or `1` if disabled)
- calls `audio_model.set_film_warmup(warmup)`

This progressively increases FiLM influence over the first training steps.

### Main constructor knobs
In `TSE_TIGER_FiLMCross`:
- `film_scale`: FiLM modulation amplitude
- `film_hidden`: hidden size of FiLM MLP
- `film_init_std`: tiny gamma-branch init std for FiLM output layer
- `film_gate_init_logit`: initial FiLM gate logit before warmup
- `spk_num_tokens`, `spk_token_dim`, `spk_tokenizer_mode`: speaker prompt token settings for cross-attention

### Expected runtime interfaces
- `forward(input, spk_emb=...)` requires speaker embeddings.
- `spk_emb` accepted shapes:
  - $[D]$
  - $[B, D]$
  - $[B*nch, D]$
- for $[B, D]$ with multi-channel internal batching, embeddings are repeated to match $B*nch$.

### Registration and config usage
- Model class name: `TSE_TIGER_FiLMCross`
- Registered in `look2hear/models/__init__.py`
- Example config entry:
  - `audionet.audionet_name: TSE_TIGER_FiLMCross`

## Implemented Description: Residual-Aware PIT for ECHI (`TSE_TIGER` + `AudioLightningModuleECHI`)

### 1) High-level difference to `TIGER` (no formulas)

Compared to `look2hear/models/tiger.py`, `TSE_TIGER` (`look2hear/models/tiger_tse.py`) keeps the same main separator backbone, but changes the training use case:
- It is used with ECHI-style sparse activity (many segments where some target slots are silent).
- It removes TIGER’s hard per-bin mask partition step (the "force mask sum to 1" block).
- It adds a residual output path by closure (optional return), and `AudioLightningModuleECHI` adds explicit residual supervision on top of speech PIT.

So the key design shift is: less hard constraint inside the mask head, more supervision-driven consistency at waveform level.

### 2) Output mixture constraint (mathematical: what is missing in `TSE_TIGER`)

In baseline `TIGER`, the complex masks are shifted per TF bin so that:

$$
\sum_{k=1}^{K} M^R_{k,f,t}=1,\qquad \sum_{k=1}^{K} M^I_{k,f,t}=0
$$

which implies:

$$
\sum_{k=1}^{K}\hat{S}_{k,f,t}=X_{f,t}
$$

In `TSE_TIGER`, this mask partition block is removed, so the model no longer enforces that TF-bin sum constraint directly.

Instead, residual closure is defined in waveform domain:

$$
\hat{r}(t)=x(t)-\sum_{k=1}^{K_s}\hat{s}_k(t)
$$

and therefore:

$$
x(t)=\sum_{k=1}^{K_s}\hat{s}_k(t)+\hat{r}(t)
$$

This guarantees closure only after defining $\hat{r}$ this way; it is not a hard TF-mask partition constraint like in `TIGER`.

### 3) Loss functions and PIT training from `configs/tiger_on_ECHI.yml`

Training uses `AudioLightningModuleECHI` with two terms:

1. Speech PIT loss (`loss.train`):
   - `PITLossWrapper`
   - base pairwise loss: `pairwise_neg_sisdr`
   - `pit_from: pw_mtx`
   - `perm_reduce: active_mean`
   - `threshold_byloss: false`

2. Residual loss:
   - residual target:
     $$
     r^*(t)=x(t)-\sum_{k=1}^{K_s}s_k^*(t)
     $$
   - normalized MSE:
     $$
     L_{\text{res}}=
     \frac{1}{B'}\sum_{b=1}^{B'}
     \frac{\frac{1}{T}\lVert \hat{r}^{(b)}-r^{*(b)}\rVert_2^2}
     {\frac{1}{T}\lVert x^{(b)}\rVert_2^2+\varepsilon}
     $$

Total training objective:

$$
L_{\text{total}}=L_{\text{speech}}+\lambda_{\text{res}}L_{\text{res}},\qquad
\lambda_{\text{res}}=0.5
$$

For PIT permutation scoring with `active_mean`, target energies
$E_j=\frac{1}{T}\sum_t (s_j^*(t))^2$ define active flags $a_j=\mathbf{1}[E_j>\tau]$ ($\tau=10^{-6}$), and weights:

$$
w_j=\frac{a_j}{\sum_m a_m+\varepsilon},\qquad
C_\pi=\sum_{j=1}^{K_s} w_j\,\ell_{\pi,j}
$$

Validation (`loss.val`) uses `PITLossWrapper` with `pairwise_neg_se_sisdr`.

## Implemented Description: `look2hear/models/tiger_tse2.py` (`TSE_TIGER2`)

### 1) High-level difference to `TIGER` (no formulas)

`TSE_TIGER2` keeps the same TSE-TIGER backbone but changes output design relative to `TIGER`:
- It can predict `K_s` speech outputs plus one explicit residual output (`predict_residual: true` in `tiger_on_ECHI2.yml`).
- It can enforce partition constraints again (`enforce_partition: true`), unlike `TSE_TIGER` where this is removed.
- It is trained with a silence-aware pairwise loss and soft activity-aware PIT reduction, tailored to sparse ECHI activity.

So the key shift vs `TIGER` is not only the residual branch, but also a different sparse-activity PIT strategy in training.

### 2) Output mixture constraint (mathematical: what is different in `TSE_TIGER2`)

Let `K_s` be number of speech outputs and `K=K_s+1` when residual is explicitly predicted.

Output definition:

$$
\text{if }predict\_residual=True:\quad
\hat{s}_k=\hat{y}_k\ (k=1,\dots,K_s),\quad \hat{r}=\hat{y}_{K_s+1}
$$

$$
\text{if }predict\_residual=False:\quad
\hat{s}_k=\hat{y}_k,\quad
\hat{r}=x-\sum_{k=1}^{K_s}\hat{s}_k
$$

With `enforce_partition=True`, masks are shifted per TF bin so:

$$
\sum_{k=1}^{K} M^R_{k,f,t}=1,\qquad
\sum_{k=1}^{K} M^I_{k,f,t}=0
$$

hence:

$$
\sum_{k=1}^{K}\hat{S}_{k,f,t}=X_{f,t}
$$

This is the main difference to `TSE_TIGER`: here, partition can be enforced across all outputs (speech + residual when enabled).

### 3) Loss functions and PIT training from `configs/tiger_on_ECHI2.yml`

Training again uses `AudioLightningModuleECHI` with:

1. Speech PIT loss (`loss.train`):
   - `PITLossWrapper`
   - base pairwise loss: `pairwise_neg_sisdr_silence_aware`
   - `pit_from: pw_mtx`
   - `perm_reduce: active_soft_mean`
   - `threshold_byloss: false`

2. Residual loss: same normalized residual MSE as above, weighted by `lambda_res=0.5`.

For the silence-aware pairwise loss, each estimate-target pair is mixed between active SI-SDR loss and silence penalty:

$$
\ell_{i,j}=a_j\,\ell^{\text{SI-SDR}}_{i,j}+(1-a_j)\,\alpha\,\ell^{\text{sil}}_{i,j}
$$

where $a_j=\sigma\!\left(\beta(\log E_j-\log\tau)\right)$, with config values
$\tau=10^{-6}$, $\beta=8.0$, $\alpha=\text{silence_weight}=0.1$.

For `active_soft_mean` permutation reduction, hard activity flags $h_j=\mathbf{1}[E_j>\tau]$ are softened by $\gamma=0.05$:

$$
w_j=\frac{h_j+\gamma(1-h_j)}{\sum_m \left(h_m+\gamma(1-h_m)\right)+\varepsilon},\qquad
C_\pi=\sum_{j=1}^{K_s} w_j\,\ell_{\pi,j}
$$

Validation uses `PITLossWrapper` with `pairwise_neg_se_sisdr`; model selection in this config monitors `val_total_loss`.

## Implemented Description: `look2hear/losses/matrix.py::PairwiseNegSISDRSilenceAware(_Loss)`

This section documents the exact implementation of `PairwiseNegSISDRSilenceAware` in `look2hear/losses/matrix.py`, including what is done in code and how it is used by TSE-ECHI training.

### 1) Purpose and motivation

`PairwiseNegSISDRSilenceAware` is a pairwise loss for sparse-activity targets:
- when a target speaker is active, it behaves like pairwise negative SI-SDR;
- when a target speaker is silent/inactive, it penalizes residual estimate energy instead of relying on SI-SDR alone.

This avoids weak/unstable supervision for silent target slots and provides a direct "be quiet" objective.

### 2) Class/API surface

- Class: `PairwiseNegSISDRSilenceAware(_Loss)`
- Alias used by configs: `pairwise_neg_sisdr_silence_aware`
  - defined as: `pairwise_neg_sisdr_silence_aware = PairwiseNegSISDRSilenceAware()`
- Input contract in `forward(...)`:
  - `ests`: $[B, K, T]$
  - `targets`: $[B, K, T]$
  - both must have identical shape; otherwise a `TypeError` is raised.
- Output:
  - pairwise loss matrix $[B, K, K]$ with estimate index `i` and target index `j`.

### 3) Constructor parameters (defaults in code)

- `zero_mean=True`: remove DC offset per waveform before loss computation.
- `take_log=True`: active SI-SDR branch is in dB (`-10*log10(...)`).
- `activity_tau=1e-6`: activity threshold reference.
- `activity_beta=8.0`: sigmoid slope in activity gating.
- `silence_weight=0.1`: scale of inactive-target silence penalty.
- `EPS=1e-8`: numerical stability floor.

### 4) Exact forward computation (code-faithful)

Given `ests, targets ∈ R^{B×K×T}`:

1. Optional zero-mean normalization (if `zero_mean=True`):
$$
\tilde{s}=s-\frac{1}{T}\sum_t s(t),\qquad
\tilde{\hat{s}}=\hat{s}-\frac{1}{T}\sum_t \hat{s}(t)
$$

2. Broadcast to all estimate-target pairs:
- `s_target = targets.unsqueeze(1)` -> shape $[B,1,K,T]$
- `s_estimate = ests.unsqueeze(2)` -> shape $[B,K,1,T]$

3. Active-target SI-SDR branch ($\ell^{\text{act}}_{i,j}$):
$$
\text{proj}_{i,j}=
\frac{\langle \hat{s}_i, s_j \rangle}{\|s_j\|_2^2+\varepsilon}\,s_j,\qquad
e_{i,j}=\hat{s}_i-\text{proj}_{i,j}
$$
$$
r_{i,j}=\frac{\|\text{proj}_{i,j}\|_2^2}{\|e_{i,j}\|_2^2+\varepsilon}
$$
If `take_log=True`:
$$
\ell^{\text{act}}_{i,j}=-10\log_{10}(r_{i,j}+\varepsilon)
$$
Else:
$$
\ell^{\text{act}}_{i,j}=-r_{i,j}
$$

4. Silent-target branch ($\ell^{\text{sil}}_{i,j}$):
$$
\ell^{\text{sil}}_{i,j}=\frac{1}{T}\sum_t \hat{s}_i(t)^2
$$
In implementation, this is first $[B,K,1]$ and then expanded to $[B,K,K]$.

5. Soft activity gate from target energy:
$$
E_j=\frac{1}{T}\sum_t s_j(t)^2,\qquad
a_j=\sigma\!\left(\beta\left(\log E_j-\log\tau\right)\right)
$$
with safety:
- `E_j` is clamped to `>= EPS` before log;
- `tau` is floored by `EPS` (`tau=max(tau, EPS)`).

6. Final pairwise loss blend:
$$
\ell_{i,j}=a_j\,\ell^{\text{act}}_{i,j} + (1-a_j)\,\alpha\,\ell^{\text{sil}}_{i,j}
$$
where $\alpha = \text{silence_weight}$.

Return tensor is $L = [\ell_{i,j}] ∈ R^{B×K×K}$.

### 5) Interpretation of the gating behavior

- If target `j` is clearly active (`E_j >> tau`): `a_j -> 1`, so loss is mostly SI-SDR for all estimates against that target.
- If target `j` is near-silent (`E_j << tau`): `a_j -> 0`, so loss emphasizes estimate-energy suppression via `silence_weight * silence_loss`.
- `activity_beta` controls transition sharpness:
  - higher beta: more hard-threshold-like;
  - lower beta: smoother interpolation.

### 6) Runtime overrides from training loop

`forward(...)` allows per-call overrides:
- `activity_tau=...`
- `activity_beta=...`
- `silence_weight=...`

`AudioLightningModuleECHI` passes these from config:
- `training.pit_activity_tau` -> `activity_tau`
- `training.pit_activity_beta` -> `activity_beta`
- `training.pit_silence_weight` -> `silence_weight`

### 7) Relation to PIT reduction (`gamma`)

`pit_activity_gamma` is **not** part of `PairwiseNegSISDRSilenceAware`.

It belongs to permutation reduction (`perm_reduce: active_soft_mean`) in `look2hear/losses/pit_wrapper.py`, where it assigns a non-zero floor weight to inactive targets when averaging per-permutation losses.

So in ECHI2 training:
- this class defines each pairwise entry $\ell_{i,j}$ (local pair loss behavior),
- PIT `active_soft_mean` with `gamma` defines how those entries are aggregated per permutation (global assignment behavior).
