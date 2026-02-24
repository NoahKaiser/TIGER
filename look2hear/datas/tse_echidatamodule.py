import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from look2hear.utils.speaker_embedding_utils import build_spk_table_from_pt

try:
    from pytorch_lightning import LightningDataModule
except Exception:
    LightningDataModule = object


def session_key_from_path(path: str) -> str:
    m = re.search(r"\b(train|dev|eval)_\d+\b", Path(path).name)
    if not m:
        raise ValueError(f"Could not extract session key from: {path}")
    return m.group(0)


class TSE_ECHIDataset(Dataset):
    """
    TSE dataset (ECHI) using preprocess_tse_echi.py manifests.

    mix.json rows:
      [mix_path, num_samples]

    target_pos{p}.json rows:
      [target_path, spk_id, num_samples]

    Iteration order:
      session0 pos1 seg0.., session0 pos2 seg0.., ... session1 pos1 seg0.., ...

    __getitem__ returns:
      mixture: Tensor[T]
      target:  Tensor[T]
      spk_idx: LongTensor scalar (collates to LongTensor [B])
      utt_id:  str
    """

    def __init__(
        self,
        json_dir: Union[str, Path],
        spk2idx: Dict[str, int],
        n_src: int = 4,
        sample_rate: int = 16000,
        segment: float = 3.0,
        hop: Optional[float] = None,
        pad_last: bool = False,
        dtype: str = "float32",
        utt_id_mode: str = "path",  # "path" or "session"
        unknown_speaker: str = "error",  # "error" or "use_unk"
        unk_idx: int = 0,
    ) -> None:
        super().__init__()
        self.json_dir = Path(json_dir)
        self.spk2idx = dict(spk2idx)
        self.n_src = int(n_src)
        self.sr = int(sample_rate)
        self.seg_len = int(round(float(segment) * self.sr))
        self.hop_len = int(round(float((hop if hop is not None else segment)) * self.sr))
        self.pad_last = bool(pad_last)
        self.dtype = str(dtype)
        self.utt_id_mode = str(utt_id_mode)
        self.unknown_speaker = str(unknown_speaker)
        self.unk_idx = int(unk_idx)

        if self.seg_len <= 0 or self.hop_len <= 0:
            raise ValueError("segment and hop must be > 0.")
        if self.n_src < 1 or self.n_src > 4:
            raise ValueError("ECHI ref positions are pos1..pos4, so n_src must be in [1..4].")
        if self.utt_id_mode not in ("path", "session"):
            raise ValueError("utt_id_mode must be 'path' or 'session'.")
        if self.unknown_speaker not in ("error", "use_unk"):
            raise ValueError("unknown_speaker must be 'error' or 'use_unk'.")

        mix_path = self.json_dir / "mix.json"
        if not mix_path.is_file():
            raise FileNotFoundError(f"Missing mix.json: {mix_path}")

        self.mix: List[Tuple[str, int]] = self._load_mix_manifest(mix_path)  # [(path, L)]
        # targets[pos-1][session_i] = (target_path, spk_idx, L)
        self.targets: List[List[Tuple[str, int, int]]] = []

        for pos in range(1, self.n_src + 1):
            p = self.json_dir / f"target_pos{pos}.json"
            if not p.is_file():
                raise FileNotFoundError(
                    f"Missing target manifest: {p}. Expected target_pos1..target_pos{self.n_src}.json in {self.json_dir}"
                )
            self.targets.append(self._load_target_manifest_with_idx(p))

        # alignment checks
        if any(len(t) != len(self.mix) for t in self.targets):
            raise ValueError(
                f"Target manifests must match mix.json length. mix={len(self.mix)} targets={[len(t) for t in self.targets]}"
            )

        for i in range(len(self.mix)):
            mix_len = int(self.mix[i][1])
            for k in range(self.n_src):
                tgt_len = int(self.targets[k][i][2])
                if tgt_len != mix_len:
                    raise ValueError(
                        f"Length mismatch at session_index={i}: mix={mix_len}, target_pos{k+1}={tgt_len}"
                    )

        # segments per session
        self.nseg_per_session: List[int] = [self._num_segments(int(L)) for _, L in self.mix]

        # prefix sums over session blocks: block_i = n_src * nseg_i
        self.cum_sessions: List[int] = [0]
        for nseg in self.nseg_per_session:
            self.cum_sessions.append(self.cum_sessions[-1] + self.n_src * nseg)

        self.total_items = self.cum_sessions[-1]
        if self.total_items == 0:
            raise RuntimeError("No segments available (segment length may be larger than recordings).")

    @staticmethod
    def _load_mix_manifest(path: Path) -> List[Tuple[str, int]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: List[Tuple[str, int]] = []
        for row in data:
            if not (isinstance(row, list) and len(row) == 2):
                raise ValueError(f"Invalid mix row in {path}: expected [path, num_samples], got: {row}")
            p, n = row
            out.append((str(p), int(n)))
        return out

    def _load_target_manifest_with_idx(self, path: Path) -> List[Tuple[str, int, int]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        out: List[Tuple[str, int, int]] = []
        for row in data:
            if not (isinstance(row, list) and len(row) == 3):
                raise ValueError(f"Invalid target row in {path}: expected [path, spk_id, num_samples], got: {row}")
            tgt_path, spk_id, n = row
            spk_id = str(spk_id)

            if spk_id in self.spk2idx:
                spk_idx = int(self.spk2idx[spk_id])
            else:
                if self.unknown_speaker == "error":
                    raise KeyError(
                        f"Speaker id '{spk_id}' (from {path}) not found in spk2idx mapping."
                    )
                spk_idx = self.unk_idx

            out.append((str(tgt_path), spk_idx, int(n)))
        return out

    def _num_segments(self, L: int) -> int:
        if L < self.seg_len:
            return 1 if self.pad_last else 0
        n_full = 1 + (L - self.seg_len) // self.hop_len
        if not self.pad_last:
            return int(n_full)
        last_start = (n_full - 1) * self.hop_len
        if last_start + self.seg_len < L:
            return int(n_full + 1)
        return int(n_full)

    def __len__(self) -> int:
        return self.total_items

    def _locate(self, idx: int) -> Tuple[int, int, int]:
        # idx -> (session_i, pos(1..n_src), seg_j)
        if idx < 0:
            idx += self.total_items
        if idx < 0 or idx >= self.total_items:
            raise IndexError(idx)

        session_i = bisect_right(self.cum_sessions, idx) - 1
        r = idx - self.cum_sessions[session_i]
        nseg = self.nseg_per_session[session_i]
        if nseg <= 0:
            raise RuntimeError(f"Session {session_i} has zero segments but contributed indices (bug).")

        pos_index = r // nseg     # 0..n_src-1
        seg_j = r % nseg          # 0..nseg-1
        pos = int(pos_index + 1)

        return int(session_i), int(pos), int(seg_j)

    def __getitem__(self, idx: int):
        session_i, pos, seg_j = self._locate(idx)

        mix_path, mix_len = self.mix[session_i]
        mix_len = int(mix_len)

        tgt_path, spk_idx, tgt_len = self.targets[pos - 1][session_i]
        tgt_len = int(tgt_len)
        if tgt_len != mix_len:
            raise RuntimeError(f"Length mismatch at session_i={session_i}, pos={pos}: mix={mix_len}, tgt={tgt_len}")

        start = seg_j * self.hop_len
        stop = start + self.seg_len

        if stop <= mix_len:
            x, _ = sf.read(mix_path, start=start, stop=stop, dtype=self.dtype, always_2d=False)
            y, _ = sf.read(tgt_path, start=start, stop=stop, dtype=self.dtype, always_2d=False)
        else:
            if not self.pad_last:
                raise RuntimeError("Tail segment produced but pad_last=False (indexing bug).")
            x, _ = sf.read(mix_path, start=start, stop=mix_len, dtype=self.dtype, always_2d=False)
            y, _ = sf.read(tgt_path, start=start, stop=mix_len, dtype=self.dtype, always_2d=False)
            pad = stop - mix_len
            x = np.pad(np.asarray(x), (0, pad), mode="constant")
            y = np.pad(np.asarray(y), (0, pad), mode="constant")

        mixture = torch.from_numpy(np.asarray(x)).float()  # [T]
        target = torch.from_numpy(np.asarray(y)).float()   # [T]
        spk_idx_t = torch.tensor(spk_idx, dtype=torch.long)

        base = mix_path if self.utt_id_mode == "path" else session_key_from_path(mix_path)
        utt_id = f"{base}|pos{pos}|seg{seg_j}"

        return mixture, target, spk_idx_t, utt_id


class TSE_ECHIDataModule(LightningDataModule):
    """
    DataModule that loads spk_emb_ecapa.pt to build spk2idx (small), then uses TSE_ECHIDataset.
    """

    def __init__(
        self,
        train_dir: Union[str, Path],
        valid_dir: Union[str, Path],
        test_dir: Union[str, Path],
        spk_emb_path: Union[str, Path],
        has_test_targets: bool = True,
        verify_spk_alignment: bool = True,
        n_src: int = 4,
        sample_rate: int = 16000,
        segment: float = 3.0,
        hop: Optional[float] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        utt_id_mode: str = "path",
    ) -> None:
        super().__init__()
        self.train_dir = str(train_dir)
        self.valid_dir = str(valid_dir)
        self.test_dir = str(test_dir)
        self.spk_emb_path = str(spk_emb_path)
        self.has_test_targets = bool(has_test_targets)
        # Kept for config compatibility; preflight is executed in audio_train_tse.py.
        self.verify_spk_alignment = bool(verify_spk_alignment)

        self.n_src = int(n_src)
        self.sample_rate = int(sample_rate)
        self.segment = float(segment)
        self.hop = hop
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.utt_id_mode = str(utt_id_mode)

        self.spk2idx: Optional[Dict[str, int]] = None
        self.data_train: Optional[TSE_ECHIDataset] = None
        self.data_val: Optional[TSE_ECHIDataset] = None
        self.data_test: Optional[TSE_ECHIDataset] = None

    def _missing_target_manifests(self, json_dir: Union[str, Path]) -> List[Path]:
        base = Path(json_dir)
        missing: List[Path] = []
        for pos in range(1, self.n_src + 1):
            p = base / f"target_pos{pos}.json"
            if not p.is_file():
                missing.append(p)
        return missing

    def _build_dataset(self, split_name: str, json_dir: str, pad_last: bool) -> TSE_ECHIDataset:
        mix_path = Path(json_dir) / "mix.json"
        if not mix_path.is_file():
            raise FileNotFoundError(
                f"{split_name} split is missing mix manifest: {mix_path}"
            )
        missing = self._missing_target_manifests(json_dir)
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"{split_name} split is missing required target manifests for n_src={self.n_src}: {missing_str}"
            )
        return TSE_ECHIDataset(
            json_dir=json_dir,
            spk2idx=self.spk2idx,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            hop=self.hop,
            pad_last=pad_last,
            utt_id_mode=self.utt_id_mode,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        # Build spk2idx (small) from .pt once per process

        spk2idx, _, _ = build_spk_table_from_pt(self.spk_emb_path, sort_ids=True)
        self.spk2idx = spk2idx

        stage = None if stage is None else str(stage).lower()
        if stage in (None, "fit"):
            self.data_train = self._build_dataset("train", self.train_dir, pad_last=False)
            self.data_val = self._build_dataset("valid", self.valid_dir, pad_last=True)

        if stage in (None, "validate"):
            self.data_val = self._build_dataset("valid", self.valid_dir, pad_last=True)

        if stage in (None, "test"):
            if not self.has_test_targets:
                if stage == "test":
                    raise RuntimeError(
                        "Requested setup(stage='test') but has_test_targets=False. "
                        "Set has_test_targets=True and provide test target_pos*.json to run test."
                    )
                self.data_test = None
            else:
                self.data_test = self._build_dataset("test", self.test_dir, pad_last=True)

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None, "Call setup(stage='fit') first."
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None, "Call setup(stage='fit' or 'validate') first."
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None, "Call setup(stage='test') first and ensure test target_pos*.json exist."
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    @property
    def make_loader(self):
        test_loader = self.test_dataloader() if self.data_test is not None else None
        return self.train_dataloader(), self.val_dataloader(), test_loader
