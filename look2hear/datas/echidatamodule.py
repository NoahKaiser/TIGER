import os
import json
import numpy as np
from typing import Any, Tuple
import soundfile as sf
import torch
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Dict, Iterable, List, Iterator
from rich import print
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def print_(message: str):
    print(message)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


"ECHIDataset(Dataset)"


class ECHIDataset(Dataset):
    def __init__(
            self,
            json_dir: str = "",
            n_src: int = 2,
            sample_rate: int = 8000,
            segment: float = 4.0,
            normalize_audio: bool = False,
    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src)) #change this for four target speaker
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio

        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate) #gives segment length in samples, segment is in seconds

        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []

        if self.n_src == 2: #make this suitable for 4 targets
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
                )
            )
            self.mix = mix_infos
            self.sources = sources_infos
            self.length = len(self.mix)

    def __len__(self):
        return self.length

    def preprocess_audio_only(self, idx: int):
    """
     change this function so that for each session multiple samples (iterated with idx) are created and not only one
    """
        if self.n_src == 2:  #make this suitable for 4 targets and change indexing, make it iterabel through samples (segments of a all sessions)
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            # Load mixture
            x, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # Load sources
            source_arrays = []
            for src in self.sources:
                s, _ = sf.read(
                    src[idx][0], start=rand_start, stop=stop, dtype="float32"
                )
                source_arrays.append(s)
            sources = torch.from_numpy(np.vstack(source_arrays))
            mixture = torch.from_numpy(x)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            # return mixture, sources, self.mix[idx][0].split("/")[-1]
            return mixture, sources, self.mix[idx][0]

    def __getitem__(self, index: int):
        return self.preprocess_audio_only(index)


class ECHIDataModule(object):
    def __init__(
            self,
            train_dir: str,
            valid_dir: str,
            test_dir: str,
            n_src: int = 2,
            sample_rate: int = 8000,
            segment: float = 4.0,
            normalize_audio: bool = False,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))

        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = ECHIDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
        )
        self.data_val = ECHIDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )
        self.data_test = ECHIDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,
            normalize_audio=self.normalize_audio,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
