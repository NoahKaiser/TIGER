from dataclasses import dataclass
from typing import Optional, Sequence
from torch.utils.data import Subset, DataLoader

@dataclass
class SubsetSpec:
    n_train: Optional[int] = 8
    n_val: Optional[int] = 8
    n_test: Optional[int] = None
    train_indices: Optional[Sequence[int]] = None  # if you want explicit indices
    val_indices: Optional[Sequence[int]] = None
    test_indices: Optional[Sequence[int]] = None

class SubsetDataModule:
    """
    Wraps an existing DataModule-like object (your ECHIDataModule) and replaces
    datasets with torch.utils.data.Subset after setup().
    """
    def __init__(self, base_dm, subset: SubsetSpec, shuffle_train: bool = False):
        self.base = base_dm
        self.subset = subset
        self.shuffle_train = shuffle_train

        # will be set after setup()
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self) -> None:
        self.base.setup()

        def _make_subset(ds, n: Optional[int], idx: Optional[Sequence[int]]):
            if ds is None:
                return None
            if idx is not None:
                return Subset(ds, list(idx))
            if n is None:
                return ds
            n = min(int(n), len(ds))
            return Subset(ds, list(range(n)))

        self.data_train = _make_subset(self.base.data_train, self.subset.n_train, self.subset.train_indices)
        self.data_val   = _make_subset(self.base.data_val,   self.subset.n_val,   self.subset.val_indices)
        self.data_test  = _make_subset(self.base.data_test,  self.subset.n_test,  self.subset.test_indices)

    @property
    def make_loader(self):
        # Recreate loaders using the SAME loader hyperparams as ECHIDataModule
        # (batch_size, num_workers, pin_memory, persistent_workers).
        train_loader = DataLoader(
            self.data_train,
            batch_size=self.base.batch_size,
            shuffle=self.shuffle_train,   # IMPORTANT: avoid shuffle if you want deterministic “same 8”
            num_workers=self.base.num_workers,
            pin_memory=self.base.pin_memory,
            persistent_workers=self.base.persistent_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.data_val,
            batch_size=self.base.batch_size,
            shuffle=False,
            num_workers=self.base.num_workers,
            pin_memory=self.base.pin_memory,
            persistent_workers=self.base.persistent_workers,
        )
        test_loader = DataLoader(
            self.data_test,
            batch_size=self.base.batch_size,
            shuffle=False,
            num_workers=self.base.num_workers,
            pin_memory=self.base.pin_memory,
            persistent_workers=self.base.persistent_workers,
        )
        return train_loader, val_loader, test_loader


