from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch


def build_spk_table_from_pt(
    spk_emb_path: Union[str, Path],
    *,
    sort_ids: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, int], List[str], torch.Tensor]:
    """
    Loads a .pt file containing a Python dict: { "P005": emb[d], ... } and returns:
      - spk2idx: mapping speaker-id -> integer index [0..N-1]
      - spk_ids: list of speaker-ids in table order
      - spk_table: Tensor [N, d] stacked in that order
    """
    spk_emb_path = Path(spk_emb_path)
    if not spk_emb_path.is_file():
        raise FileNotFoundError(f"Speaker embedding file not found: {spk_emb_path}")

    obj = torch.load(str(spk_emb_path), map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise ValueError(
            f"{spk_emb_path} must contain a non-empty dict mapping spk_id->embedding, got: {type(obj)}"
        )

    spk_ids = list(obj.keys())
    if sort_ids:
        spk_ids = sorted(spk_ids)

    # Validate and stack
    first = obj[spk_ids[0]]
    first_t = torch.as_tensor(first)
    if first_t.ndim != 1:
        raise ValueError(f"Embedding for {spk_ids[0]} must be 1D, got shape: {tuple(first_t.shape)}")
    d = int(first_t.shape[0])

    rows = []
    for sid in spk_ids:
        v = torch.as_tensor(obj[sid])
        if v.ndim != 1 or int(v.shape[0]) != d:
            raise ValueError(f"Embedding dim mismatch for {sid}: expected ({d},), got {tuple(v.shape)}")
        rows.append(v.to(dtype=dtype))

    spk_table = torch.stack(rows, dim=0).contiguous()  # [N, d]
    spk2idx = {sid: i for i, sid in enumerate(spk_ids)}
    return spk2idx, spk_ids, spk_table