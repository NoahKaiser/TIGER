#!/usr/bin/env python3
"""
inspect_ecapa_embeddings.py

Loads embeddings produced by compute_spk_embeddings_ecapa.py and:
  1) prints basic metadata and tensor shapes
  2) runs cosine similarity checks

Usage:
  python inspect_ecapa_embeddings.py --emb_pt /path/to/ecapa_embeddings.pt
  python inspect_ecapa_embeddings.py --emb_pt /path/to/ecapa_embeddings.pt --meta_json /path/to/ecapa_embeddings_meta.json
  python inspect_ecapa_embeddings.py --emb_pt /path/to/ecapa_embeddings.pt --id1 P01 --id2 P02
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_pt", type=str, required=True, help="Path to ecapa_embeddings.pt")
    ap.add_argument("--meta_json", type=str, default=None, help="Optional path to ecapa_embeddings_meta.json")
    ap.add_argument("--id1", type=str, default=None, help="First speaker/file id (dict key), e.g., P01")
    ap.add_argument("--id2", type=str, default=None, help="Second speaker/file id (dict key), e.g., P02")
    ap.add_argument("--topk", type=int, default=5, help="Show top-k nearest neighbors for id1 (cosine sim).")
    args = ap.parse_args()

    emb_path = Path(args.emb_pt)
    if not emb_path.exists():
        raise FileNotFoundError(f"emb_pt not found: {emb_path}")

    spk_emb = torch.load(str(emb_path), map_location="cpu")
    if not isinstance(spk_emb, dict) or len(spk_emb) == 0:
        raise RuntimeError("Loaded object is not a non-empty dict. Did you pass the correct .pt file?")

    # Optional metadata
    if args.meta_json is not None:
        meta_path = Path(args.meta_json)
        if not meta_path.exists():
            raise FileNotFoundError(f"meta_json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = None

    keys = sorted(spk_emb.keys())
    first_key = keys[0]
    first_emb = spk_emb[first_key]

    print("=== Embedding file inspection ===")
    print(f"Path: {emb_path}")
    print(f"Num items: {len(spk_emb)}")
    print(f"Keys (first 10): {keys[:10]}")
    print(f"Example key: {first_key}")
    print(f"Example embedding dtype: {first_emb.dtype}")
    print(f"Example embedding shape: {tuple(first_emb.shape)}")

    if meta is not None:
        print("\n=== Metadata ===")
        print(json.dumps(meta, indent=2))

    # Basic validation: all embeddings same dim and 1-D
    dims = set()
    bad = []
    for k, v in spk_emb.items():
        if not torch.is_tensor(v):
            bad.append((k, "not a tensor"))
            continue
        if v.ndim != 1:
            bad.append((k, f"ndim={v.ndim}"))
            continue
        dims.add(int(v.numel()))
    if bad:
        print("\n[WARN] Some entries are not 1-D tensors:")
        for k, reason in bad[:10]:
            print(f"  - {k}: {reason}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
    if len(dims) == 1:
        print(f"\nAll embeddings have consistent dimension D={next(iter(dims))}.")
    else:
        print(f"\n[WARN] Embedding dimensions are inconsistent: {sorted(dims)}")

    # Cosine similarity checks
    def get_id(default_idx: int) -> str:
        if default_idx >= len(keys):
            raise RuntimeError("Not enough embeddings to pick defaults.")
        return keys[default_idx]

    id1 = args.id1 if args.id1 is not None else get_id(0)
    id2 = args.id2 if args.id2 is not None else get_id(1) if len(keys) > 1 else get_id(0)

    if id1 not in spk_emb:
        raise KeyError(f"id1={id1} not found. Available keys include: {keys[:10]}")
    if id2 not in spk_emb:
        raise KeyError(f"id2={id2} not found. Available keys include: {keys[:10]}")

    e1 = spk_emb[id1].float()
    e2 = spk_emb[id2].float()

    cos12 = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
    print("\n=== Cosine similarity ===")
    print(f"cos({id1}, {id2}) = {cos12:.6f}")

    # Self-similarity (should be 1.0 up to numerical error)
    cos11 = F.cosine_similarity(e1.unsqueeze(0), e1.unsqueeze(0)).item()
    print(f"cos({id1}, {id1}) = {cos11:.6f}")

    # Nearest neighbors for id1
    if len(keys) > 1:
        E = torch.stack([spk_emb[k].float() for k in keys], dim=0)  # [N, D]
        sims = F.cosine_similarity(e1.unsqueeze(0), E, dim=1)       # [N]
        # Exclude self by setting to -inf
        self_idx = keys.index(id1)
        sims[self_idx] = float("-inf")

        topk = min(args.topk, len(keys) - 1)
        vals, idxs = torch.topk(sims, k=topk, largest=True)

        print(f"\n=== Top-{topk} nearest to {id1} (cosine) ===")
        for rank, (v, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
            print(f"{rank:02d}) {keys[idx]} : {v:.6f}")
    else:
        print("\nOnly one embedding found; nearest-neighbor search skipped.")


if __name__ == "__main__":
    main()
