#!/usr/bin/env python
"""
compute_neighbors_gpu.py

Load mean-pooled embeddings (.npy), move to GPU, compute full cosine
similarity matrix, rank neighbors for each point, and save indices.
"""

import argparse
import numpy as np
import torch

def main(emb_path: str, 
         out_path: str, 
         use_fp16: bool = False, 
         emb_path_2: str = None, 
         method: str = None,
         ignore_same = True):
    # 1. Load embeddings
    embs = np.load(emb_path)                 # shape: (N, D)
    embs_2 = None
    if not emb_path_2 is None:
        embs_2 = np.load(emb_path_2)

    if method is None or (method == "false"):
        neighbors_np = np.linspace(0, embs_2.shape[0]-1, embs_2.shape[0]).astype(int)
        neighbors_np = np.repeat(np.array([neighbors_np]), embs.shape[0], axis=0)
        print(neighbors_np)
        np.save(out_path, neighbors_np)
        return 
    elif method != "nearest_neighbor":
        raise NotImplementedError(f"Method {method} is not implemented for preselection.")
    
    # 2. Move to GPU tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_fp16 else torch.float32

    embs_t = torch.from_numpy(embs).to(device, dtype=dtype)
    print(embs_t)

    # 3. Normalize to unit length
    embs_norm = embs_t / embs_t.norm(dim=1, keepdim=True)

    if not embs_2 is None:
        embs_2_t = torch.from_numpy(embs_2).to(device, dtype=dtype)
        embs_2_norm = embs_2_t / embs_2_t.norm(dim=1, keepdim=True)
        print(embs_2_norm)


    # 4. Compute cosine similarity matrix
    with torch.no_grad():
        if embs_2 is None:
            sim = embs_norm @ embs_norm.T        # on GPU
        else:
            sim = embs_norm @ embs_2_norm.T
        print(sim)
        neighbors = torch.argsort(sim, dim=1, descending=True)

    neighbors_np = neighbors.cpu().numpy()     # shape
    print(neighbors_np)
    np.save(out_path, neighbors_np)
    print(f"Saved neighbor indices array of shape {neighbors_np.shape} to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Rank all points by cosine similarity on GPU"
    )
    p.add_argument("--embeddings",    required=True,
                   help="Path to embeddings .npy file")
    p.add_argument("--embeddings_2",    default=None,
                   help="Path to second embeddings .npy file")
    p.add_argument("--output",        required=True,
                   help="Path to save neighbor indices .npy")
    p.add_argument("--method", required=True,
                    help="Which kind of neighborhood to use.")
    p.add_argument("--use_fp16", action="store_true",
                   help="Use float16 on GPU (halves memory for sim matrix)")

    args = p.parse_args()
    main(emb_path=args.embeddings, 
         out_path=args.output, 
         use_fp16=args.use_fp16, 
         emb_path_2=args.embeddings_2, 
         method=args.method)