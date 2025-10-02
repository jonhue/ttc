import numpy as np, torch
from verl import DataProto
from tensordict import TensorDict
from activeft.acquisition_functions.vtl import VTL
from activeft.acquisition_functions.cosine_similarity import CosineSimilarity

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.dataset.sift_utils import Retriever


class SiftWorker(Worker):
    def __init__(self, config=None):
        print("Initialize Sift worker.")
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subset_size = config.dynamic.subset_size
        self.emb_all = None
        self.val = None
        self.acquisition_function = config.dynamic.acquisition_function
        if self.acquisition_function == "vtl":
            self.sift_lambda = config.dynamic.sift_lambda
            self.af = VTL(target=torch.Tensor().to(self.device),
                          num_workers=1, subsample=False,
                          force_nonsequential=False,
                          noise_std=np.sqrt(self.sift_lambda))
        elif self.acquisition_function == "cosine":
            self.af = CosineSimilarity(target=torch.Tensor().to(self.device),
                                       num_workers=1, subsample=False)
        elif self.acquisition_function == "random":
            self.af = None
        elif self.acquisition_function == "cosine_neighbors":
            self.af = None
        else:
            raise NotImplementedError(f"Acquisition function {self.acquisition_function} not implemented.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        pass  # required by interface

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_embeddings(self, emb_all: np.ndarray, emb_val: np.ndarray):
        print("Set embedding Sift workers.")
        self.emb_all = np.asarray(emb_all, dtype=np.float32, order="C")
        self.val     = np.asarray(emb_val,     dtype=np.float32, order="C")
        print(self.emb_all.shape)
        print(self.val.shape)

    def cosine_neighbors(self, cand_idx: np.ndarray):
        embs_t = torch.from_numpy(self.val).to(self.device, dtype=torch.float16)
        embs_2_t = torch.from_numpy(self.emb_all[cand_idx]).to(self.device, dtype=torch.float16)
        embs_norm = embs_t / embs_t.norm(dim=1, keepdim=True)
        embs_2_norm = embs_2_t / embs_2_t.norm(dim=1, keepdim=True)
        with torch.no_grad():
            sim = embs_norm @ embs_2_norm.T
            neighbors = torch.argsort(sim, dim=1, descending=True)
        neighbors_np = neighbors.cpu().numpy()     # shape (N, N)
        sim_flat = np.reshape(neighbors_np.T, -1)
        unique = np.array(list(dict.fromkeys(sim_flat)))
        return unique[:self.subset_size]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def search(self, cand_idx: np.ndarray):
        assert cand_idx.shape[0] >= self.subset_size, f"The number of candidate indices {cand_idx.shape[0]} is smaller than the required subset size {self.subset_size}."

        if self.acquisition_function == "random":
            print("Perform random search.")
            selected = np.random.choice(cand_idx, size=self.subset_size, replace=False)
            return selected, np.zeros(self.subset_size)

        if self.acquisition_function == "cosine_neighbors":
            print("Perform cosine neighborhood search.")
            return self.cosine_neighbors(cand_idx), np.zeros(self.subset_size)

        print("Perform sift search.")
        C = self.emb_all[cand_idx]
        retriever = Retriever(C, acquisition_function=self.af, device=self.device)
        vals, idx, _, _ = retriever.search(self.val, N=self.subset_size)
        return cand_idx[idx], vals
