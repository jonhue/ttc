from typing import NamedTuple, Tuple
from activeft.acquisition_functions import Targeted
from activeft.acquisition_functions.vtl import VTL
from activeft.acquisition_functions import AcquisitionFunction, Targeted
from activeft import ActiveDataLoader
from activeft.data import Dataset as AbstractDataset
import torch
import time
import concurrent.futures
import numpy as np


class Dataset(AbstractDataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]


class Retriever:
    """
    Retriever that scores items in a corpus using SIFT and returns the top-N.
    There is no presampling; the full corpus is considered for selection.
    """

    corpus: np.ndarray

    def __init__(
        self,
        corpus: np.ndarray,
        acquisition_function: AcquisitionFunction | None = None,
        lambda_: float = 0.01,
        device: torch.device | None = None,
    ):
        """
        :param corpus: Full corpus embeddings of shape (num_items, d).
        :param lambda_: Value of the lambda parameter of SIFT.
        :param device: Device to use for computation.
        """
        self.corpus = corpus.astype("float32")
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print("Sift Device")
        print(self.device)
        print("Corpus Shape")
        print(self.corpus.shape[0])
        if acquisition_function is not None:
            self.acquisition_function = acquisition_function
        else:
            self.acquisition_function = VTL(
                target=torch.Tensor(),
                num_workers=1,
                subsample=False,
                force_nonsequential=False,
                noise_std=np.sqrt(lambda_),
            )

    def search(
        self,
        query: np.ndarray,
        N: int,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        r"""
        :param query: Query embedding (of shape $m \times d$), comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of length $N$), array of selected indices (of length $N$), array of corresponding embeddings (of shape $N \times d$), retrieval time.
        """
        D, I, V, retrieval_time = self.batch_search(
            queries=np.array([query]),
            N=N,
            mean_pooling=mean_pooling,
            threads=threads,
        )
        return D[0], I[0], V[0], retrieval_time

    def batch_search(
        self,
        queries: np.ndarray,
        N: int,
        mean_pooling: bool = False,
        threads: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        r"""
        :param queries: $n$ query embeddings (of combined shape $n \times m \times d$), each comprised of $m$ individual embeddings.
        :param N: Number of results to return.
        :param mean_pooling: Whether to use the mean of the query embeddings.
        :param threads: Number of threads to use.

        :return: Array of acquisition values (of shape $n \times N$), array of selected indices (of shape $n \times N$), array of corresponding embeddings (of shape $n \times N \times d$), retrieval time.
        """
        queries = queries.astype("float32")
        n, m, d = queries.shape
        assert d == self.corpus.shape[1]
        mean_queries = np.mean(queries, axis=1)

        def engine(i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            dataset = Dataset(torch.tensor(self.corpus))
            target = torch.tensor(
                queries[i] if not mean_pooling else mean_queries[i].reshape(1, -1)
            )

            if isinstance(self.acquisition_function, Targeted):
                self.acquisition_function.set_target(target)

            sub_indexes, values = ActiveDataLoader(
                dataset=dataset,
                batch_size=N,
                acquisition_function=self.acquisition_function,
                device=self.device,
            ).next()
            values = np.array(values)
            indices = np.array(sub_indexes)
            embeddings = np.array(self.corpus[sub_indexes])

            return values, indices, embeddings

        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            results = list(executor.map(engine, range(n)))
        resulting_values = [r[0] for r in results]
        resulting_indices = [r[1] for r in results]
        resulting_embeddings = [r[2] for r in results]
        retrieval_time = time.time() - t_start
        dtype = None
        return (
            np.array(resulting_values, dtype=dtype),
            np.array(resulting_indices, dtype=dtype),
            np.array(resulting_embeddings, dtype=dtype),
            retrieval_time,
        )
