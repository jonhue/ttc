import os
import argparse
import numpy as np
import pandas as pd

from data.load_dataset import load_dataset_hf
from data.utils.data_handling import write_hf_to_json


# 1. Load your dataset from Hugging Face
def sort_dataset_similarities(  dataset_name: str,
                                similarities_file_name: str,
                                output_path: str,
                                start_idx: int = 0, 
                                num_el: int = None,
                                category = None,
                                embeddings_file = None):
        
    if dataset_name == "lasgroup/verifiable-corpus":
        ds = load_dataset_hf(dataset_name,
                    category = category,
                    embeddings_file=embeddings_file,
                    output_path = None,)
    else:
        raise ValueError(f"{dataset_name} not implemented.")

    
    sim = np.load(similarities_file_name)

    sim_flat = np.reshape(sim.T, -1)

    unique = np.array(list(dict.fromkeys(sim_flat)))

    if num_el is None:
        last_el = unique.shape[0]
    else:
        last_el = start_idx + num_el 
    unique = unique[start_idx: last_el]
    
    # 2. Order dataset
    print(f"Dataset size: {unique.shape[0]}")
    ds_transformed = ds.select(unique)
    print(ds_transformed.column_names)
    
    print(f"Export to file {output_path}.")
    print(f"Final number dataset samples: {len(ds_transformed)}")
    
    write_hf_to_json(
            ds=ds_transformed,
            output_path=output_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce sorted dataset that can be used for training on most relevant questions."
    )
    parser.add_argument(
        "--dataset_name", type=str,
        help="HF dataset name."
    )
    parser.add_argument(
        "--similarities_file_name", type=str,
        help="File containing the similarities between the embeddings."
    )
    parser.add_argument(
        "--output_path", type=str,
        help="File where the dataset will be saved."
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index used from the final dataset."
    )
    parser.add_argument(
        "--num_el", type=int, default=None,
        help="End index used from the final dataset."
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Which datasoure to filter."
    )
    parser.add_argument(
        "--embeddings_file", type=str, default=None,
        help="Where the embeddings for the dataset lie."
    )
    args = parser.parse_args()
    sort_dataset_similarities(
        dataset_name=args.dataset_name, 
        similarities_file_name=args.similarities_file_name, 
        output_path=args.output_path,
        start_idx=args.start_idx, 
        num_el=args.num_el,
        category=args.category,
        embeddings_file=args.embeddings_file)