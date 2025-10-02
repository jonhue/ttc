import os
import argparse
import numpy as np
import pandas as pd

from data.load_dataset import load_dataset_hf
from data.utils.data_handling import write_hf_to_json

USER = os.environ.get("USER")

#import pyarrow.json as paj
#import pyarrow as pa
#from datasets import load_dataset, Features, Sequence, Value, Dataset, load_from_disk
#import json

# def _cast_large_strings(ds, columns):
#     new_feats = ds.features.copy()
#     for col in columns:
#         if col in ds.column_names:
#             new_feats[col] = Value("large_string")
#     ds = ds.cast(new_feats)
#     return ds

# 1. Load your dataset from Hugging Face
#    e.g. the “squad” dataset, split “train”
def sort_dataset_similarities(  dataset_name: str,
                                similarities_file_name: str,
                                output_path: str,
                                start_idx: int = 0, 
                                num_el: int = None,
                                drop_duplicates: bool = False,
                                max_num_duplicates = 16,
                                index_list = None,
                                data_source = None,
                                embeddings_file = None):
    #filter_columns = ["idx", "dataset", "elo", "achievement_prior", "answer", "prompt", "tests", "embedding"]
    #str_columns = ["idx", "dataset", "elo", "achievement_prior", "answer", "prompt", "tests"]

    
    if data_source == "false":
        data_source = None
        
    if dataset_name == "lasgroup/ttt_reasoning_dataset":
        ds = load_dataset_hf(dataset_name,
                    dataset_split = "train",
                    output_path = None,
                    category = data_source,
                    embeddings_file=embeddings_file)
    else:
        raise ValueError(f"{dataset_name} not implemented.")

    #if not embeddings_file is None:
    #    ds = ds.add_column("embedding", [embeddings[i] for i in range(len(ds))])

    #if not data_source is None:
    #    filtered_indices = ds.filter(lambda ex: ex["kind"] == data_source)["idx"]
    
    sim = np.load(similarities_file_name)
    
    # if not "elo" in ds.column_names:
    #     ds = ds.add_column("elo", [0] * len(ds))
    # else:
    #     ds = ds.map(lambda ex: {"elo": "-" if ex["elo"] is None else ex["elo"]})

    if not index_list is None:
        sim_flat = np.reshape(sim[index_list], -1)
    else:
        sim_flat = np.reshape(sim.T, -1)

    unique = list(dict.fromkeys(sim_flat))

    if not data_source is None:
        #unique = [idx for idx in unique if idx in ds["idx"]]
        #unique = list(set(unique) & set(ds["idx"]))
        unique = np.array(unique)
        ds_idx = np.array(ds["idx"])
        mask = np.isin(unique, ds_idx)         
        unique = unique[mask]                  
        #unique = unique.tolist()               
    unique = np.array(unique)

    if num_el is None:
        last_el = unique.shape[0]
    else:
        if drop_duplicates:
            last_el = start_idx + max_num_duplicates * num_el 
        else:
            last_el = start_idx + num_el 
    unique = unique[start_idx: last_el]
    
    # 2. Order dataset
    print(f"Dataset size: {unique.shape[0]}")
    ds_transformed = ds.select(unique)
    print(ds_transformed.column_names)

    #ds_transformed = _cast_large_strings(ds_transformed, columns=str_columns)
    # if data_source == "math": ### Todo: Make loading more seemless
    #     print("Load Achievement Data.")
    #     path = f"/users/{USER}/ldiazbone-shared/data/verl_data/"
    #     achievement_prior_qwen = pd.read_csv(path + "achievability_estimates/math_Qwen_3_8B.csv").set_index("indices")["scores"].to_dict()
    #     achievement_prior_apertus = pd.read_csv(path + "achievability_estimates/math_apertus_8B.csv").set_index("indices")["scores"].to_dict()    
    #     ds_transformed = ds_transformed.map(lambda ex: {"achievement_apertus": achievement_prior_apertus[int(ex["idx"])] if int(ex["idx"]) in achievement_prior_apertus.keys() else np.random.uniform(0, 1)})
    
    path = f"/users/{USER}/ldiazbone-shared/data/verl_data/"
    achievement_prior_qwen = pd.read_csv(path + "achievability_estimates/lasgroup_ttt_reasoning_dataset_Qwen_Qwen3-8B.csv").set_index("indices")["scores"].to_dict()
    ds_transformed = ds_transformed.map(lambda ex: {"achievement_Qwen_3_8B": achievement_prior_qwen[int(ex["idx"])] if int(ex["idx"]) in achievement_prior_qwen.keys() else np.random.uniform(0, 1)})
    ds_transformed = ds_transformed.add_column("achievement_prior", (np.array(ds_transformed["achievement_Qwen_3_8B"]) )) # + np.array(ds_transformed["achievement_apertus"]

    print(f"Export to file {output_path}.")
    print(ds_transformed.features)


    # 3. 
    # if drop_duplicates:
    #     seen_questions = set()
    #     filtered_data = []
    #     for example in ds_transformed:
    #         if len(filtered_data) >= num_el:
    #             break
    #         question = example["conversations"][0]["value"]
    #         if question not in seen_questions:
    #             seen_questions.add(question)
    #             filtered_data.append(example)
    #     ds_transformed = Dataset.from_list(filtered_data)
    print(f"Final number dataset samples: {len(ds_transformed)}")
    
    #ds_transformed = ds_transformed.select_columns(filter_columns)
    
    # ds_transformed.to_json(
    #     output_path,
    #     batch_size=10_000,
    #     num_proc=8
    # )
    write_hf_to_json(
            ds=ds_transformed,
            output_path=output_path
        )
    # ds_transformed.to_json(
    #         output_path,
    #         batch_size=15_000,
    #         num_proc=8,
    #         lines=False,          # <-- JSON array, not JSONL
    #         orient="records",     # list of objects
    #         force_ascii=False,    # optional: keep Unicode chars
    #         indent=2              # optional: pretty print
    #     )

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
        "--drop_duplicates", type=bool, default=False,
        help="Wheter to drop or keep duplicate questions in the dataset."
    )
    parser.add_argument(
        "--index_list", type=int, nargs='+', default=None,
        help="List of question indices to use."
    )
    parser.add_argument(
        "--data_source", type=str, default=None,
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
        drop_duplicates=args.drop_duplicates,
        index_list=args.index_list,
        data_source=args.data_source,
        embeddings_file=args.embeddings_file)

# python sort_dataset.py --dataset_name open-thoughts/OpenThoughts3-1.2M --output_path /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/open_thoughts_sorted_aim.jsonl --similarities_file_name /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/similarities_aime25_open_thoughts.npy --start_idx 0 --end_idx
# python sort_dataset.py --dataset_name open-thoughts/OpenThoughts3-1.2M --output_path /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/open_thoughts_sorted_aim.parquet --similarities_file_name /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/similarities_aime25_open_thoughts.npy --start_idx 0 --end_idx
# python sort_dataset.py --dataset_name open-thoughts/OpenThoughts3-1.2M --output_path /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/open_thoughts_sorted_aime_24_50000.jsonl --similarities_file_name /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/similarities_aime_24_open_thoughts.npy --start_idx 0 --end_idx 50_000
# python sort_dataset.py --dataset_name open-thoughts/OpenThoughts3-1.2M --output_path /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/open_thoughts_sorted_math_500_50000.jsonl --similarities_file_name /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/similarities_math_500_open_thoughts.npy --start_idx 0 --end_idx 50_000
# python sort_dataset.py --dataset_name open-thoughts/OpenThoughts3-1.2M --output_path /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/open_thoughts_sorted_amc_23_50000.jsonl --similarities_file_name /capstor/store/cscs/swissai/infra01/users/ldiazbone/data/similarities_amc_23_open_thoughts.npy --start_idx 0 --end_idx 50_000