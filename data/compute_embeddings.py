#!/usr/bin/env python
import os
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Test
import argparse
from data.load_dataset import load_dataset_hf
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def compute_embeddings(dataset_name: str,
                       question_key: str,
                       output_path: str,
                       model_id: str,
                       batch_size: int = 8,
                       start_data_index: int = 0,
                       end_data_index: int = None,
                       normalize = True,
                       category: str = None):
    print("Model:", model_id)
    # 1. Load dataset
    ds = load_dataset_hf(dataset_name=dataset_name,
                         category=category,
                         output_path=None)
    print(ds)

    end_data_index = end_data_index if end_data_index is not None else len(ds)
    ds = ds.select(range(start_data_index, end_data_index))

    n_examples = len(ds)
    print(f"Loaded dataset with {n_examples} examples.")

    # 2. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                      output_hidden_states=True,torch_dtype=torch.bfloat16)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    model.to(device)

    def formatter(question):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False  # TODO: ablate adding generation prompt
        )

    # 3. Prepare DataLoader
    def collate_fn(batch):
        texts = [formatter(ex[question_key]) for ex in batch]
        enc = tokenizer(texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt")
        return enc

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_fn)

    # 4. Iterate and compute embeddings
    all_embs = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader, start=1)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
            # last hidden layer: (batch, seq_len, hidden_dim)
            hidden = outputs.hidden_states[-1]  # (B, S, D)

            # compute last-token embedding
            lengths   = attention_mask.sum(dim=1)      # (B,)
            last_idx  = lengths - 1                   # (B,)
            batch_idx = torch.arange(hidden.size(0), device=device)
            last_h    = hidden[batch_idx, last_idx, :]  # (B, D)
            #all_embs.append(last_h.detach().cpu().numpy())
            all_embs.append(last_h.detach().cpu().to(torch.float32).numpy())

            if step % 10 == 0 or step == len(loader):
                print(f"  → Processed batch {step}/{len(loader)}")
                print(f"[step {step}] Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                gc.collect()

            del input_ids, attention_mask, outputs, hidden      # Try avoiding OOM
            torch.cuda.empty_cache()

    # 5. Concatenate and save
    embeddings = np.vstack(all_embs)                                    # shape: (N, hidden_dim)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)           # (N, 1)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)   # avoid div-by-zero
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings array of shape {embeddings.shape} to:\n  {output_path}")
    output_file = np.load(output_path)
    print(f"Output embeddings contain array of shape {embeddings.shape} to:\n  {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save question embeddings"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        help="HF dataset name."
    )
    parser.add_argument(
        "--category", type=str,
        help="HF dataset category."
    )
    parser.add_argument(
        "--model_id", type=str,
        help="Huggingface model to use for the embeddings."
    )
    parser.add_argument(
        "--question_key", type=str, default="problem",
        help="Column within dataset corresponding to problem/question."
    )
    parser.add_argument(
        "--output_path", type=str, default="data/embeddings.npy",
        help="File where the embeddings .npy will be saved."
    )
    parser.add_argument(
        "--start_data_index", type=int, default=0,
        help="Start index of subset of the dataset."
    )
    parser.add_argument(
        "--end_data_index", type=int, default=None,
        help="End index of subset of the dataset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size (adjust to fit your GPU/CPU memory)."
    )
    args = parser.parse_args()
    compute_embeddings(
        dataset_name=args.dataset_name,
        question_key=args.question_key,
        output_path=args.output_path,
        model_id=args.model_id,
        batch_size=args.batch_size,
        start_data_index=args.start_data_index,
        end_data_index=args.end_data_index,
        category = args.category,
    )