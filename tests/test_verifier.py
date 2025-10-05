import os
import sys
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl.utils import hf_tokenizer

from TTRL.verl.verl.workers.verifier import generate_verification

def main():
    model_name = "TIGER-Lab/general-verifier"  # small and fast for debugging
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer = hf_tokenizer(
        "TIGER-Lab/general-verifier",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models need left padding for correct batched generation
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    # # Use left padding for batched causal generation to avoid head-shape issues
    # try:
    #     tokenizer.padding_side = "left"
    # except Exception:
    #     pass

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Minimal config object with required field
    config = SimpleNamespace(model=SimpleNamespace(max_new_tokens=1024))

    # Example input(s)
    message = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
).format(question="What is 2 + 2?", ground_truth="4", student_answer="5")
    message2 = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
).format(question="What is 2 + 2?", ground_truth="4", student_answer="Hmm. I think it's 5.")
    messages = [message, message2]

    responses = generate_verification(model, tokenizer, config, messages)
    print("Responses:", responses)

if __name__ == "__main__":
    main()
