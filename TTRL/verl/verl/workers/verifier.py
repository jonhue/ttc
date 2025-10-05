import logging
import os
import re
from typing import List
from collections import defaultdict

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl import DataProto
from tensordict import TensorDict

from verl.utils.reward_score import multi_source_reward

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
)

VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str:
    r"""
    Extract the last occurrence of a boxed answer from the input text.

    Returns:
        The content inside the last \boxed{...} or None if not found.
    """
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_last_final_answer(text: str) -> str:
    """
    Try to extract the final answer from the text using several candidate patterns.

    Returns:
        The extracted answer as a string, or None if none of the patterns match.
    """
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]

    last_match = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if last_match and last_match.endswith(stop_word):
            last_match = last_match[:-len(stop_word)].strip()

    return last_match


def extract_solution(solution_str: str) -> str:
    boxed_answer = extract_last_boxed(solution_str)
    if boxed_answer:
        return boxed_answer
    return extract_last_final_answer(solution_str)


def generate_verification(model, tokenizer, config, messages: List[str]):
    # Generate verification responses using HF transformers (batched)
    enc = tokenizer(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.model.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract only the newly generated tokens (excluding the prompt part), with left padding
    responses = []
    for i in range(input_ids.size(0)):
        gen_only = gen_ids[i, input_ids.size(1):]
        text = tokenizer.decode(gen_only).strip() #, skip_special_tokens=True
        responses.append(text)
    return responses


def generate_batched_verification(model, tokenizer, config, messages: List[str], batch_size: int = 8):
    model.eval()
    responses = []
    with torch.no_grad():
        for start in range(0, len(messages), batch_size):
            batch_msgs = messages[start:start + batch_size]

            enc = tokenizer(
                batch_msgs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.model.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Slice off the prompt (per-batch) and decode
            prompt_len = input_ids.size(1)
            for i in range(input_ids.size(0)):
                gen_only = gen_ids[i, prompt_len:]
                text = tokenizer.decode(gen_only).strip()  # keep original behavior
                responses.append(text)

            # Optional: free memory between batches
            del enc, input_ids, attention_mask, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return responses


def compute_score_verifier(tokenizer, ground_truth: str, solution: str | None, verification_text: str) -> float:
    score = 0.0
    # Penalize if solution extraction failed.
    if solution is None or solution == "":
        score -= 0.5
    # Award a score and adjust based on token length difference if verification passes.
    elif VERIFIER_PASS_TAG in verification_text:
        score += 1.0
        tokenized_solution = tokenizer.encode(solution)
        tokenized_ground_truth = tokenizer.encode(ground_truth)
        # Penalize based on the absolute difference in token count (capped to 10 tokens).
        difference = abs(len(tokenized_solution) - len(tokenized_ground_truth))
        difference = min(difference, 10)
        score -= difference * 0.05
    return float(score)


class RewardModelWorker(Worker):
    def __init__(self, config):
        """
        Initializes the reward model worker with its configuration and sampling parameters.
        """
        super().__init__()
        self.config = config
        # Debug print controls (analogous to NaiveRewardManager)
        # Print up to this many examples per batch for debugging
        self.num_examine = getattr(self.config, "num_examine", 0)
        # Truncate very long strings to this many characters in debug output
        self.max_print_chars = getattr(self.config, "max_print_chars", 2500)

        self.reward_kwargs = dict(getattr(self.config, "reward_kwargs", {}))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialize the language model and tokenizer.
        """
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        # Tokenizer
        from verl.utils.fs import copy_to_local
        # local_path = copy_to_local(self.config.model.path)
        # print(local_path)
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=trust_remote_code,
        )
        local_path = copy_to_local(self.config.model.input_tokenizer)
        self.train_tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # self.tokenizer = hf_tokenizer(
        #     local_path,
        #     trust_remote_code=trust_remote_code,
        # )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models need left padding for correct batched generation
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.path,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            device_map=None
        ).to("cuda")
        self.model.eval()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        """
        Compute the reward model score for each data item.

        For every data instance, the function decodes the sequence of prompt and response
        tokens, extracts the solution, and then uses a language model to verify the answer.
        A reward score is then computed based on whether the verified answer is correct and the
        token length difference from the ground truth.

        Returns:
            A DataProto object containing the computed reward scores.
        """
        response_strs = []
        prompt_strs = []
        ground_truths = []
        reward_styles = []
        questions = []
        valid_response_lengths = []
        extra_infos = []
        indices = []

        # Process each data item to create a sequence string and extract necessary fields.
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            # Extract question and ground truth from non-tensor batch.
            question = data_item.non_tensor_batch["extra_info"].get("problem", None)
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            reward_style = data_item.non_tensor_batch["reward_model"]["style"]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            index = extra_info["index"]

            questions.append(question)
            ground_truths.append(ground_truth)
            reward_styles.append(reward_style)
            extra_infos.append(extra_info)
            indices.append(int(index))

            # Decode response string
            valid_response_ids = response_ids[:valid_response_length]
            extra_infos[i]["truncated"] = response_ids.shape[-1] > valid_response_length
            if reward_style == "verifier":
                valid_response_ids = valid_response_ids[-1024:]  # avoid risk of getting too long answer extracted
            response_str = self.train_tokenizer.decode(valid_response_ids) # , skip_special_tokens=True
            response_strs.append(response_str)

            # Decode prompt string for debug prints
            prompt_str = self.train_tokenizer.decode(valid_prompt_ids) #, skip_special_tokens=True
            prompt_strs.append(prompt_str)

        # Split indices by reward style
        verifier_indices = [i for i, s in enumerate(reward_styles) if s == "verifier"]
        non_verifier_indices = [i for i, s in enumerate(reward_styles) if s != "verifier"]

        # Prepare messages for the verification prompt
        messages = []
        solutions = []
        for i in verifier_indices:
            solution = extract_solution(response_strs[i])
            messages.append(
                VERIFIER_PROMPT_TEMPLATE.format(
                    question=questions[i],
                    ground_truth=ground_truths[i],
                    student_answer=solution,
                )
            )
            solutions.append(solution)

        # Generate verification responses (if any)
        responses = []
        if len(messages) > 0:
            #responses = generate_verification(self.model, self.tokenizer, self.config, messages)
            responses = generate_batched_verification(self.model, self.tokenizer, self.config, messages)

        # Initialize reward tensor with the same shape as responses.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        #print(indices)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=reward_tensor.device)
        #print(index_tensor)
        # 1) Compute rewards for non-verifier items via compute_score
        #reward_extra_info = defaultdict(list)

        printed = 0
        for i in non_verifier_indices:
            result = multi_source_reward.compute_score(
                solution=response_strs[i],
                ground_truth=ground_truths[i],
                reward_style=reward_styles[i],
                extra_info=extra_infos[i],
                **self.reward_kwargs
            )
            score = result["score"]
            reward_tensor[i, valid_response_lengths[i] - 1] = score

            # Debug prints (analogous to naive reward manager)
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                print("NON-VERIFIER:")
                print("[reward_style]", str(reward_styles[i])[: self.max_print_chars])
                print("[prompt]", str(prompt_strs[i])[: self.max_print_chars])
                print("[response]", str(response_strs[i])[-self.max_print_chars :])
                print("[solution]", str(response_strs[i])[-self.max_print_chars :])
                print("[ground_truth]", str(ground_truths[i])[: self.max_print_chars])
                for k, v in result.items():
                    print(f"[{k}]", str(v))

        # 2) Compute rewards via verifier
        printed = 0
        assert len(responses) == len(verifier_indices) and len(solutions) == len(verifier_indices)
        for idx_in_batch, i in enumerate(verifier_indices):
            score = compute_score_verifier(
                tokenizer=self.tokenizer,
                ground_truth=ground_truths[i],
                solution=solutions[idx_in_batch],
                verification_text=responses[idx_in_batch],
            )
            reward_tensor[i, valid_response_lengths[i] - 1] = score

            # Debug prints (analogous to naive reward manager)
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                print("VERIFIER:")
                print("[reward_style]", str(reward_styles[i])[: self.max_print_chars])
                print("[prompt]", str(prompt_strs[i])[: self.max_print_chars])
                print("[response]", str(response_strs[i])[-self.max_print_chars :])
                print("[solution]", str(solutions[idx_in_batch])[-self.max_print_chars :])
                print("[verification_text]", str(responses[idx_in_batch])[-self.max_print_chars :])
                print("[ground_truth]", str(ground_truths[i])[: self.max_print_chars])
                print("[score]", score)

        batch = TensorDict({"rm_scores": reward_tensor, "index": index_tensor}, batch_size=reward_tensor.shape[0])
        torch.cuda.empty_cache()
        return DataProto(batch=batch)
