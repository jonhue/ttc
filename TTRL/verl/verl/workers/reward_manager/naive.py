
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import multi_source_reward


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, reward_fn_key="data_source", reward_kwargs: dict = {}) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.max_print_chars = 2500

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "index": data.batch["index"], "reward_extra_info": {}}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids) # , skip_special_tokens=True
            response_str = self.tokenizer.decode(valid_response_ids) # , skip_special_tokens=True

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            reward_style = data_item.non_tensor_batch["reward_model"]["style"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            extra_info["truncated"] = response_ids.shape[-1] > valid_response_length
            index = extra_info["index"]
            reward_extra_info["index"].append(index)

            score = multi_source_reward.compute_score(
                solution=response_str,
                ground_truth=ground_truth,
                reward_style=reward_style,
                extra_info=extra_info,
                **self.reward_kwargs,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[datasource]", data_source)
                print("[prompt]", str(prompt_str)[:self.max_print_chars])
                print("[response]", str(response_str)[-self.max_print_chars:])
                print("[ground_truth]", str(ground_truth)[:self.max_print_chars])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", str(value)[:self.max_print_chars])
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
                "index": index
            }
        else:
            return reward_tensor
