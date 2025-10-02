import copy
import os
import wandb
import re
from typing import List, Optional, Union
import time

import ray
import datasets
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


def inverse_sigmoid(a, eps):
    a = np.clip(a, eps, 1 - eps)
    return np.log(a / (1 - a))


class DynamicRLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        val_data_files: Union[str, List[str]] = None,
        processor: Optional[ProcessorMixin] = None,
        suffix_prompt: Optional[str] = None,
        total_epochs: int = 30,
    ):
        start_initialization = time.time()
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        if (not val_data_files is None) and (not isinstance(data_files, (List, ListConfig))):
            val_data_files = [val_data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        if not val_data_files is None:
            self.val_data_files = copy.deepcopy(val_data_files)
            self.val_original_data_files = copy.deepcopy(val_data_files)  # use for resume
        else:
            self.val_data_files = None
            self.val_original_data_files = None
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        self.suffix_prompt = suffix_prompt

        self.filter_achievability = config.dynamic.filter_achievability.enable
        self.subset_size = config.dynamic.subset_size
        self.total_data_size = config.dynamic.total_data_size
        if self.filter_achievability:
            self.update_delay = config.dynamic.subset_size // config.train_batch_size  # update every few batches
        else:
            self.update_delay = self.total_data_size  # don't update again
        if self.filter_achievability:
            self.min_ach_band = config.dynamic.filter_achievability.min_ach_band
            self.max_ach_band = config.dynamic.filter_achievability.max_ach_band
            self.min_questions_in_band = config.dynamic.filter_achievability.min_questions_in_band
            self.linear_estimation_offset_clip = config.dynamic.filter_achievability.linear_estimation_offset_clip

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
        self.cur_dataframe = self.dataframe.select(np.array(list(range(0, min(self.total_data_size, len(self.dataframe))))))

        print("Initialize dataset with size: " + str(self.dataframe.shape[0]))
        self.cur_ach_est = np.array([extra_info["achievement_prior"] for extra_info in self.dataframe["extra_info"]]).astype(float)
        self.orig_indices = np.array([extra_info["index"] for extra_info in self.dataframe["extra_info"]]).astype(int)
        self.train_embeddings = np.stack(self.dataframe["embedding"], axis=0) if not self.config.dynamic.maj_on_test else None
        self.index_map = {v: i for i, v in enumerate(self.orig_indices)}
        if not self.val_data_files is None:
            self.val_embeddings = np.array([self.val_dataframe["embedding"][i] for i in range(len(self.val_dataframe))])
            print(f"Validation Embeddings Shape: {self.val_embeddings.shape}")
        end_initialization = time.time()
        print(f"Dynamic Dataset Initialization Time: {end_initialization - start_initialization}")

    def _set_sift_worker(self, sift_worker = None):
        self.sift_worker = sift_worker
        self.sift_worker.set_embeddings(emb_all=self.train_embeddings, emb_val=self.val_embeddings)

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)
        if not self.val_data_files is None:
            val_data_files = self.val_data_files if not use_origin_parquet else self.val_original_data_files
            for i, parquet_file in enumerate(val_data_files):
                self.val_data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _add_suffix_to_entry(self, entry):
        # Assuming 'text' is the field where the prompt should be added
        entry[self.prompt_key][-1]["content"] = entry[self.prompt_key][-1]["content"] + self.suffix_prompt
        return entry

    def _read_files_and_tokenize(self):
        dataframes = []

        train_data_files = self.val_data_files if self.config.dynamic.maj_on_test else self.data_files
        for parquet_file in train_data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        if self.config.dynamic.maj_on_test:
            self.dataframe = self.dataframe.map(lambda ex: {
                "reward_model": {
                    "style": f"maj_{ex['reward_model']['style']}",
                    "ground_truth": ""
                }
            })

        if not self.val_data_files is None:
            val_dataframes = []
            for parquet_file in self.val_data_files:
                # read parquet files and cache
                val_dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
                val_dataframes.append(val_dataframe)
            self.val_dataframe: datasets.Dataset = datasets.concatenate_datasets(val_dataframes)

        # Filter by kind
        self.filter_kind = self.config.dynamic.filter_kind
        if self.filter_kind is not None and self.filter_kind != "":
            self.dataframe = self.dataframe.filter(lambda ex: ex["reward_model"]["style"] == self.filter_kind)

        print(f"dataset len: {len(self.dataframe)}")

        if self.suffix_prompt is not None:
            # add suffix prompt
            print(f"Apply suffix prompt {self.suffix_prompt}")
            self.dataframe = self.dataframe.map(self._add_suffix_to_entry)

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")
            if not self.val_data_files is None:
                self.val_dataframe = self.val_dataframe.filter(
                    lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                    <= self.max_prompt_length,
                    num_proc=self.num_workers,
                    desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
                )
                print(f"filtered validation dataset len: {len(self.val_dataframe)}")

    def __len__(self):
        return len(self.cur_dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _estimate_achievability_questions(self, train_ach_metrics, eps = 0.05):
        """
        Implements a iterative estimation of the current achievability of each question, based on prior estimation and linear update.
        """
        if len(list(train_ach_metrics.keys())) > 0:
            next_ach_est = np.zeros_like(self.cur_ach_est)
            inverse_mask = np.ones_like(self.cur_ach_est)

            indices = np.array([self.index_map[int(x)] for x in train_ach_metrics.keys()]).astype(int)
            values = np.array([train_ach_metrics[x] for x in train_ach_metrics.keys()]).astype(float)

            if np.isnan(values).any():
                print(f"Nan in train_ach_metrics: {np.where(np.isnan(values))[0][:10]}")
                values[np.isnan(values)] = 0.0

            next_ach_est[indices] = values
            inverse_mask[indices] = 0

            linear_est_offset = np.mean(inverse_sigmoid(values, eps) - inverse_sigmoid(self.cur_ach_est[indices], eps))
            clipped_linear_est_offset = np.clip(linear_est_offset, -self.linear_estimation_offset_clip, self.linear_estimation_offset_clip)
            next_ach_est = next_ach_est + inverse_mask * 1 / (1 + np.exp(-(inverse_sigmoid(self.cur_ach_est, eps) + clipped_linear_est_offset)))

            prediction_achievement = np.mean(self.cur_ach_est[indices])
            self.cur_ach_est = next_ach_est
            assert not np.isnan(self.cur_ach_est).any(), f"NaNs found at indices: {np.where(np.isnan(self.cur_ach_est))[0][:10]}"
            assert np.isfinite(self.cur_ach_est).all(), f"Non-finite values at indices: {np.where(~np.isfinite(self.cur_ach_est))[0][:10]}"
            return {"linear_estimation_offset": linear_est_offset, "prediction_achievement": prediction_achievement} # Zero out as used
        else:
            return {} # Return original as not used

    def _filter_question_sift(self, preselected_indices): # dataset
        print(f"Start Retrieval ({time.time()})")
        start = time.time()
        result = self.sift_worker.search(preselected_indices)
        if isinstance(result, list):
            result = result[0]
        sift_indices, sift_acquisition_values = result
        end = time.time()
        print(f"Sift Retrieval Time: {end-start}")
        return sift_indices, sift_acquisition_values

    def _filter_questions_band(self, min_questions_in_band, min_val = 0.25, max_val = 0.6):
        midpoint = (min_val + max_val) / 2.0
        values = self.cur_ach_est
        indices_in_range = np.where((values >= min_val) & (values <= max_val))[0]
        if len(indices_in_range) >= min_questions_in_band:
            selected_indices = indices_in_range
        else:
            selected_indices = list(indices_in_range)
            remaining = min_questions_in_band - len(indices_in_range)
            outside_indices = np.where((values < min_val) | (values > max_val))[0]
            sorted_outside = outside_indices[np.argsort(np.abs(values[outside_indices] - midpoint))]
            selected_indices = np.concatenate([indices_in_range, sorted_outside[:remaining]])
        assert selected_indices.shape[0] >= min_questions_in_band
        return selected_indices

    def subsample_dataset(self, global_steps, train_ach_metrics):
        """Randomly subsample at most `max_samples` entries from the dataset."""
        metrics = {}

        if global_steps % self.update_delay == 0:
            start_pre = time.time()
            if self.filter_achievability:
                achievability_metrics = self._estimate_achievability_questions(train_ach_metrics)
                metrics.update(achievability_metrics)
                preselected_indices = self._filter_questions_band(min_questions_in_band=self.min_questions_in_band,min_val=self.min_ach_band, max_val=self.max_ach_band)
            else:
                preselected_indices = np.array([i for i in range(len(self.dataframe))])
            end_pre = time.time()
            print(f"Prefiltering Time: {end_pre-start_pre}")
            start_sift = time.time()
            sift_indices, sift_acquisition_values = self._filter_question_sift(preselected_indices=preselected_indices)
            end_sift = time.time()
            print(f"Sift Time: {end_sift-start_sift}")
            metrics.update({"sift_min_acquisition_val": np.min(sift_acquisition_values), "sift_mean_acquisition_val": np.mean(sift_acquisition_values), "sift_max_acquisition_val": np.max(sift_acquisition_values), "num_questions_pre_selected": len(preselected_indices)})
            print(f"Datset Size: {sift_indices.shape[0]}")
            self.cur_dataframe = self.dataframe.select(sift_indices)
            selected_questions = pd.DataFrame(np.array([np.array([self.cur_dataframe["extra_info"][i]["index"] for i in range(len(self.cur_dataframe))]), sift_acquisition_values, np.array([self.cur_dataframe["extra_info"][i]["description"] for i in range(len(self.cur_dataframe))])]).T, columns=["idx", "sift_acquisition_value", "description"]) #, self.cur_ach_est[selected_band_indices][sift_indices], , "achievability_estimation"

            metrics.update({"achievability_estimation": wandb.Histogram(self.cur_ach_est), "selected_question": wandb.Table(dataframe=selected_questions)})
            return metrics, {}, True
        else:
            return metrics, train_ach_metrics, False

    def __getitem__(self, item):
        """
        Note that we also return the ralw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.cur_dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            # raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            ### Only used for non-thinking mode in Qwen3-8B
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.config.thinking)

            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt")#, add_special_tokens=False
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt) #, add_special_tokens=False
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
                print("Truncate Prompt.")
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def state_dict(self):
        out = {}
        out.update(
            {
                "dyn_data:cur_ach_est": self.cur_ach_est,
            }
        )
        return out

    def load_state_dict(self, state):
        if "dyn_data:cur_ach_est" in state:
            self.cur_ach_est = np.array(state.get("dyn_data:cur_ach_est"))
