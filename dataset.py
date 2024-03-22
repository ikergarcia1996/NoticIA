import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import prompts
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def prepare_data(
    tokenizer: PreTrainedTokenizerBase,
    inference: bool,
    headline: str,
    body: str,
    summary: str,
    max_length: int,
    prompt_loss_weight: float = 0.0,
    prompt_name: str = None,
) -> Optional[BatchEncoding]:
    """
    Prepare the data to be feeded into the model.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use.
        inference (`bool`):
            Whether to use the model in inference mode or not. If True, the summary will not be included in the input.
        headline (`str`):
            The headline of the article.
        body (`str`):
            The body of the article.
        summary (`str`):
            The summary to the clickbait question.
        max_length (`int`):
            The maximum length of the input.
        prompt_loss_weight (`float`):
            The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
            of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
        prompt_name (`str`):
            The name of the prompt to use in prompts.py. Defaults to `None`.
            If none, we will use the `summarize_clickbait_short_prompt` prompt.

    Returns:
        `BatchEncoding`: `BatchEncoding` with the prepared data.

    """
    if prompt_name is None:
        prompt_name = "summarize_clickbait_short_prompt"
    try:
        prompt_fn = getattr(prompts, prompt_name, None)
    except AttributeError:
        raise ValueError(f"The prompt {prompt_name} does not exist in prompts.py")

    prompt = prompt_fn(headline, body)

    if tokenizer.chat_template is None:
        # Set ChatML template
        tokenizer.chat_template = (
            "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        )

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    formatted_input = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": summary},
        ],
        tokenize=False,
    )

    if not formatted_input.endswith(tokenizer.eos_token):
        formatted_input += f"{tokenizer.eos_token}"

    prompt_tokens = tokenizer(
        text=formatted_prompt,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_tokens = tokenizer(
        text=formatted_input,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )

    if input_tokens["input_ids"][-1] != tokenizer.eos_token_id:
        if len(input_tokens["input_ids"]) >= max_length:
            input_tokens["input_ids"] = input_tokens["input_ids"][: max_length - 1]
            input_tokens["attention_mask"] = input_tokens["attention_mask"][
                : max_length - 1
            ]
        # Add eos token to fix issues with model generating very long outputs
        input_tokens["input_ids"] = input_tokens["input_ids"] + [tokenizer.eos_token_id]
        input_tokens["attention_mask"] = input_tokens["attention_mask"] + [1]

    # Find the prompt length
    if len(prompt_tokens["input_ids"]) > len(input_tokens["input_ids"]):
        raise ValueError(
            f"The prompt length is greater than the input length. Something went wrong.\n"
            f"Prompt length: {len(prompt_tokens['input_ids'])}\n"
            f"Input length: {len(input_tokens['input_ids'])}\n"
            f"Prompt: {prompt}\n"
            f"Input: {summary}\n"
        )

    prompt_length = len(prompt_tokens["input_ids"])
    input_length = len(input_tokens["input_ids"])
    result_length = input_length - prompt_length

    if inference:
        # labels = input_tokens["input_ids"][prompt_length:]
        # prompt_tokens["labels"] = labels
        return prompt_tokens

    # Create the weight mask
    loss_weight_mask = np.ones(input_length, dtype=np.float32)
    # The sum of the loss of the prompt tokens should be equal
    # to 'prompt_loss_weight' percent of the total loss
    prompt_token_weight = (
        result_length * prompt_loss_weight
    )  # 'prompt_loss_weight' percent of the total loss
    try:
        prompt_token_weight = prompt_token_weight * (
            result_length / (result_length * (1 - prompt_loss_weight))
        )  # Scale so result tokens can have 1.0 weight
        prompt_token_weight = (
            prompt_token_weight / prompt_length
        )  # Divide by the number of prompt tokens
    except ZeroDivisionError:
        logging.warning(
            "Found division by zero in prompt token weight calculation. "
            "The input might be larger than the maximum length. We will skip it."
        )

        return None

    loss_weight_mask[:prompt_length] = prompt_token_weight

    input_tokens["loss_weight_mask"] = loss_weight_mask
    input_tokens["labels"] = input_tokens["input_ids"].copy()

    return input_tokens


class ClickBaitDataset(Dataset):
    """
    Dataset for the ClickBait model.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split_or_path: str,
        max_length: int = 2048,
        inference: bool = False,
        prompt_loss_weight: float = 0.0,
        use_clean_web_text: bool = False,
        prompt_name: str = None,
    ):
        """
        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                The tokenizer to use.
            split_or_path (`str`):
                The path to the dataset in jsonl format or the path in the HuggingFace Hub.
            max_length (`int`):
                The maximum length of the input.
            inference (`bool`):
                Whether to use the model in inference mode or not. If True, the summary will not be included in the input.
            prompt_loss_weight (`float`):
                The weight of the prompt tokens in the loss. If set to '0.05' the prompt tokens will have a total weight
                of 5% in the loss while the result tokens will have a total weight of 95%. Defaults to `0.05`.
            use_clean_web_text (`bool`):
                Whether to use the clean version of the web text or not. Defaults to `False`.
            prompt_name (`str`):
                The name of the prompt to use in prompts.py. Defaults to `None`.
        """

        if os.path.exists(split_or_path):
            logging.warning(f"Loading dataset from local file {split_or_path}")
            with open(split_or_path, "r", encoding="utf8") as f:
                self.dataset = [json.loads(line) for line in f]

            self.dataset_name = os.path.basename(os.path.splitext(split_or_path)[0])

        else:
            logging.warning(f"Loading dataset from HuggingFace Hub {split_or_path}")
            self.dataset_name = split_or_path
            if split_or_path == "all":
                dataset = load_dataset("Iker/NoticIA")
                self.dataset = []
                for example in dataset["train"]:
                    self.dataset.append(example)
                for example in dataset["validation"]:
                    self.dataset.append(example)
                for example in dataset["test"]:
                    self.dataset.append(example)
            else:
                self.dataset = load_dataset("Iker/NoticIA", split=split_or_path)

        if prompt_name is not None:
            self.dataset_name += f"_{prompt_name}"

        self.data = []

        logging.info(
            f"Loading dataset {self.dataset_name}. use_clean_web_text = {use_clean_web_text}"
        )

        for example in tqdm(self.dataset, desc="Loading dataset"):
            formatted_data = prepare_data(
                tokenizer=tokenizer,
                inference=inference,
                headline=example["web_headline"].strip(),
                body=example[
                    "web_text" if not use_clean_web_text else "clean_web_text"
                ].strip(),
                summary=example["summary"].strip(),
                max_length=max_length,
                prompt_loss_weight=prompt_loss_weight,
                prompt_name=prompt_name,
            )
            if formatted_data is not None:
                self.data.append(formatted_data)

        logging.info(f"Loaded {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].copy()

    def get_summaries(self):
        return [x["summary"] for x in self.dataset]

    def get_name(self):
        return self.dataset_name


@dataclass
class DataCollatorForClickBait:
    """
    Adapted from transformers.DataCollatorForSeq2Seq to handle CoLLIE data.

    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
