import json
import logging
import math
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from config import ModelArguments
from dataset import ClickBaitDataset
from eval import evaluate_predictions
from torch.utils.data import Dataset
from transformers import (
    DataCollator,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    StoppingCriteria,
    TrainingArguments,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    DataLoader,
    EvalLoopOutput,
    IterableDatasetShard,
    deepspeed_init,
    denumpify_detensorize,
    find_batch_size,
    has_length,
    logger,
    nested_concat,
    nested_numpify,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, PredictionOutput, speed_metrics
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME, is_safetensors_available

if is_safetensors_available():
    import safetensors.torch


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops: List[torch.tensor]):
        super().__init__()
        self.stops = stops
        logging.info(f"Stopping criteria words ids: {self.stops}")

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for seq in input_ids:
            for stop in self.stops:
                stop = stop.to(device=seq.device, dtype=seq.dtype)
                if (
                    len(seq) >= len(stop)
                    and torch.all((stop == seq[-len(stop) :])).item()
                ):
                    return True
        return False


class StopAfterTokenIsGenerated(LogitsProcessor):
    def __init__(self, stops: List[torch.tensor], eos_token_id: int):
        super().__init__()

        self.stops = stops
        self.eos_token_id = eos_token_id
        logging.info(f"Stopping criteria words ids: {self.stops}")
        self.first_batch = True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
                search or log softmax for each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

        """
        if self.first_batch:
            self.first_batch = False
            return scores

        for seq_no, seq in enumerate(input_ids):
            # logging.info(seq_no)
            for stop in self.stops:
                stop = stop.to(device=seq.device, dtype=seq.dtype)
                if (
                    len(seq) >= len(stop)
                    and torch.all((stop == seq[-len(stop) :])).item()
                ):
                    scores[seq_no, :] = -float("inf")
                    scores[seq_no, self.eos_token_id] = 0
                    # logging.info(f"Stopping criteria found: {stop}")
                    break

        return scores

    def reset(self):
        self.first_batch = True


class ClickbaitTrainer(Seq2SeqTrainer):
    """
    The ClickbaitTrainer is an adaptation of the 🤗 Transformers Trainer.

    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
            <Tip>
            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.
            </Tip>
        args ([`Seq2SeqTrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`Seq2SeqTrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise.
        train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`, *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.
            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.
            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values.
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.
            Note that the labels (second parameter) will be `None` if the dataset does not have them.
    Important attributes:
        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
          to `False` if model parallel or deepspeed is used, or if the default
          `TrainingArguments.place_model_on_device` is overridden to return `False` .
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        stop_words: Optional[List[List[int]]] = None,
    ):
        self.first_train_batch = True

        if tokenizer is None:
            raise ValueError(
                "You should supply a tokenizer for the correct functionality of the trainer"
            )

        if args.deepspeed is not None:
            try:
                leaf_module = model.model.layers[0].block_sparse_moe.__class__
            except AttributeError:
                leaf_module = None
                logging.warning(
                    "Deepspeed enabled. The current model is not a MoE model."
                )

            if leaf_module is not None:
                try:
                    from deepspeed.utils import set_z3_leaf_modules

                    set_z3_leaf_modules(model, [leaf_module])
                    logging.warning(
                        f"MoE model detected. We have used the deepspeed set_z3_leaf_modules function to ensure"
                        f" that the model works correctly. The leaf module used is {leaf_module}. "
                        f"See https://github.com/microsoft/DeepSpeed/pull/5008 for more details."
                    )
                except ImportError:
                    logging.warning(
                        "set_z3_leaf_modules function not found. You are not running the latest version of DeepSpeed. "
                        "You can safely ignore this warning if you are not using MoE models. If you are, "
                        "the training/inference will fail. Please update to the latest version of DeepSpeed "
                        "to fix this issue. More details at https://github.com/microsoft/DeepSpeed/pull/5008. "
                        "We will attempt to continue the training/inference."
                    )

                except Exception as e:
                    logging.warning(
                        f"MoE model detected. "
                        f"Something went wrong when trying to use the deepspeed set_z3_leaf_modules function. "
                        f"Please see https://github.com/microsoft/DeepSpeed/pull/5008 for more details.\n"
                        f"The leaf module used is {leaf_module}.\n"
                        f"Error message: {e}\n"
                        f"We will attempt to continue the training/inference, although it will probably freeze/crash."
                    )
        else:
            logging.warning("Deepspeed disabled")
        # HuggingFace mad TrainingArguments inmutable and therefore the next function crashes
        # We made TrainingArguments mutable again
        TrainingArguments.__setattr__ = object.__setattr__

        # Ensure that the values are floats
        args.set_optimizer(
            name=args.optim,
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            beta1=float(args.adam_beta1),
            beta2=float(args.adam_beta2),
            epsilon=float(args.adam_epsilon),
            args=args.optim_args,
        )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,  # We don't want to save the tokenizer with the model or use it for padding
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # Change the tqdm progress callback with `RichProgressCallback`
        # _prev_progress_callback = self.pop_callback(ProgressCallback)
        # if _prev_progress_callback:
        #    self.add_callback(RichProgressCallback)
        # print("init", eval_dataset)
        # print("init", self.eval_dataset)
        self.tokenizer = tokenizer
        self.stop_words = stop_words

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            raise ValueError("You should supply a labels key to compute the loss")

        if "loss_weight_mask" in inputs:
            loss_weight_mask = inputs.pop("loss_weight_mask")
        else:
            raise ValueError(
                "You should supply a loss_weight_mask key to compute the loss"
            )

        # Print first batch of training data for debugging
        if self.first_train_batch and self.tokenizer is not None:
            self.first_train_batch = False
            print_input_ids = inputs["input_ids"][:8].clone().detach().cpu()
            print_attention_mask = inputs["attention_mask"][:8].clone().detach().cpu()
            print_labels = labels[:8].clone().detach().cpu()
            print_loss_weight_mask = loss_weight_mask[:8].clone().detach().cpu()

            print("*** First batch of training data ***")
            print("-- input_ids --")
            if self.tokenizer is not None:
                print_input_ids[print_input_ids == -100] = self.tokenizer.pad_token_id
                print(self.tokenizer.batch_decode(print_input_ids))
            else:
                print(print_input_ids.tolist())
            print("-- attention_mask --")
            print(print_attention_mask.tolist())
            print("-- labels --")
            if self.tokenizer is not None:
                print_labels[print_labels == -100] = self.tokenizer.pad_token_id
                print(self.tokenizer.batch_decode(print_labels))
            else:
                print(print_labels[:8].tolist())
            print("-- loss_weight_mask --")
            print(print_loss_weight_mask.tolist())
            print()

        outputs = model(**inputs, use_cache=False)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # logging.info(f"logits size: {logits.size()}")
        # logging.info(f"labels size: {labels.size()}")

        model_name = unwrap_model(model)._get_name()
        if (
            model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            or model_name == "PeftModelForCausalLM"
        ):
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss_weight_mask = loss_weight_mask[..., 1:].contiguous()

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss_weight_mask = loss_weight_mask.view(-1).to(dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

        # logging.info(f"logits size: {logits.size()}")
        # logging.info(f"labels size: {labels.size()}\n\n")
        loss = loss_fct(logits, labels)
        loss = torch.sum(loss * loss_weight_mask) / torch.sum(loss_weight_mask)

        return (loss, outputs) if return_outputs else loss

    # Modify the Seq2SeqTrainer from transformers to only save the LoRA weights if we are using a LoRA model
    # Original trainer saves the full state dict. It doesn't make sense for us to create a full copy
    # of LLaMA weights each time we save a checkpoint since we do not modify them.
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()

        # Find out if the model is a LoRA Peft Model
        # try:
        #     from peft import PeftModel, LoraModel

        #     if isinstance(unwrap_model(self.model), PeftModel):
        #         if isinstance(unwrap_model(self.model).base_model, LoraModel):
        #             unwrap_model(self.model).save_pretrained(
        #                 output_dir,
        #             )
        #             return
        # except ImportError:
        #     pass

        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        if not isinstance(self.model, PreTrainedModel) and not (
            PeftModel and isinstance(self.model, PeftModel)
        ):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
                )
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME)
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def evaluate(
        self,
        eval_dataset: Optional[ClickBaitDataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            eval_dataset (`ClickBaitDataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        predictions = self.eval_predict(
            eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            step="validation",
        )

        predictions = predictions.predictions
        predictions[predictions == -100] = self.tokenizer.pad_token_id
        predictions[predictions == -100] = self.tokenizer.pad_token_id

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # predictions = [prediction.split("[/INST]")[-1] for prediction in predictions]
        predictions = [prediction.strip() for prediction in predictions]
        for i in range(len(predictions)):
            prediction = predictions[i]
            if "\n" in prediction:
                prediction = prediction.split("\n", maxsplit=1)[-1]
                prediction = prediction.replace("\n", " ")
                prediction = re.sub(r" +", " ", prediction)
                prediction = prediction.strip()
                predictions[i] = prediction

        output_dir = os.path.join(
            self.args.output_dir, f"checkpoint-{self.state.global_step}"
        )

        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(
            output_dir,
            f"predictions_{eval_dataset.get_name()}.json",
        )
        with open(os.path.join(output_filename), "w", encoding="utf8") as f:
            logging.info(f"Writing predictions to {output_filename}")
            outputs = []
            for prediction, gold in zip(predictions, eval_dataset.get_summaries()):
                outputs.append(
                    {"prediction": prediction, "gold": gold},
                )

            print(json.dump(outputs, f, ensure_ascii=False, indent=4))

        self.accelerator.wait_for_everyone()

        task_scores = evaluate_predictions(
            predictions_file=output_filename,
        )

        # Add test_ prefix to report test scores
        task_scores = {
            f"{metric_key_prefix}_{task}": score for task, score in task_scores.items()
        }
        task_scores["step"] = self.state.global_step
        if self.accelerator.is_main_process:
            logging.info(task_scores)

        self.log(task_scores)

        return task_scores

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> "PredictionOutput":
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if (
            gen_kwargs.get("num_beams") is None
            and self.args.generation_num_beams is not None
        ):
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        return self.eval_predict(
            test_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            step="test",
        )

    def eval_predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        step: str = "test",
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
            step: Validation or test step

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible

        self._memory_tracker.start()

        if step == "validation":
            # print("get_eval_dataloader")
            # print(f"test_dataset: {test_dataset}")
            test_dataloader = self.get_eval_dataloader(test_dataset)
            # print(f"test_dataloader: {test_dataloader}")
        else:
            # print(f"test_dataset: {test_dataset}")
            test_dataloader = self.get_test_dataloader(test_dataset)
            # print(f"test_dataloader: {test_dataloader}")

        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop

        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step

            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name])
                if args.include_inputs_for_metrics
                else None
            )

            # Update containers on host
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                losses_host = (
                    losses
                    if losses_host is None
                    else nested_concat(losses_host, losses, padding_index=-100)
                )
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100
                )
                inputs_decode = self.gather_function((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )

            if labels is not None:
                labels = self.gather_function((labels))
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[
                f"{metric_key_prefix}_jit_compilation_time"
            ] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape
            == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        if self.stop_words is not None:
            stop_criteria = StopAfterTokenIsGenerated(
                stops=[torch.tensor(stop_word) for stop_word in self.stop_words.copy()],
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            stop_criteria = None

        generated_tokens = self.model.generate(
            **generation_inputs,
            **gen_kwargs,
            # stopping_criteria=StoppingCriteriaList([self.stop_criteria]),
            logits_processor=LogitsProcessorList([stop_criteria])
            if stop_criteria is not None
            else None,
        )

        # Get only the new tokens
        generated_tokens = generated_tokens[
            :, generation_inputs["input_ids"].shape[1] :
        ]

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length
            )
        elif (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
            )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1
                )
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            elif self.tokenizer is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


def get_correct_torch_dtype(
    quantization: int,
    model_args: ModelArguments,
    training_args: Seq2SeqTrainingArguments,
) -> "str":
    """
    Returns the correct torch dtype based on the model and training arguments (if quantization is enabled).

    Args:
        quantization (`int`, optional):
            '4' or '8' for 4 bits or 8 bits quantization or None for 16/32bits training. Defaults to `None`.
        model_args (:class:`~transformers.ModelArguments`):
            The model arguments.
        training_args (:class:`~transformers.Seq2SeqTrainingArguments`):
            The training arguments.

    Returns:
        :obj:`str`: The correct torch dtype.
    """

    if isinstance(quantization, str):
        quantization = int(quantization)

    if quantization in [4, 8]:
        if training_args.fp16:
            if model_args.torch_dtype in ["auto", None]:
                logging.warning(
                    "Quantification and fp16 are enabled, but torch_dtype is not set. Setting torch_dtype to float16."
                )

            elif model_args.torch_dtype != "float16":
                logging.warning(
                    f"Quantification and fp16 are enabled, but torch_dtype is set to {model_args.torch_dtype}. "
                    "This can cause issues. We will override torch_dtype to float16."
                )
            return "float16"

        elif training_args.bf16:
            if model_args.torch_dtype in ["auto", None]:
                logging.warning(
                    "Quantification and bf16 are enabled, but torch_dtype is not set. Setting torch_dtype to bfloat16."
                )
            elif model_args.torch_dtype != "bfloat16":
                logging.warning(
                    f"Quantification and bf16 are enabled, but torch_dtype is set to {model_args.torch_dtype}. "
                    "This can cause issues. We will override torch_dtype to bfloat16."
                )
            return "bfloat16"

    return model_args.torch_dtype
