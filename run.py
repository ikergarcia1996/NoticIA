import gc
import glob
import json
import logging
import os
import sys
import re
import torch
import torch.utils.data


from config import DataTrainingArguments, ModelArguments
from dataset import ClickBaitDataset, DataCollatorForClickBait
from datasets import DatasetDict
from eval import evaluate_predictions
from model.load_model import load_model, merge_lora_model
from trainer import ClickbaitTrainer, get_correct_torch_dtype
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from torch.utils.data import ConcatDataset


def clean_cache():
    """Clean cache to avoid memory leak.
    This fixes this issue: https://github.com/huggingface/transformers/issues/22801"""

    logging.info(
        f"Cleaning GPU memory. Current memory usage: {torch.cuda.memory_allocated()}"
    )
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"GPU memory usage after cleaning: {torch.cuda.memory_allocated()}")


def train_clickbait(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
):
    logging.info(f"Loading {model_args.model_name_or_path} model...")

    model, tokenizer = load_model(
        inference=False,
        model_weights_name_or_path=model_args.model_name_or_path,
        quantization=model_args.quantization,
        use_lora=model_args.use_lora,
        lora_r=model_args.lora_r,
        lora_target_modules=model_args.lora_target_modules,
        torch_dtype=get_correct_torch_dtype(
            quantization=model_args.quantization,
            model_args=model_args,
            training_args=training_args,
        ),
        force_auto_device_map=model_args.force_auto_device_map,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        use_better_transformer=model_args.use_better_transformer,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention=model_args.use_flash_attention,
        fsdp_training=len(training_args.fsdp) > 1
        or training_args.fsdp_config is not None,
        max_memory_MB=model_args.max_memory_MB,
    )

    logging.info("Loading datasets...")

    training_datasets = []
    for train_dataset_no, train_dataset_path in enumerate(data_args.train_datasets):
        train_dataset = ClickBaitDataset(
            tokenizer=tokenizer,
            split_or_path=train_dataset_path,
            max_length=data_args.max_seq_length,
            inference=False,
            prompt_loss_weight=data_args.prompt_loss_weight,
            use_clean_web_text=data_args.use_clean_web_text,
            prompt_name=None
            if data_args.train_datasets_prompts is None
            else data_args.train_datasets_prompts[train_dataset_no],
        )
        training_datasets.append(train_dataset)

    train_dataset = ConcatDataset(training_datasets)

    dev_datasets = DatasetDict()
    for dev_dataset_no, dev_dataset_path in enumerate(data_args.validation_datasets):
        dev_dataset = ClickBaitDataset(
            tokenizer=tokenizer,
            split_or_path=dev_dataset_path,
            max_length=data_args.max_seq_length,
            inference=True,
            prompt_loss_weight=data_args.prompt_loss_weight,
            use_clean_web_text=data_args.use_clean_web_text,
            prompt_name=None
            if data_args.validation_datasets_prompts is None
            else data_args.validation_datasets_prompts[dev_dataset_no],
        )
        if data_args.validation_datasets_prompts is not None:
            name = f"{os.path.basename(dev_dataset_path)}_{data_args.validation_datasets_prompts[dev_dataset_no]}"
        else:
            name = os.path.basename(dev_dataset_path)
        dev_datasets[name] = dev_dataset

    trainer = ClickbaitTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets,
        args=training_args,
        data_collator=DataCollatorForClickBait(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
        ),
        stop_words=None
        if model_args.stop_words is None
        else [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in model_args.stop_words
        ],
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Save the model
    # trainer.save_model()


def inference_clickbait(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    checkpoint_path: str = None,
):
    if not training_args.predict_with_generate:
        logging.warning(
            "You have set predict_with_generate to False. We will only compute the loss"
            " on the test set. If you want to generate predictions, set"
            " predict_with_generate to True."
        )

        if not training_args.prediction_loss_only:
            logging.warning(
                "You have set predict_with_generate to False, so you only "
                "want to compute the loss on the test set. But you have set "
                "prediction_loss_only to False. This is contradictory, please "
                "review you configuration. We will attempt to continue but "
                "you might get unexpected results."
            )

    if training_args.do_train:
        if not checkpoint_path:
            logging.warning(
                "You are doing inference after training a model! We will load the "
                f"pretrained model saved in {training_args.output_dir}."
            )
            if model_args.use_lora:
                model_path = model_args.model_name_or_path
                lora_weights_name_or_path = training_args.output_dir
            else:
                model_path = training_args.output_dir
                lora_weights_name_or_path = None
        else:
            logging.warning(
                "You are doing inference after training a model! We will load the "
                f"pretrained model saved in {checkpoint_path}."
            )
            if model_args.use_lora:
                model_path = model_args.model_name_or_path
                lora_weights_name_or_path = checkpoint_path
            else:
                model_path = checkpoint_path
                lora_weights_name_or_path = None
    else:
        if not checkpoint_path:
            model_path = model_args.model_name_or_path
            lora_weights_name_or_path = model_args.lora_weights_name_or_path
        else:
            if model_args.use_lora:
                model_path = model_args.model_name_or_path
                lora_weights_name_or_path = checkpoint_path
            else:
                model_path = checkpoint_path
                lora_weights_name_or_path = None

    if model_args.use_lora and lora_weights_name_or_path is None:
        logging.warning(
            "You are have specified to use LORA, but have not specified a path to the "
            "LORA weights. We will attempt to load the LORA weights from the same "
            f"path as the model weights: {model_path}."
        )

    delete_merged_model: bool = False
    if model_args.merge_lora_before_inference:
        logging.info(
            "You have specified to merge the LORA weights before inference. We will attempt to do so."
        )
        if model_args.quantization_inference is None:
            logging.warning(
                "You have specified to create a merged model (merge_lora_before_inference=True), but you have not"
                " specified a quantization precision. Model loads without quantization are automatically merged when"
                " loaded for inference, so there is no need to save a merged model and reaload it. This flag is only"
                " useful when you want to merge a model and then load it using 4 bits ot 8 bits quantization or if you"
                " plan to release the merged model."
            )
        if os.path.exists(os.path.join(training_args.output_dir, "merged_model")):
            logging.info(
                f"A merged model already exists at {os.path.join(training_args.output_dir,'merged_model')}. We will"
                " use this model."
            )
            delete_merged_model = False

        else:
            merge_lora_model(
                weights_path=model_path,
                lora_weights_name_or_path=lora_weights_name_or_path,
                torch_dtype=model_args.torch_dtype,
                output_path=os.path.join(training_args.output_dir, "merged_model"),
            )
            delete_merged_model = not model_args.keep_merged_model_after_eval

        model_path = os.path.join(training_args.output_dir, "merged_model")
        lora_weights_name_or_path = None
        clean_cache()  # Ensure that nothing remains in the cache, as we will load the mergen model next.

    model, tokenizer = load_model(
        inference=True,
        model_weights_name_or_path=model_path,
        quantization=model_args.quantization_inference,
        use_lora=lora_weights_name_or_path is not None,
        lora_weights_name_or_path=lora_weights_name_or_path,
        force_auto_device_map=model_args.force_auto_device_map,
        torch_dtype=get_correct_torch_dtype(
            quantization=model_args.quantization_inference,
            model_args=model_args,
            training_args=training_args,
        ),
        use_better_transformer=model_args.use_better_transformer,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention=model_args.use_flash_attention,
        max_memory_MB=model_args.max_memory_MB,
    )

    trainer = ClickbaitTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForClickBait(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        ),
        stop_words=None
        if model_args.stop_words is None
        else [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in model_args.stop_words
        ],
    )

    output_dir = (
        training_args.output_dir if checkpoint_path is None else checkpoint_path
    )

    for test_dataset_no, test_dataset_path in enumerate(data_args.test_datasets):
        test_dataset = ClickBaitDataset(
            tokenizer=tokenizer,
            split_or_path=test_dataset_path,
            max_length=data_args.max_seq_length,
            inference=True if training_args.predict_with_generate else False,
            prompt_loss_weight=0.0,
            use_clean_web_text=data_args.use_clean_web_text,
            prompt_name=None
            if data_args.test_datasets_prompts is None
            else data_args.test_datasets_prompts[test_dataset_no],
        )

        logging.info(f"Running inference on {test_dataset_path}")
        predictions = trainer.predict(test_dataset)

        predictions_name = (
            "test"
            if test_dataset_path == "test"
            else os.path.basename(os.path.splitext(test_dataset_path)[0])
        )

        if data_args.test_datasets_prompts is not None:
            predictions_name += f"_{data_args.test_datasets_prompts[test_dataset_no]}"

        predictions_output = os.path.join(
            output_dir,
            f"predictions_{predictions_name}.json",
        )

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                with open(predictions_output, "w", encoding="utf8") as f:
                    logging.info(f"Writing predictions to {predictions_output}")
                    predictions = predictions.predictions
                    # Switch all -100 to tokenizer.pad_token_id, so we can decode the predictions
                    predictions[predictions == -100] = tokenizer.pad_token_id

                    try:
                        predictions = tokenizer.batch_decode(
                            predictions, skip_special_tokens=True
                        )

                        # predictions = [
                        #    prediction.split(get_start_response_token(tokenizer))[-1]
                        #    for prediction in predictions
                        # ]
                        predictions = [prediction.strip() for prediction in predictions]

                        for i in range(len(predictions)):
                            prediction = predictions[i]
                            if "\n" in prediction:
                                prediction = prediction.split("\n", maxsplit=1)[-1]
                                prediction = prediction.replace("\n", " ")
                                prediction = re.sub(r" +", " ", prediction)
                                prediction = prediction.strip()
                                predictions[i] = prediction

                    except OverflowError:
                        raise OverflowError(
                            f"Unable to decode predictions: {predictions}"
                        )

                    outputs = []
                    for prediction, gold in zip(
                        predictions, test_dataset.get_summaries()
                    ):
                        outputs.append(
                            {"prediction": prediction, "gold": gold},
                        )

                    print(json.dump(outputs, f, ensure_ascii=False, indent=4))
            else:
                metrics_name = os.path.join(
                    output_dir,
                    f"predictions_{os.path.basename(predictions_name)}.metrics.txt",
                )
                with open(metrics_name, "w", encoding="utf8") as f:
                    logging.info(f"Writing metrics to {metrics_name}")
                    json.dump(predictions.metrics, fp=f, ensure_ascii=False, indent=4)

        if training_args.predict_with_generate and trainer.is_world_process_zero():
            task_scores = evaluate_predictions(
                predictions_file=predictions_output,
            )
            # Add test_ prefix to report test scores
            task_scores = {f"test_{task}": score for task, score in task_scores.items()}
            # Report
            if checkpoint_path is not None:
                step = int(checkpoint_path.split("-")[-1])
                task_scores["step"] = step

            trainer.log(task_scores)

    if delete_merged_model:
        logging.info(f"Deleting merged model at {model_path}")
        import shutil

        try:
            shutil.rmtree(model_path)
        except OSError as e:
            logging.error(
                f"Unable to delete the merged model {model_path} : {e.strerror}\n"
                "You may need to delete the merged model manually."
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    logging.info(f"Sys args {sys.argv}")

    if len(sys.argv) > 0 and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        logging.info(f"Loading json config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1])
        )

    elif len(sys.argv) > 0 and sys.argv[-1].endswith(".yaml"):
        # If we pass only one argument to the script, and it's the path to a yaml file,
        # let's parse it to get our arguments.
        logging.info(f"Loading yaml config {sys.argv[-1]}")
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[-1])
        )
    else:
        logging.info("No config file passed, using command line arguments.")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.do_train and data_args.train_datasets is not None:
        train_clickbait(
            model_args,
            data_args,
            training_args,
        )
        clean_cache()
        if model_args.use_lora and model_args.merge_lora_after_training:
            merge_lora_model(
                weights_path=model_args.model_name_or_path,
                lora_weights_name_or_path=training_args.output_dir,
                torch_dtype=model_args.torch_dtype,
                output_path=os.path.join(training_args.output_dir, "merged_model"),
            )
            clean_cache()

    if training_args.do_predict and data_args.test_datasets is not None:
        if not data_args.evaluate_all_checkpoints:
            inference_clickbait(
                model_args,
                data_args,
                training_args,
            )
            clean_cache()
        else:
            # Find all checkpoints in the output directory
            checkpoints = [
                c
                for c in glob.glob(
                    os.path.join(training_args.output_dir, "checkpoint-*"),
                )
                if os.path.isdir(c)
            ]

            # Sort checkpoints by step number
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

            logging.info(
                f"Found {len(checkpoints)} checkpoints in {training_args.output_dir}:"
                f" {checkpoints} . We will evaluate each of them."
            )

            # Evaluate each checkpoint
            for checkpoint in checkpoints:
                inference_clickbait(
                    model_args,
                    data_args,
                    training_args,
                    checkpoint_path=checkpoint,
                )
                clean_cache()
