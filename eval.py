import json
import logging
from evaluate import load
from typing import Dict
import os
import random
import string


def clean_text(text: str) -> str:
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove newlines and multiple spaces
    text = text.replace("\n", " ").strip()
    text = " ".join(text.split()).strip()

    # lowercase
    text = text.lower()

    return text


def evaluate_predictions(
    predictions_file: str,
) -> Dict[str, float]:
    """
    Args:
        The path to the predictions file in json format. It must be a list of dictionaries
        with the following keys: "prediction" and "gold".

    Returns:
        Dict[str, float]: A dictionary containing the scores for each task
        present in the dataset.
    """

    results_scores: Dict[str, float] = {}

    with open(predictions_file, "r", encoding="utf8") as file:
        predictions = json.load(file)
        logging.info("Loading rouge...")
        # Random string
        experiment_id = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        rouge = load("rouge", experiment_id=experiment_id)
        logging.info("Loading sacrebleu...")
        bleu = load("sacrebleu", experiment_id=experiment_id)
        logging.info("Evaluating predictions...")
        for example in predictions:
            clean_prediction = clean_text(example["prediction"])
            clean_summary = clean_text(example["gold"])

            rouge.add_batch(
                predictions=[clean_prediction], references=[[clean_summary]]
            )
            bleu.add_batch(predictions=[clean_prediction], references=[[clean_summary]])

        bleu = bleu.compute()
        rouge = rouge.compute(use_aggregator=True, rouge_types=["rouge1"])

        result_name = os.path.basename(
            os.path.basename(os.path.splitext(predictions_file)[0])
        )
        results_scores[f"{result_name}/bleu"] = bleu["score"]
        results_scores[f"{result_name}/rouge"] = rouge["rouge1"]

    metrics_file = os.path.join(
        os.path.dirname(predictions_file), f"{result_name}_metrics.json"
    )

    with open(metrics_file, "w", encoding="utf8") as f:
        json.dump(results_scores, f, indent=4, ensure_ascii=False)

    logging.info(f"Results saved in {metrics_file}")

    return results_scores
