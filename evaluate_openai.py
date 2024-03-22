from eval import evaluate_predictions
from openai import OpenAI
from prompts import summarize_clickbait_short_prompt
import json
from tqdm import tqdm
import os
import argparse
import time
from datasets import load_dataset

from typing import Dict, Union, Any

_PRICING = {
    "gpt-4-0125-preview_input": 0.01 / 1000,
    "gpt-4-0125-preview_output": 0.03 / 1000,
    "gpt-4-vision-preview_input": 0.01 / 1000,
    "gpt-4-vision-preview_output": 0.03 / 1000,
    "gpt-4_input": 0.03 / 1000,
    "gpt-4_output": 0.06 / 1000,
    "gpt-4-32k_input": 0.06 / 1000,
    "gpt-4-32k_output": 0.12 / 1000,
    "gpt-3.5-turbo-0125_input": 0.0005 / 1000,
    "gpt-3.5-turbo-0125_output": 0.0015 / 1000,
    "gpt-3.5-turbo_finetuned_input": 0.0030 / 1000,
    "gpt-3.5-turbo_finetuned_output": 0.0060 / 1000,
}


def compute_cost(model: str, data: Dict[str, Union[int, str, Any]]):
    """Compute the cost of using the given model with the given data.
    For gpt models data will be a dictionary with the key "input tokens" and "output tokens"

    Args:
        model (str): The model to use.
        data (Dict[str, str]): The input data.

    Returns:
        float: The cost of using the model with the given data in USD.
    """

    if "gpt" in model:
        return (
            data["prompt_tokens"] * _PRICING[f"{model}_input"]
            + data["completion_tokens"] * _PRICING[f"{model}_output"]
        )
    else:
        raise ValueError(
            f"Unsupported model: {model}. Supported models are: {list(_PRICING.keys())}"
        )


def evaluate_openai(model_name):
    """
    Evaluate the model on the Clickbait Challenge dataset.

    Args:
        model_name (str): The name of the model to evaluate.
    """

    # Prepare data
    dataset = load_dataset("Iker/NoticIA", split="test")
    total_cost = 0.0
    data = []
    for example in tqdm(dataset, desc="Loading dataset"):
        headline = example["web_headline"].strip()
        body = example["web_text"].strip()
        summary = example["summary"].strip()

        prompt = summarize_clickbait_short_prompt(headline, body)
        data.append(([{"role": "user", "content": prompt}], summary))

    # Evaluate
    print("Evaluating...")
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    os.makedirs(f"results/zero/{model_name}", exist_ok=True)

    outputs = []
    
    with tqdm(total=len(data), desc=f"Cost ${total_cost:.2f}") as pbar:
        for i, (prompt, summary) in enumerate(data):
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt,
            )
            total_cost += compute_cost(model=model_name, data=dict(completion.usage))
            output = completion.choices[0].message.content
            outputs.append({"prediction": output, "gold": summary})
            pbar.set_description(f"Cost ${total_cost:.2f}")
            pbar.update(1)
            time.sleep(0.1)

    with open(
        f"results/zero/{model_name}/predictions_test.json", "w", encoding="utf8"
    ) as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

    # Evaluate predictions
    results = evaluate_predictions(
        predictions_file=f"results/zero/{model_name}/predictions_test.json"
    )
    print(json.dumps(results, indent=4, ensure_ascii=False))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4-0125-preview",
        help="The name of the model to evaluate.",
    )
    args = parser.parse_args()

    evaluate_openai(args.model_name)

# python3 evaluate_openai.py --model_name gpt-4-0125-preview
# python3 evaluate_openai.py --model_name gpt-3.5-turbo-0125
