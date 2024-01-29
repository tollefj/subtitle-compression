from time import time
from typing import List

import numpy as np

from utils.data_utils import get_data, get_split


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, evaluator, tokenizer):
    print("Computing metrics...")

    start_time = time()
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels
    )
    preds = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]

    results = {}
    result = evaluator.get_metrics(
        predictions=decoded_preds, references=decoded_labels
    )
    # result["gen_len"] = np.mean(preds)

    results["regular"] = result
    results["grounded"] = evaluator.get_metrics(
        decoded_preds, decoded_labels, ground=True
    )
    results["lead8"] = evaluator.get_metrics(
        decoded_preds, decoded_labels, lead_n=8
    )
    results["lead16"] = evaluator.get_metrics(
        decoded_preds, decoded_labels, lead_n=16
    )

    # round all values to 4
    for k, v in results.items():
        results[k] = {k: round(v, 4) for k, v in results[k].items()}

    end_time = time()
    print(f"Finished computing metrics in {end_time - start_time} seconds")
    return result


def tokenize_data(
    data_path,
    compression_rate,
    tokenizer,
    lang1,
    lang2,
    prefix,
    max_input_length=256,
    max_target_length=256,
    sizes: List[int] = None,
):
    dataset = get_data(data_path, lang1, lang2, compression_rate)

    if sizes:
        dataset["train"] = (
            dataset["train"].shuffle(seed=42).select(range(sizes[0]))
        )
        dataset["test"] = (
            dataset["test"].shuffle(seed=42).select(range(sizes[1]))
        )
        dataset["validation"] = (
            dataset["validation"].shuffle(seed=42).select(range(sizes[2]))
        )

    def mapping(examples):
        if prefix:
            inputs = [f"{prefix} {ex}" for ex in examples[lang1]]
        else:
            inputs = examples[lang1]
        targets = examples[lang2]
        model_inputs = tokenizer(
            inputs, max_length=max_input_length, truncation=True
        )
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(mapping, batched=True)
