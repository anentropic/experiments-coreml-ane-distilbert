import argparse
import sys

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger

from loader import load_coreml, load_pytorch
from utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])


MODEL_RESULT_KEYS = {
    0: "negative",
    1: "positive",
}


def load_data() -> list[str]:
    """
    Load the SST-2 dataset from Hugging Face's datasets library.
    
    The model contains three datasets: train, validation, and test.
    The items in the datasets are dictionaries of the form:
        {
            "idx": 0,  # a unique identifier for the item
            "sentence": "...",
            "label": 1,  # 0 or 1, the expected classification
        }

    We will take just the sentences from the 'test' dataset to use as inputs.
    """
    logger.debug("Loading dataset...")
    with timer() as timing:
        dataset = load_dataset("sst2")
    logger.debug(f"Loaded dataset in {timing.execution_time_ns / 1e6:.2f}ms")

    return [
        item['sentence']
        for item in dataset['test'].to_iterable_dataset()
    ]


def coreml(inputs: list[str]):
    model, tokenizer = load_coreml()

    logger.debug("Tokenizing inputs...")
    encoded_inputs = []
    for input_ in inputs:
        encoded = tokenizer(
            [input_],
            return_tensors="np",
            max_length=128,
            padding="max_length",
        )
        input_ids = encoded["input_ids"].astype(np.int32)
        attention_mask = encoded["attention_mask"].astype(np.int32)
        encoded_inputs.append((input_ids, attention_mask))

    logger.debug("Performing inference...")
    with timer() as timing:
        for input_ids, attention_mask in encoded_inputs:
            model.predict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
    print(f"Inferred {len(inputs)} inputs in {timing.execution_time_ns / 1e6:.2f}ms")


def pytorch(inputs: list[str]):
    model, tokenizer = load_pytorch()
        
    logger.debug("Tokenizing inputs...")
    encoded_inputs = [
        tokenizer(
            [input_],
            return_tensors="pt",
            max_length=128,
            padding="max_length",
        )
        for input_ in inputs
    ]

    logger.debug("Performing inference...")
    with timer() as timing:
        with torch.no_grad():
            for encoded in encoded_inputs:
                model(**encoded)
    print(f"Inferred {len(inputs)} inputs in {timing.execution_time_ns / 1e6:.2f}ms")


if __name__ == "__main__":
    inputs = load_data()

    parser = argparse.ArgumentParser()
    parser.add_argument("--coreml", action="store_true")
    parser.add_argument("--pytorch", action="store_true")
    args = parser.parse_args()

    if not (args.coreml or args.pytorch):
        print("Please specify --coreml or --pytorch")
        sys.exit(1)
    if args.coreml:
        coreml(inputs)
    if args.pytorch:
        pytorch(inputs)
