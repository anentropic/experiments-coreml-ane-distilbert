import multiprocessing as mp
import numpy as np
import os
import sys

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import timer


"""
This is Apple's "ane-distilbert-base-uncased-finetuned-sst-2-english" model
but running under PyTorch instead of CoreML, so won't actually make use of
the ANE chip.
"""


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])

# prevent "Disabling parallelism to avoid deadlocks" warning from huggingface/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_REPO = "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"

MODEL_FILENAME = "DistilBERT_fp16.mlpackage"

MODEL_RESULT_KEYS = {
    0: "negative",
    1: "positive",
}


def _load_model():
    logger.debug("Loading CoreML model...")
    with timer() as timing:
        mlmodel = AutoModelForSequenceClassification.from_pretrained(
            MODEL_REPO, trust_remote_code=True, return_dict=False, revision="main"
        )
    logger.debug(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return mlmodel, tokenizer


def child_process(conn):
    try:
        mlmodel, tokenizer = _load_model()
    except:
        import traceback
        traceback.print_exc()
        conn.send(False)
        conn.close()
        return
    else:
        # Signal to the parent process that we are ready to receive inputs
        conn.send(True)

    while True:
        input_str = conn.recv()

        # If the input is None, exit the loop and terminate the child process
        if input_str is None:
            break

        logger.debug("Tokenizing input...")
        with timer() as timing:
            inputs = tokenizer(
                [input_str],
                return_tensors="pt",
                max_length=128,
                padding="max_length",
            )
        logger.debug(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")
        
        logger.debug("Performing inference...")
        with timer() as timing:
            with torch.no_grad():
                outputs = mlmodel(**inputs)
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        # Apply softmax to the logits output
        # (converts them into probabilities that sum to 1)
        probs = F.softmax(outputs[0], dim=1)

        conn.send(probs.tolist())

    # Clean up resources and exit the child process
    conn.close()


def run_server():
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child_process, args=(child_conn,))
    p.start()

    # Wait for the child process to signal that it's ready to receive inputs
    if not parent_conn.recv():
        raise RuntimeError("Child process failed to load model")

    try:
        while True:
            input_str = input("Enter text to classify (or 'ctrl+D' to quit): ")
            parent_conn.send(input_str)

            output = parent_conn.recv()
            prediction_index = np.argmax(output)
            print(
                f"Sentiment prediction: {MODEL_RESULT_KEYS[prediction_index]} "
                f"({output[0][prediction_index]:.2%})"
            )
    except (EOFError, KeyboardInterrupt):
        # tell child to close their conn and finish
        print()
        parent_conn.send(None)

    # Clean up resources and wait for the child process to terminate before exiting
    logger.debug("Waiting for child process to terminate...")
    p.join()
    logger.debug("Child process terminated.")


if __name__ == '__main__':
    run_server()
