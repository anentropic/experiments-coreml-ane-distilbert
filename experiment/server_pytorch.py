import argparse
import multiprocessing as mp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .loader import load_pytorch, MODEL_REPO
from .utils import timer


"""
This is Apple's "ane-distilbert-base-uncased-finetuned-sst-2-english" model
but running under PyTorch instead of CoreML, so won't actually make use of
the ANE chip.

Can also pass a model name as an argument to the script to load a different
model from HuggingFace.
"""


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])


MODEL_RESULT_KEYS = {
    0: "NEGATIVE",
    1: "POSITIVE",
}


def child_process(conn, model_name: str, use_mps: bool):
    try:
        model, tokenizer = load_pytorch(model_name)
    except:
        import traceback
        traceback.print_exc()
        conn.send(False)
        conn.close()
        return

    device = None
    if use_mps:
        device = torch.device('mps')
        model.to(device)

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

        if use_mps:
            inputs.to(device)

        logger.debug("Performing inference...")
        with timer() as timing:
            with torch.no_grad():
                outputs = model(**inputs)
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        logger.debug(outputs)

        # Apply softmax to the logits output
        # (converts them into probabilities that sum to 1)
        probs = F.softmax(outputs[0], dim=1)

        conn.send(probs.tolist())

    # Clean up resources and exit the child process
    conn.close()


def run_server():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model", type=str, default=MODEL_REPO,
        help="Name of the model to load from HuggingFace"
    )
    argparser.add_argument(
        "--use-mps", action="store_true",
        help="Use the MPS backend for inference"
    )
    args = argparser.parse_args()

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child_process, args=(child_conn, args.model, args.use_mps))
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
