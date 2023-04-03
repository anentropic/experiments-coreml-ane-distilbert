import multiprocessing as mp
import sys
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .loader import load_coreml
from .utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])


@dataclass(frozen=True)
class Result:
    probabilities: dict[float]
    predicted_class: str


def get_apple_result(outputs):
    """
    Parse the output from the Apple version of the model
    """
    exp_ = np.exp(outputs['logits'])
    # Apply softmax to the logits output
    # (converts them into probabilities that sum to 1)
    probs = exp_ / np.sum(exp_)
    prediction_index = np.argmax(probs)  # 0 or 1
    return Result(
        probabilities={
            "NEGATIVE": probs[0][0],
            "POSITIVE": probs[0][1],
        },
        predicted_class="POSITIVE" if prediction_index else "NEGATIVE",
    )


def get_exported_result(outputs):
    """
    Parse the output from a version of the model exported via
    https://github.com/huggingface/exporters
    """
    return Result(
        probabilities=outputs['probabilities'],
        predicted_class=outputs['classLabel'],
    )


def child_process(conn, model_path: str | None):
    try:
        model, tokenizer = load_coreml(model_path)
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
                return_tensors="np",
                max_length=128,
                padding="max_length",
            )
        logger.debug(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")
        
        logger.debug("Performing inference...")
        with timer() as timing:
            outputs = model.predict({
                "input_ids": inputs["input_ids"].astype(np.int32),
                "attention_mask": inputs["attention_mask"].astype(np.int32),
            })
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        logger.debug(outputs)

        if model_path:
            result = get_exported_result(outputs)
        else:
            result = get_apple_result(outputs)

        conn.send(result)

    # Clean up resources and exit the child process
    conn.close()


def run_server():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child_process, args=(child_conn, model_path))
    p.start()

    # Wait for the child process to signal that it's ready to receive inputs
    if not parent_conn.recv():
        raise RuntimeError("Child process failed to load model")

    try:
        while True:
            input_str = input("Enter text to classify (or 'ctrl+D' to quit): ")
            parent_conn.send(input_str)

            result = parent_conn.recv()
            print(
                f"Sentiment prediction: {result.predicted_class} "
                f"({result.probabilities[result.predicted_class]:.2%})"
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
