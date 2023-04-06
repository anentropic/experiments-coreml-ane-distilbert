import multiprocessing as mp
import sys

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


def child_process(conn, model_path: str | None):
    try:
        model, tokenizer = load_coreml(model_path, "bert-base-uncased")
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
        input_vals = conn.recv()

        # If the input is None, exit the loop and terminate the child process
        if input_vals is None:
            break
        
        context, question = input_vals

        logger.debug("Tokenizing input...")
        with timer() as timing:
            inputs = tokenizer(
                question,
                context,
                return_tensors="np",
                padding="max_length",
                max_length=384,
                return_token_type_ids=True,
            )
        logger.debug(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")
        
        logger.debug("Performing inference...")
        with timer() as timing:
            outputs = model.predict({
                "wordIDs": inputs["input_ids"].astype(np.int32),
                "wordTypes": inputs["token_type_ids"].astype(np.int32),
            })
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        # logger.debug(outputs)

        start_logits = outputs["startLogits"][0]
        end_logits = outputs["endLogits"][0]

        start_index = int(np.argmax(start_logits))
        end_index = int(np.argmax(end_logits))

        answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
        answer = tokenizer.decode(answer_tokens)

        conn.send(answer)

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
            context = input("Enter your context (or 'ctrl+D' to quit): ")
            if not context:
                context = "My name is Paul and I live in Cambridge"
                print(f"Using default context: \"{context}\"")
            question = input("Enter your question (or 'ctrl+D' to quit): ")
            if not question:
                question = "Where do I live?"
                print(f"Using default question: \"{question}\"")
            parent_conn.send((context, question))

            answer = parent_conn.recv()
            print(f"Answer: {answer}")
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
