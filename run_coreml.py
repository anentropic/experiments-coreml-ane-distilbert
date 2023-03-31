import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from utils import timer

"""
adapted from:
https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english
(incomplete example at bottom of page)

see server.py for a usable program built on this example
"""

print("Loading tokenizer...")
with timer() as timing:
    tokenizer = AutoTokenizer.from_pretrained(
        "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"
    )
print(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

print("Tokenizing input...")
with timer() as timing:
    inputs = tokenizer(
        ["The Neural Engine is really fast"],
        return_tensors="np",
        max_length=128,
        padding="max_length",
    )
print(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")

print("Loading CoreML model...")
with timer() as timing:
    mlmodel = ct.models.MLModel("DistilBERT_fp16.mlpackage")
print(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

print("Running CoreML model...")
with timer() as timing:
    outputs_coreml = mlmodel.predict({
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
    })
print(f"Ran CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

print(outputs_coreml)

# apply softmax to the logits output
# (converts them into probabilities that sum to 1)
exp_ = np.exp(outputs_coreml['logits'])
probs_output = exp_ / np.sum(exp_)

print(probs_output[0])

prediction = np.argmax(probs_output, axis=1)

logit_key = {
    0: "negative",
    1: "positive",
}

print(f"Sentiment prediction: {logit_key[prediction[0]]} ({probs_output[0][prediction[0]]:.2%})")
