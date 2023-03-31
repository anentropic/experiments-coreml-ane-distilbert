import coremltools as ct
import numpy as np
import torch
import transformers
from ane_transformers.huggingface import distilbert as ane_distilbert


"""
This is the code from:
https://github.com/apple/ml-ane-transformers

The end result is ANE DistilBERT converted to CoreML package format.

We don't need to run this as we can just download the result from:
https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english
(except to test our toolchain is working - it is!)
From that link we can derive the code in run_coreml.py
"""

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
baseline_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_name,
    return_dict=False,
    torchscript=True,
).eval()

optimized_model = ane_distilbert.DistilBertForSequenceClassification(
    baseline_model.config).eval()

optimized_model.load_state_dict(baseline_model.state_dict())

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenized = tokenizer(
    ["Sample input text to trace the model"],
    return_tensors="pt",
    max_length=128,  # token sequence length
    padding="max_length",
)

traced_optimized_model = torch.jit.trace(
    optimized_model,
    (tokenized["input_ids"], tokenized["attention_mask"])
)

ane_mlpackage_obj = ct.convert(
    traced_optimized_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(
            f"input_{name}",
            shape=tensor.shape,
            dtype=np.int32,
        )
        for name, tensor in tokenized.items()
    ],
    compute_units=ct.ComputeUnit.ALL,
)
out_path = "HuggingFace_ane_transformers_distilbert_seqLen128_batchSize1.mlpackage"
ane_mlpackage_obj.save(out_path)
