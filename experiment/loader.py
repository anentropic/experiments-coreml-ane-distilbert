import os
import sys
import tempfile

import coremltools as ct
from huggingface_hub import snapshot_download, try_to_load_from_cache
from loguru import logger
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])

# prevent "Disabling parallelism to avoid deadlocks" warning from huggingface/tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


TokenizerT = PreTrainedTokenizer | PreTrainedTokenizerFast

MODEL_NAME = "apple/ane-distilbert-base-uncased-finetuned-sst-2-english"

MODEL_FILENAME = "DistilBERT_fp16.mlpackage"


def _pre_cache():
    if not try_to_load_from_cache(MODEL_NAME, f"{MODEL_FILENAME}/Manifest.json"):
        logger.debug("Pre-caching model in local huggingface hub...")
        snapshot_download(
            repo_id=MODEL_NAME,
            allow_patterns=f"{MODEL_FILENAME}/*",
        )


def _load_coreml_model(model_path: str) -> ct.models.MLModel:
    logger.debug(f"Loading CoreML model '{model_path}'...")
    with timer() as timing:
        model = ct.models.MLModel(model_path)
    logger.debug(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")
    return model


def load_coreml(
    local_path: str | None = None,
    tokenizer_model_name: str = MODEL_NAME,
) -> tuple[ct.models.MLModel, TokenizerT]:
    """
    The .mlpackage is a dir so we have to use snapshot_download to download it,
    and we have to give a local path and opt out of symlinks, otherwise the
    model loader will fail (can't follow symlinks I guess). Fortunately we can
    still get the benefit of caching via the hub library.
    """
    if local_path:
        model = _load_coreml_model(local_path)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.debug("Downloading CoreML model...")
            with timer() as timing:
                # ensure it's cached first (local_dir_use_symlinks=False will use the cache
                # if present already, but won't fill it)
                _pre_cache()
                snapshot_path = snapshot_download(
                    repo_id=MODEL_NAME,
                    allow_patterns=f"{MODEL_FILENAME}/*",
                    local_dir=tmp_dir,
                    local_dir_use_symlinks=False,
                )
            logger.debug(f"Downloaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

            model = _load_coreml_model(f"{snapshot_path}/{MODEL_FILENAME}")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer


def load_pytorch(model_name: str):
    logger.debug(f"Loading PyTorch model '{model_name}'...")
    with timer() as timing:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True, return_dict=False, revision="main"
        )
    logger.debug(f"Loaded PyTorch model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer
