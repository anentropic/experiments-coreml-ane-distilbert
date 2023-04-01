import os
import sys
import tempfile

import coremltools as ct
from huggingface_hub import snapshot_download, try_to_load_from_cache
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import timer


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


def _pre_cache():
    if not try_to_load_from_cache(MODEL_REPO, f"{MODEL_FILENAME}/Manifest.json"):
        logger.debug("Pre-caching model in local huggingface hub...")
        snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns=f"{MODEL_FILENAME}/*",
        )


def load_coreml():
    """
    The .mlpackage is a dir so we have to use snapshot_download to download it,
    and we have to give a local path and opt out of symlinks, otherwise the
    model loader will fail (can't follow symlinks I guess). Fortunately we can
    still get the benefit of caching via the hub library.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.debug("Downloading CoreML model...")
        with timer() as timing:
            # ensure it's cached first (local_dir_use_symlinks=False will use the cache
            # if present already, but won't fill it)
            _pre_cache()
            snapshot_path = snapshot_download(
                repo_id=MODEL_REPO,
                allow_patterns=f"{MODEL_FILENAME}/*",
                local_dir=tmp_dir,
                local_dir_use_symlinks=False,
            )
        logger.debug(f"Downloaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

        logger.debug("Loading CoreML model...")
        with timer() as timing:
            model = ct.models.MLModel(f"{snapshot_path}/{MODEL_FILENAME}")
        logger.debug(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer


def load_pytorch():
    logger.debug("Loading CoreML model...")
    with timer() as timing:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_REPO, trust_remote_code=True, return_dict=False, revision="main"
        )
    logger.debug(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer
