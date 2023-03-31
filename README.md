# experiments-coreml-ane-distilbert

Experimenting with https://github.com/apple/ml-ane-transformers

Basically, Apple provide a version of DistilBERT model that should run on the Neural Engine (ANE) co-processor of Apple Silicon devices, when run via CoreML.

The specific model they provide has been pre-trained on the [SST 2](https://huggingface.co/datasets/sst2) dataset, so it provides sentiment classification (either "positive" or "negative").

This repo contains a trivial experiment with this model - a cli app you can run that will classify the sentiment of your inputs.

Prerequisites:

- an Apple Silicon mac
- Python 3.10
- [install Poetry](https://python-poetry.org/docs/#installation)

Prepare the environment:

```
poetry install
poetry shell
```

Example usage:

```
$ python server.py
Enter text to classify (or 'ctrl+D' to quit): I like cheese
Sentiment prediction: positive (95.75%)
Enter text to classify (or 'ctrl+D' to quit): I hate cheesecake
Sentiment prediction: negative (99.63%)
Enter text to classify (or 'ctrl+D' to quit):
```

With some logging we can observe the startup...

```
Loading CoreML model...
Loaded CoreML model in 4360.14ms
Loading tokenizer...
Loaded tokenizer in 406.96ms
```

...and inference times:

```
Tokenizing input...
Tokenized input in 0.89ms
Performing inference...
Inferred in 4.71ms
```

(My macbook air is a 1st gen M1, 16GB RAM)

The inference time seems to be highly variable. The first inference can take 100-300ms, but it gets much faster (single digits ms) if you make rapid-fire requests.  I am guessing perhaps it needs constant activity to have all the code/data primed in memory (cache?) in some way, and a short idle is enough for it to need reloading.  This is distinct from the ~5 seconds needed to load and initialise the CoreML model itself and the tokenizer.

## Is it really running on the ANE?

https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english says:

> PyTorch does not utilize the ANE, and running this version of the model with PyTorch on the CPU or GPU may actually be slower than the original. To take advantage of the hardware acceleration of the ANE, use the Core ML version of the model, **DistilBERT_fp16.mlpackage**.

So that is what we've done. But how to be sure?  After all, CoreML is kind of a black box that decides whether to use CPU, GPU or ANE depending on what it thinks is best for your model code.

I asked ChatGPT for suggestions how to do this and it hallucinated a couple of options:

```python
# `pip install nein`

# Use nein to check if the Neural Engine is being used
with nein.scope():
    outputs_coreml = mlmodel.predict({
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
    })
```

and

```python
import coremltools as ct

# If the ct.neural_engine.scope() call succeeds then the model is running on Neural Engine
with ct.neural_engine.scope():
    outputs_coreml = mlmodel.predict({
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
    })
```

These both look like exactly the kind of thing I want. Unfortunately it seems that neither of them actually exist 😞

Googling for myself I saw a few suggestions to use a Python library [asitop](https://github.com/tlkh/asitop) and watching if the ANE starts consuming any power. (this is )

I was unable to observe any flicker of power consumption from ANE when inferring individual phrases using the current `server.py`.

Either:

- this CoreML model doesn't actually run on the ANE as promised
- `asitop` doesn't work properly
- I need to batch process a big chunk of inferrences to get a noticeable power consumption

Alternatively, we could ask if ~4.5ms is a fast inference for this model, perhaps by running the unconverted version under PyTorch and comparing timings.

### Update

Using the following approach I was able to observe small power spikes on the ANE when running inference:

```
sudo powermetrics --sample-rate=500 | grep -i "ANE Power"
```

`powermetrics` is the Apple cli tool that generates the data consumed by `asitop`. Setting a short sample rate (in ms) allowed to observe the tiny spikes from my tiny inputs:

```
ANE Power: 0 mW
ANE Power: 29 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 0 mW
ANE Power: 37 mW
ANE Power: 0 mW
ANE Power: 0 mW
```
