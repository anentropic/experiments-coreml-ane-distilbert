# experiments-coreml-ane-distilbert

Experimenting with https://github.com/apple/ml-ane-transformers

Basically, Apple provide a version of DistilBERT model that should run on the Neural Engine (ANE) co-processor of Apple Silicon devices, when run via CoreML. It is derived from `bert-base-uncased` which was a 110M param model (but DistilBERT may be smaller?).

The specific model they provide has been pre-trained on the [SST 2](https://huggingface.co/datasets/sst2) dataset, so it provides sentiment classification (either "positive" or "negative"). Being "uncased" it doesn't care about capitalisation of the input.

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
$ python -m experiment.server
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

Googling for myself I saw a few suggestions to use a Python library [asitop](https://github.com/tlkh/asitop) and watching if the ANE starts consuming any power. Fortunately most of the time the ANE is doing nothing, so we have a good baseline to observe against.

I was unable to observe any flicker of power consumption from ANE in `asitop` when inferring individual phrases using the current `experiment/server.py`.

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

### Update

I implemented the PyTorch version as `experiment/server_pytorch.py`.

Roughly I see rapid-fire cli queries take 50-70ms, so it's approx 10x slower. And no power spikes on the ANE as expected, it's not being used.

One curious thing I observed is it doesn't always give the same prediction as the CoreML model (!)

For example:
```
$ python -m experiment.server
Enter text to classify (or 'ctrl+D' to quit): I like cheese
Tokenizing input...
Tokenized input in 0.57ms
Performing inference...
Inferred in 139.39ms
{'logits': array([[-1.5722656,  1.5429688]], dtype=float32)}
Sentiment prediction: POSITIVE (95.75%)

$ python -m experiment.server_pytorch
Enter text to classify (or 'ctrl+D' to quit): I like cheese
Tokenizing input...
Tokenized input in 0.63ms
Performing inference...
Inferred in 70.10ms
(tensor([[ 0.1999, -0.2167]]),)
Sentiment prediction: NEGATIVE (60.27%)
```

Other inputs give more similar results:
```
$ python -m experiment.server
Enter text to classify (or 'ctrl+D' to quit): I like cheesecake
Tokenizing input...
Tokenized input in 0.61ms
Performing inference...
Inferred in 131.57ms
{'logits': array([[-2.9511719,  3.0585938]], dtype=float32)}
Sentiment prediction: POSITIVE (99.76%)

$ python -m experiment.server_pytorch
Enter text to classify (or 'ctrl+D' to quit): I like cheesecake
Tokenizing input...
Tokenized input in 0.61ms
Performing inference...
Inferred in 71.98ms
(tensor([[-2.7818,  2.8868]]),)
Sentiment prediction: POSITIVE (99.66%)
```

### Update

To avoid having to manually enter prompts in quick succession via the cli, I made `experiments/benchmark.py`.

This loads the SST2 dataset (which the model was originally trained on) and runs inference on the 1821 examples in the "test" segment of the dataset.

Unfortunately the `coremltools` Python library [does not support batch inference](https://github.com/apple/coremltools/issues/196) (which is available in CoreML via ObjC) ... I would imagine batch inference is possible in PyTorch too. So I have just done a series of individual inferences in a tight loop, after pre-prepping the tokens etc.

We can see some expected results:

```
$ python -m experiment.benchmark --coreml
Inferred 1821 inputs in 7980.14ms

$ python -m experiment.benchmark --pytorch
Inferred 1821 inputs in 80253.39ms
```

We average ~4.4ms per inference with CoreML on the ANE, and ~44s under PyTorch on the CPU (or GPU?). So the CoreML version accelerated on the ANE is almost exactly 10x faster.

We can see a bigger power spike on the ANE this time:
```
ANE Power: 0 mW
ANE Power: 8 mW
ANE Power: 1325 mW
ANE Power: 1460 mW
ANE Power: 1658 mW
ANE Power: 1732 mW
ANE Power: 1819 mW
ANE Power: 1461 mW
ANE Power: 1282 mW
ANE Power: 1414 mW
ANE Power: 2775 mW
ANE Power: 2510 mW
ANE Power: 2331 mW
ANE Power: 1542 mW
ANE Power: 1587 mW
ANE Power: 1494 mW
ANE Power: 1405 mW
ANE Power: 891 mW
ANE Power: 0 mW
```

This is big enough to observe with `asitop` too:

![Screenshot 2023-04-01 at 11 06 54](https://user-images.githubusercontent.com/147840/229279886-f45400ab-16b8-41ab-891f-5329867e983a.png)

### Update

I tried tracking the memory usage when running the benchmarks, using [memray](https://bloomberg.github.io/memray/).

Disclaimer: no idea how accurate this is, given that neither code path is straightforward Python code.

I'm also not entirely sure how to read the Summary report that memray produces. It looks to me like the 'Total Memory' column is cumulative, with the overall total in the first row

For `python -m memray run experiment/benchmark.py --coreml` I get results like:
```
┃                                                     ┃    <Total ┃     Total ┃
┃ Location                                            ┃   Memory> ┃  Memory % ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇
┃ _run_code at                                        │ 182.612MB │    95.51% │
```
Possibly this means the real total memory is 191MB?

I get basically the same numbers if I run memray with the `--native` flag.

For `python -m memray run experiment/benchmark.py --pytorch` I get results like:
```
┃                                                     ┃    <Total ┃     Total ┃
┃ Location                                            ┃   Memory> ┃  Memory % ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇
│ _run_code at                                        │   1.201GB │    96.80% │
```

Again, similar number with the `--native` flag.

So, if this is measuring anything meaningful, it seems like the CoreML optimised model uses significantly (~6.6x) less memory.

## Further questions

1. We compared here the same model, `apple/ane-distilbert-base-uncased-finetuned-sst-2-english`, run via either PyTorch or CoreML.

    But, as I understand it, that model is a rewrite of the original DistilBERT PyTorch model, changing some details ([as described here](https://machinelearning.apple.com/research/neural-engine-transformers)) to ensure that CoreML will be able to run it on the ANE.

    It's possible those changes harm the PyTorch performance?  How does say [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) compare, running under PyTorch?

2. Huggingface are working on their own wrapper for coremltools https://github.com/huggingface/exporters

   Can I take the original HF `distilbert-base-uncased-finetuned-sst-2-english` and export it with that tool in a way that runs on the ANE?

### 1. Testing the original non-Apple PyTorch model

How well does the original `distilbert-base-uncased-finetuned-sst-2-english` model run compared to the Apple-modified version under PyTorch?

I updated the benchmark script to test this. Here's the results from a single run of each:

```
$ python -m experiment.benchmark --pytorch
Inferred 1821 inputs in 71321.61ms

$ python -m experiment.benchmark --pytorch-model-name distilbert-base-uncased-finetuned-sst-2-english
Inferred 1821 inputs in 64386.90ms
```

So the speed is comparable to the Apple-modified version, possibly a little faster, but still approx 8-10x slower than the CoreML version running on the ANE.

### 2. Testing exported models

Using https://github.com/huggingface/exporters I exported the HF `distilbert-base-uncased-finetuned-sst-2-english` model a few times with different options.

e.g.
```
python -m exporters.coreml --model=distilbert-base-uncased-finetuned-sst-2-english \
                           --feature=sequence-classification models/defaults.mlpackage
```

- `--compute_units=all` or `--compute_units=cpu_and_ne`
- `--quantize=float16`

Curiously the exported CoreML models loaded much faster than Apple's one:
```
Loaded CoreML model in 340.59ms
```
(vs ~4s for Apple's model)

However none of them ran on the ANE. So just specifying the target compute unit and quantizing to `float16` are not enough, we need the other refactorings that Apple have made in order for that to work. (Nice to see that the quantized version 'just worked' though).

And all of them ran approx 40% slower than the Apple one did under PyTorch, ~60ms per inference (vs ~44ms for Apple model under PyTorch).

I haven't included the exported models in this repo but if you export your own then you can try them in the benchmark script. I got results like:
```
python -m experiment.benchmark --coreml-model-path models/Model.mlpackage
Inferred 1821 inputs in 101774.21ms
```

One nice thing is the HF exporter generates a model which returns a nicer data structure from `model.predict`:

```python
{'var_482': array([[0.53114289, 0.46885711]]), 'probabilities': {'NEGATIVE': 0.5311428904533386, 'POSITIVE': 0.4688571095466614}, 'classLabel': 'NEGATIVE'}
```

i.e. you don't have to do the softmax + argmax to get probability and predicted class from the raw logits, it's already done for you.

(I think this may be what the `--feature=sequence-classification` option does).

## Faster PyTorch?

https://towardsdatascience.com/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1

> PyTorch v1.12 introduces GPU-accelerated training on Apple Silicon

`ane_transformers` specifies `"torch>=1.10.0,<=1.11.0"`, perhaps specifically to avoid this? (or that was just the latest version when authored)

Seems worth exploring though.

https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/

PyTorch claims 6-7x faster training and ~14x faster inference for Huggingface BERT using the Metal Performance Shaders backend on an M1 Ultra, "using batch size=64" (batch inference? training?)

The previous article only achieved ~2x speedup on a regular M1 though.

I added this option to both `server_pytorch` and `benchmark` scripts.

I had to upgrade to PyTorch 2.0 to get it run with no warnings.

The results were interesting - no speedup for the `apple/ane-distilbert-base-uncased-finetuned-sst-2-english` model, but a significant speedup for the vanilla `distilbert-base-uncased-finetuned-sst-2-english` model.

```
$ python -m experiment.benchmark --pytorch --pytorch-use-mps
Inferred 1821 inputs in 78389.13ms

$ python -m experiment.benchmark --pytorch-model-name distilbert-base-uncased-finetuned-sst-2-english --pytorch-use-mps
Inferred 1821 inputs in 22854.41ms
```

This is down to ~12.5ms per inference, so about 3.5x faster than CPU PyTorch and only about 3x slower than the CoreML ANE-accelerated version.

## What is this model anyway?

`apple/ane-distilbert-base-uncased-finetuned-sst-2-english` is clearly derived from `distilbert-base-uncased-finetuned-sst-2-english`.

But... what if I want to run a different fine-tuned DistilBERT variant on the ANE?

We can see on their repo the following explanation of how the model was generated:

> ...we initialize the baseline model as follows:
```python
import transformers
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
baseline_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    model_name,
    return_dict=False,
    torchscript=True,
).eval()
```

> Then we initialize the mathematically equivalent but optimized model, and we restore its parameters using that of the baseline model:
```python
from ane_transformers.huggingface import distilbert as ane_distilbert
optimized_model = ane_distilbert.DistilBertForSequenceClassification(
    baseline_model.config).eval()
optimized_model.load_state_dict(baseline_model.state_dict())
```

So it's clear that we are using the weights from `distilbert-base-uncased-finetuned-sst-2-english` and plugging them into the refactored ANE model. Then there's a couple more steps to export a CoreML after that.

In other words, it seems like it should be possible to export an ANE-accelerated version of any of the `bert-base` models on HuggingFace: https://huggingface.co/models?search=bert-base
