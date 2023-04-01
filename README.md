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
$ python experiment/server.py
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

I was unable to observe any flicker of power consumption from ANE when inferring individual phrases using the current `experiment/server.py`.

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

### Update

To avoid having to manually enter prompts in quick succession via the cli, I made `experiments/benchmark.py`.

This loads the SST2 dataset (which the model was originally trained on) and runs inference on the 1821 examples in the "test" segment of the dataset.

Unfortunately the `coremltools` Python library [does not support batch inference](https://github.com/apple/coremltools/issues/196) (which is available in CoreML via ObjC) ... I would imagine batch inference is possible in PyTorch too. So I have just done a series of individual inferences in a tight loop, after pre-prepping the tokens etc.

We can see some expected results:

```
$ python experiment/benchmark.py --coreml
Inferred 1821 inputs in 7980.14ms

$ python experiment/benchmark.py --pytorch
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
