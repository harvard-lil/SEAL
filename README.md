# IVAN: Empirical Interpretability for Value Alignment in RLHF

> TODO: Link to paper

## Summary
- [Getting started](#getting-started)
- [CLI: `compare`](#cli-compare)
- [CLI: `rewrite`](#cli-rewrite)
- [CLI: `generate-batch`](#cli-taxonomy-generate-batch)
- [CLI: `process-batch`](#cli-taxonomy-process-batch)
- [Cite](#cite)

---

## Getting started

**Machine-level dependencies:**
- [Python 3.11+](https://python.org)
- [Python Poetry](https://python-poetry.org/)

```bash
# Clone project
git clone https://github.com/harvard-lil/IVAN.git

# Install project and its dependencies
poetry install

# Copy and edit environment variables
cp .env.example .env
nano .env # (or any text editor)
```

[👆 Back to the summary](#summary)

---

## CLI: `compare`

This command allows for evaluating a series of reward models against Anthropic/hh-rlhf, and assess whether they align with human preferences.

Currently set up to evaluate [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) against:
- [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2)
- [OpenAssistant/reward-model-deberta-v3-large](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large)
- [OpenAssistant/reward-model-deberta-v3-base](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base)
- [OpenAssistant/reward-model-electra-large-discriminator](https://huggingface.co/OpenAssistant/reward-model-electra-large-discriminator)

```bash
poetry run python ivan.py compare

# Only evaluate the first 100 entries:
poetry run python ivan.py compare --limit=100
```

Results are saved as CSV under `data/compare`.

[👆 Back to the summary](#summary)

---

## CLI: `rewrite`

This command slightly rewrites and a set number of rows from the Anthropic/hh-rlhf using a text generation model, and analyses it using: 
- Two reward models, to measure how much the rewriting affected rewards:
  - [OpenAssistant/reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2), which was fine-tuned using Anthropic/hh-rlhf (`ALIGNED`)
  - [OpenAssistant/reward-model-deberta-v3-large](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large), which was not (`CONTROL`)
- A text-similarity model ([BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)), to control how much the rewriting affected the "meaning" of the prompts.
- OpenAI's GPT-4-Turbo run part of the `taxonomy` experiment on both original and rewritten prompts, as a way to control if GPT-4 Turbo's assessment was also affected by this rewritting.

By default, this command uses the [Ollama API](https://ollama.com/) to run inference against [mistral:7b-instruct-fp16](https://ollama.com/library/mistral:7b-instruct-fp16). An Ollama server must be ready, and the targeted model must be available.

```bash
# Runs the rewrite experiment
poetry run python ivan.py rewrite --limit=100

# Runs the rewrite experiment using a specific text generation model
poetry run python ivan.py rewrite --limit=100 --ollama-model-id="llama2:13b-instruct-fp16"
```

Results are saved as CSV under `data/rewrite`.

[👆 Back to the summary](#summary)

---

## CLI: `taxonomy create-batch`

**Alignment Taxonomy part 1:**

In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
This command generates a ["batch" file](https://platform.openai.com/docs/api-reference/batch) that can be used by the OpenAI API to process a large amount of requests efficiently.
The command will also perform a cost estimate and ask for confirmation before uploading files for processing.

```bash
# Run the taxonomy experiment against the whole dataset
poetry run python ivan.py taxonomy create-batch

# Run the experiment against a subset
poetry run python ivan.py taxonomy create-batch --limit=100
```

The command will return a series of batch ids that can be used to pull results from the experiment with `taxonomy process-batch`.

Intermediary data is saved under `data/taxonomy-input`.

[👆 Back to the summary](#summary)

---

## CLI: `taxonomy process-batch`

**Alignment Taxonomy part 2:**

In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
This command allows for pulling a series of completed batches from the OpenAI API and:
- Pull the raw results of the various batches into a single JSONL file
- Validate and save processed results in a single CSV. This format allows for easier processing and for associating analysis results with their original prompts.

This command allows for patching invalid entries on the fly using the OpenAI API (optional).

```bash
poetry run python ivan.py taxonomy process-batch --batch-ids="batch_foobar1,batch_foobar2,batch_foobar3"
```

Results are saved as under `data/taxonomy-output`.

[👆 Back to the summary](#summary)

---

## CLI: `taxonomy run-control-batch`

**Alignment Taxonomy part 3:**

In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
This command picks random entries from the dataset and runs the same experiment against an open source model, as a way to generate control data.

By default, this command uses the [Ollama API](https://ollama.com/) to run inference against [gemma:7b-instruct-fp16](https://ollama.com/library/gemma:7b-instruct-fp16). An Ollama server must be ready, and the targeted model must be available.

```bash
poetry run python ivan.py taxonomy run-control-batch

# Only run on 5 rows of the Anthropic/hh-rlhf dataset
poetry run python ivan.py taxonomy run-control-batch --limit=5

# Run command using a specific model
poetry run python ivan.py taxonomy run-control-batch --model=llama3:8b-instruct-fp16

# Run command against a specific set of row_ids
poetry run python ivan.py taxonomy run-control-batch --set-row-ids="98434, 156621, 149074, 103182, 82050"
```

Results are saved as under `data/taxonomy-control`.

[👆 Back to the summary](#summary)

---

## Cite

> TODO

[👆 Back to the summary](#summary)
