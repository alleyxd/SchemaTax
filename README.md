# SchemaTax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23754532.svg)](https://doi.org/10.5281/zenodo.23754532)

Does forcing a model to answer in JSON cost accuracy? This repo separates three things that "JSON mode" usually
changes at once:

- **enforcement**: is the schema enforced by constrained decoding?
- **visibility**: is the schema shown to the model in the prompt?
- **field order**: does the `reasoning` field come before or after the `answer`?

and measures each on GSM8K (reasoning), AG News (classification) and CoNLL-2003 (entity extraction) with Qwen3-1.7B,
4B and 8B served by vLLM. 300 examples per task, 12,600 requests in total, every response saved in `results/`.

Paper: [`paper/schematax.pdf`](paper/schematax.pdf) ([Zenodo](https://doi.org/10.5281/zenodo.23754532))

## Results

GSM8K accuracy:

| Condition | 1.7B | 4B | 8B |
|---|---|---|---|
| free text, "The answer is N" | 0.827 | 0.923 | 0.927 |
| JSON, schema shown, unconstrained | 0.370 | 0.910 | 0.820 |
| JSON, schema shown and enforced | 0.710 | 0.897 | 0.923 |
| JSON, schema enforced but not shown | 0.697 | 0.770 | 0.840 |
| enforced, `answer` before `reasoning` | 0.133 | 0.240 | 0.297 |
| enforced, `answer` only | 0.120 | 0.260 | 0.303 |

AG News accuracy and CoNLL entity F1:

| Condition | AG 1.7B | AG 4B | AG 8B | CoNLL 1.7B | CoNLL 4B | CoNLL 8B |
|---|---|---|---|---|---|---|
| free text | 0.507 | 0.777 | 0.823 | 0.321 | 0.476 | 0.380 |
| JSON, shown, unconstrained | 0.410 | 0.853 | 0.803 | 0.000 | 0.508 | 0.476 |
| JSON, shown and enforced | 0.820 | 0.850 | 0.803 | 0.430 | 0.511 | 0.474 |
| JSON, enforced, not shown | 0.400 | 0.823 | 0.783 | 0.387 | 0.620 | 0.574 |

What that adds up to:

- **Enforcement is close to free when the schema is shown.** On GSM8K the 4B and 8B models are within noise of free
  text (-2.7 and -0.3 points) and use half the tokens. The 1.7B model does pay about 12 points.
- **Field order is the real tax.** Answer-first schemas drop accuracy by 63 to 69 points, to the level of no reasoning,
  and 55 of the 56 answers that hit the length limit were the model repeating itself after committing to a wrong
  answer. (This replicates [Dylan Castillo's result on GPT-4o-mini](https://www.dsdev.in/order-of-fields-in-structured-output-can-hurt-llms-output) on open models.)
- **Hidden schemas cost reasoning**: 9 to 15 points on GSM8K.
- **Unconstrained JSON breaks small models.** At 1.7B half the answers don't parse: LaTeX escapes inside JSON, and the
  model copying the JSON Schema itself (`"type": "object", "properties": {...}`) with its answers inside.
- **Showing the schema can hurt extraction.** At 4B and 8B, the models fill the catch-all `misc` field with numbers
  and dates when they can see it; hiding the schema raises CoNLL F1 by 10 points.

## Reproduce

Start any OpenAI-compatible server with structured output, for example vLLM:

```bash
docker run --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-8B --served-model-name local --gpu-memory-utilization 0.65
```

Then:

```bash
pip install -e .
python -m schematax.run --label qwen3-8b --n 300      # writes results/qwen3-8b.jsonl, resumes if interrupted
python -m schematax.analyze                           # tables with paired bootstrap intervals
```

`run_models.sh` runs several models in a row, one vLLM container each. `--duty 0.5` (the default) pauses after every
batch for as long as the batch took, so the GPU is busy about half the time.

`schematax/tasks.py` has every prompt, schema and scorer. Each journal line holds the task, condition, example id,
score, token count, finish reason and the raw response.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests cover answer parsing and scoring for every task, so a change to a scorer can't silently shift the results.

## Limitations

One model family, one serving stack, greedy decoding with thinking turned off. CoNLL uses exact string match, so its
absolute F1 is low for every condition; the comparisons between conditions are what matter. Prompts were not tuned per
condition.

## Citation

```bibtex
@misc{ashter2026schematax,
  title  = {The Schema Tax Is a Schema Design Tax: When Constrained JSON Decoding Costs Accuracy, and When It Helps},
  author = {Ashter, Ali},
  year   = {2026},
  doi    = {10.5281/zenodo.23754532},
  url    = {https://doi.org/10.5281/zenodo.23754532}
}
```

## License

MIT
