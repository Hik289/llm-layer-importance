# Artifact Guide

Operational notes for reproducing `Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning` from the public `llm-layer-importance` repository.

## Review Path

- `peft_pretraining/`: Project-specific implementation subtree.
- Root-level entry points: `layer_remove.py`, `torchrun_main.py`.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `exp_requirements.txt`: Experiment-specific Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
```

If no smoke command is tracked, use the README Quick Start with the smallest seed, sample, or task count.

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `python layer_remove.py`
- `python torchrun_main.py`

## Data And Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
