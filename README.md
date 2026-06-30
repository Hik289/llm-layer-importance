<div align="center">

# Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

**Xinyuan Song, Keyu Wang, PengXiang Li, Lu Yin, Shiwei Liu**

[![ICASSP 2026](https://img.shields.io/badge/ICASSP%202026-Accepted-success.svg)](https://2026.ieeeicassp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.02091-b31b1b.svg)](https://arxiv.org/abs/2510.02091)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2510.02091)
[![Code](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/Hik289/llm-layer-importance)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Official implementation for "Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning."**

**Accepted by ICASSP 2026.**

</div>

---

## Overview

This repository contains the implementation and experimental code for our
ICASSP 2026 paper, [Demystifying the Roles of LLM Layers in Retrieval,
Knowledge, and Reasoning](https://arxiv.org/abs/2510.02091).

## Abstract

Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.

## Repository Contents

| Component | Location | Purpose |
|-----------|----------|---------|
| Layer removal | `layer_remove.py` | Remove individual transformer layers and save modified checkpoints. |
| Layer pruning | `peft_pretraining/layer_pruning*.py` | Run layer pruning experiments across model families. |
| Head pruning | `peft_pretraining/head_pruning.py` | Analyze attention head importance. |
| Angular distance | `peft_pretraining/utils/angular_distance.py` | Measure layer similarity using ShortGPT-style angular distance. |
| Metrics | `peft_pretraining/utils/metrics.py` | Utility metrics for pruning and analysis. |

---

## Quick Start

### Setup

Configure the environment using the following command lines:
```bash
conda create -n layer-importance python=3.9 -y
conda activate layer-importance
pip install -r requirements.txt
```

### Layer Removal and Performance Analysis

The main script for analyzing layer importance is `layer_remove.py`. This script allows you to remove specific layers from LLMs and evaluate the performance impact.

```bash
# Remove a specific layer from LLaMA-7B
python layer_remove.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --layer_index 1 \
    --save_path ./llama_7b_removed_1
```

### Layer Pruning Experiments

For comprehensive layer pruning experiments, use the following scripts:

```bash
# Enhanced layer pruning
python peft_pretraining/layer_pruning_enhanced.py

# Head pruning analysis  
python peft_pretraining/head_pruning.py

# DeepSeek specific layer pruning
python peft_pretraining/layer_pruning_deepseek.py
```


### Angular Distance Analysis

Calculate the angular distance between different layers to understand layer relationships. Based on modifications from [ShortGPT](https://github.com/sramshetty/ShortGPT/tree/hf-models).

```bash
cd peft_pretraining/utils
python angular_distance.py --model_path <model_path> --n_samples <n_samples>
# Example:
# python angular_distance.py --model_path meta-llama/Llama-2-7b-hf --n_samples 1000
```

### Evaluation and Performance Drop

Calculate the performance drop after removing different layers. We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

```bash
# Install lm_eval
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Key Findings

Our research reveals that:

1. **Layer Importance Varies by Task**: Knowledge and retrieval tasks rely primarily on shallow layers, while reasoning tasks require deeper layers.

2. **Evaluation Metric Matters**: Likelihood-based metrics suggest deeper layers are less important, but generation-based evaluation reveals their critical role.

3. **Context-Dependent Depth Usage**: The importance of layers varies significantly based on the specific evaluation setting and task requirements.

## Repository Structure

```text
llm-layer-importance/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- exp_requirements.txt
|-- layer_remove.py
|-- torchrun_main.py
`-- peft_pretraining/
    |-- layer_pruning.py
    |-- layer_pruning_enhanced.py
    |-- layer_pruning_deepseek.py
    |-- head_pruning.py
    |-- cumulative_replace.py
    |-- layer_change.py
    |-- qwen3_angular_analysis.py
    `-- utils/
        |-- angular_distance.py
        |-- metrics.py
        `-- short_hf.py
```

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry:

```bibtex
@inproceedings{song2026demystifyingrolesllmlayers,
  title     = {Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning},
  author    = {Xinyuan Song and Keyu Wang and PengXiang Li and Lu Yin and Shiwei Liu},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  note      = {Accepted. arXiv:2510.02091},
  url       = {https://arxiv.org/abs/2510.02091}
}
```

## Acknowledgement

This repository builds upon various open-source projects. We thank the authors of [ShortGPT](https://github.com/sramshetty/ShortGPT/tree/hf-models) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their foundational work.
