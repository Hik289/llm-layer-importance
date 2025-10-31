# Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning

This repository contains the implementation and experimental code for our research paper "Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning".

**Authors:** Xinyuan Song, Keyu Wang, PengXiang Li, Lu Yin, Shiwei Liu

**Paper:** [arXiv:2510.02091](https://arxiv.org/abs/2510.02091)

## Abstract

Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.

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

- `layer_remove.py`: Main script for layer removal experiments
- `peft_pretraining/`: Core analysis modules
  - `layer_pruning*.py`: Various layer pruning implementations
  - `head_pruning.py`: Attention head analysis
  - `angular_distance.py`: Layer relationship analysis
  - `utils/`: Utility functions for metrics and analysis

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry:

```bibtex
@misc{song2025demystifyingrolesllmlayers,
      title={Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning}, 
      author={Xinyuan Song and Keyu Wang and PengXiang Li and Lu Yin and Shiwei Liu},
      year={2025},
      eprint={2510.02091},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.02091}, 
}
```

## Acknowledgement

This repository builds upon various open-source projects. We thank the authors of [ShortGPT](https://github.com/sramshetty/ShortGPT/tree/hf-models) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their foundational work.
