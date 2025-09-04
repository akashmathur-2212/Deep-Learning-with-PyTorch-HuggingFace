# LLM Finetuning Toolkit

A comprehensive guide and codebase for fine-tuning Large Language Models (LLMs) using popular frameworks such as:

| Framework | Description | Key Features |
|----------|-------------|--------------|
| [🤗 Hugging Face](https://huggingface.co/docs/peft/index) | Most popular LLM training/inference ecosystem | PEFT(LoRA / QLoRA), Datasets, Accelerate, Trainer |
| 🦥 [Unsloth](https://github.com/unslothai/unsloth) | Super-optimized training library for LLaMA models | Fastest LoRA training, no CUDA issues |
| 🦎 [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | YAML-based finetuning config on top of HuggingFace | Dataset streaming, low RAM usage |

---

## Table of Contents

- [Overview](#overview)
- [Frameworks Covered](#frameworks-covered)
- [Fine-tuning Methods](#fine-tuning-methods)
- [Exercise Notebooks](#exercise-notebooks)
- [Hyperparameters and Recommendations](#hyperparameters-and-recommendations)

---

## Overview

Large Language Models (LLMs) can be adapted to specific tasks and domains via **fine-tuning**. This repository explores various frameworks, methods, and techniques to fine-tune models.

We cover both **full finetuning** and **parameter-efficient tuning** (like LoRA, QLoRA), across multiple frameworks.

### Parameter-Efficient Fine-Tuning (PEFT)

As language models grow larger, traditional fine-tuning becomes increasingly challenging. A full fine-tuning of even a 1.7B parameter model requires substantial GPU memory, makes storing separate model copies expensive, and risks catastrophic forgetting of the model's original capabilities. Parameter-efficient fine-tuning (PEFT) methods address these challenges by modifying only a small subset of model parameters while keeping most of the model frozen.

Traditional fine-tuning updates all model parameters during training, which becomes impractical for large models. PEFT methods introduce approaches to adapt models using fewer trainable parameters - often less than 1% of the original model size. This dramatic reduction in trainable parameters enables:

- Fine-tuning on consumer hardware with limited GPU memory
- Storing multiple task-specific adaptations efficiently
- Better generalization in low-data scenarios
- Faster training and iteration cycles

---


## Fine-tuning Methods

| Method         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Full Finetuning | Update **all model weights** – highest accuracy but memory intensive      |
| LoRA           | Inject adapters into layers to tune a few parameters                       |
| QLoRA          | Use quantized models with LoRA (4-bit/8-bit) for very low memory training  |
| DPO / PPO      | Reinforcement-based fine-tuning for alignment (RLHF)                       |


## Exercise Notebooks

| Title	| Description	| Exercise	| Notebook	| Colab
|----------|-------------|--------------|--------------|--------------|
| LoRA Fine-tuning | Learn how to fine-tune models using LoRA adapters	| 🐢 Train a model using LoRA <br> 🐕 Experiment with different rank values<br> 🦁 Compare performance with full fine-tuning	| - [🤗PEFT Notebook](https://github.com/akashmathur-2212/PyTorch/blob/main/llm-finetuning/%F0%9F%A4%97%20PEFT/mistral_finetuning_QLoRA.ipynb)<br> - [Unsloth Notebook](https://github.com/akashmathur-2212/PyTorch/blob/main/llm-finetuning/unsloth/Qwen3_14B-Finetuning.ipynb)<br> - [Axolotl Notebook](https://github.com/akashmathur-2212/PyTorch/blob/main/llm-finetuning/axolotl/finetuning_Llama_3.2_1B_axolotl.ipynb)<br>	| - Unsloth [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WF2IM5EEwW4L1hwM4cG6OzU9pNyOlMbj?usp=sharing) <br> - Axolotl [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N-Nx5HKmNLxJwGtW6M1-I4bnT1LhIkRA?usp=sharing)|


## Hyperparameters and Recommendations

| Hyperparameter | Function | Recommended Settings|
|----------|-------------|--------------|
| LoRA Rank (r) | Controls the number of trainable parameters in the LoRA adapter matrices. A higher rank increases model capacity but also memory usage.| 8, 16, 32, 64, 128 | Choose 16 or 32 | 
| LoRA Alpha (lora_alpha)| Scales the strength of the fine-tuned adjustments in relation to the rank (r).| r (standard) or r * 2 (common heuristic). More details here.|
| LoRA Dropout| A regularization technique that randomly sets a fraction of LoRA activations to zero during training to prevent overfitting. Not that useful, so we default set it to 0. | 0 (default) to 0.1 | A regularization term that penalizes large weights to prevent overfitting and improve generalization. Don't use too large numbers! | 0.01 (recommended) - 0.1 | 
| Warmup Steps | Gradually increases the learning rate at the start of training. | 5-10% of total steps | 
| Scheduler Type | Adjusts the learning rate dynamically during training. | linear or cosine |
| Seed (random_state) | A fixed number to ensure reproducibility of results. | Any integer (e.g., 42, 3407) | 
| Target Modules | Specify which parts of the model you want to apply LoRA adapters to — either the attention, the MLP, or both. | **Attention**: q_proj, k_proj, v_proj, o_proj<br> **MLP**: gate_proj, up_proj, down_proj<br> **Recommended** to target all major linear layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.|
---
