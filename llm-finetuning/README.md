# 🔧 LLM Finetuning Toolkit

A comprehensive guide and codebase for fine-tuning Large Language Models (LLMs) using popular frameworks such as:

- 🤗 Hugging Face Transformers
- 🦥 [Unsloth](https://github.com/unslothai/unsloth)
- 🦎 [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- 🦜 LoRA / QLoRA (Parameter-Efficient Finetuning)
- 🔍 Comparison of techniques and performance

---

## Table of Contents

- [Overview](#overview)
- [Frameworks Covered](#frameworks-covered)
- [Fine-tuning Methods](#fine-tuning-methods)
- [Quick Start](#quick-start)
- [Comparison Table](#comparison-table)
- [Best Practices](#best-practices)
- [References](#references)

---

## Overview

Large Language Models (LLMs) can be adapted to specific tasks and domains via **fine-tuning**. This repository explores various frameworks, methods, and techniques to fine-tune models.

We cover both **full finetuning** and **parameter-efficient tuning** (like LoRA, QLoRA), across multiple frameworks.

---

## Frameworks Covered

| Framework | Description | Key Features |
|----------|-------------|--------------|
| 🤗 Hugging Face | Most popular LLM training/inference ecosystem | PEFT(LoRA / QLoRA), Datasets, Accelerate, Trainer |
| 🦥 Unsloth | Super-optimized training library for LLaMA models | Fastest LoRA training, no CUDA issues |
| 🦎 Axolotl | YAML-based finetuning config on top of HuggingFace | Dataset streaming, low RAM usage |

---

## Fine-tuning Methods

| Method         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| Full Finetuning | Update **all model weights** – highest accuracy but memory intensive      |
| LoRA           | Inject adapters into layers to tune a few parameters                       |
| QLoRA          | Use quantized models with LoRA (4-bit/8-bit) for very low memory training  |
| DPO / PPO      | Reinforcement-based fine-tuning for alignment (RLHF)                       |

---
