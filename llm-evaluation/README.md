# LLM Evaluation Toolkit

A comprehensive guide and codebase for evaluating Large Language Models (LLMs) based appliatons using popular methods such as:

| Framework | Description |
|----------|--------------|
| [Ragas](https://docs.ragas.io/en/stable/) | Ragas is an open-source evaluation framework for Retrieval-Augmented Generation (RAG) pipelines. |
| [LLM-as-a-Judge](https://arxiv.org/abs/2411.15594) | Automated Evaluation using SOTA models.|

---

## Table of Contents

- [Overview](#overview)
- [Frameworks Covered](#frameworks-covered)
- [Exercise Notebooks](#exercise-notebooks)

---

## Overview

Evaluation of Large language models (LLMs) is often a difficult endeavour: given their broad capabilities, the tasks given to them often should be judged on requirements that would be very broad, and loosely-defined. This repository explores various frameworks, methods, and techniques to evalaute them.

---

## Frameworks Covered

**1. [Ragas](https://docs.ragas.io/en/stable/)**

Ragas is an open-source evaluation framework for Retrieval-Augmented Generation (RAG) pipelines, providing metrics to assess both the retrieval and generation components of a RAG system without needing human-annotated "ground truth" data. By offering LLM-based, reference-free evaluation metrics, RAGAs helps developers measure aspects like faithfulness, answer relevance, and context relevance, enabling faster and more objective assessment cycles for RAG applications.

**2. LLM-as-a-Judge**

LLM as a judge refers to using large language models themselves to evaluate outputs from other models. Instead of relying solely on humans, LLM judges can assess quality. It is often essential in testing multiagent systems and AI applications. Instead of relying solely on human annotations or traditional metrics, developers use powerful LLMs to assess responses based on quality, accuracy, relevance, coherence, and more. This approach enables automated, scalable, and cost-effective evaluation and in many cases, LLM judges perform comparably to human reviewers.

---

## Exercise Notebooks

| Title	| Description	| Notebook	| Colab
|----------|-------------|--------------|--------------|
| Evaluation with [Ragas](https://docs.ragas.io/en/stable/) and Advanced Retrieval Methods Using LangChain | - Learn how to leverage RAGAS for evaluations. <br> - Explore multiple Retrieval Systems to improve the quality of our generations.<br> - Synthetic Data (Ground Truth / Reference Dataset) Creation using SOTA models.<br> - LLM-as-a-Judge Evaluation using SOTA models.| [Notebook](https://github.com/akashmathur-2212/Deep-Learning-with-PyTorch-HuggingFace/blob/main/llm-evaluation/evaluation_with_ragas_langchain.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZvHk4ckZtf4DtzJPC7zysrwEpqmCrQ6L/view?usp=sharing) |
| LLM-as-a-Judge | Learn how to automate evaluations using LLM-as-a-judge | [Notebook](https://github.com/akashmathur-2212/Deep-Learning-with-PyTorch-HuggingFace/blob/main/llm-evaluation/automated_evaluation_using_llm_as_judge.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZR7tunTgqVenzEx_Jc2WXfg6o2XCVqt2?usp=sharing) |


---
