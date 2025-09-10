#!/usr/bin/env python
# coding: utf-8

import os
from dotenv import load_dotenv
load_dotenv()

from distilabel.models.llms import OllamaLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
from distilabel.steps.tasks import TextGeneration


## Dataset

loader = LoadDataFromHub(
    repo_id="distilabel-internal-testing/instruction-dataset-mini",
    split="test",
    batch_size=2
)
loader.load()

# LLM
llm = OllamaLLM(model="gemma3:4b")  # Must match ollama model name
llm.load()


# Distilabel Pipeline

with Pipeline(  # 
    name="simple-text-generation-pipeline",
    description="A simple text generation pipeline",
) as pipeline:  # 
    load_dataset = LoadDataFromHub(  # 
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(  # 
        name="text_generation",
        llm=llm,  # 
    )

    load_dataset >> text_generation # type: ignore


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            load_dataset.name: {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            }
        },
    )

