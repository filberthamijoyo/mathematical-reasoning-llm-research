# Mathematical Reasoning in Language Models

This repository contains research and implementation of fine-tuning language models for enhanced mathematical reasoning capabilities.

## Overview

The project explores various techniques to improve mathematical reasoning abilities in language models, particularly focusing on:

- Fine-tuning TinyLLama models (1.1B parameters) on mathematical datasets
- Applying parameter-efficient fine-tuning techniques (LoRA)
- Evaluating performance on GSM8K benchmark
- Analyzing step-by-step reasoning capabilities

## Files in this Repository

- `large_model_math_reasoning.ipynb`: Implementation focusing on larger models with parameter-efficient fine-tuning
- `tinyllama_fine_tuning.ipynb`: Code for fine-tuning TinyLlama-1.1B-Chat for mathematical reasoning
- `mathematical_reasoning_research_paper.pdf`: Research findings and methodology

## Datasets Used

- PrimeIntellect/NuminaMath-QwQ-CoT-5M
- GSM8K
- reasoning-machines/gsm-hard

## Setup

The notebooks are self-contained with installation instructions for required packages:
- transformers
- peft
- accelerate
- bitsandbytes
- datasets
