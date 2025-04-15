# Mathematical Reasoning in Language Models

This repository contains research and implementation of fine-tuning language models for enhanced mathematical reasoning capabilities.

## Overview

This research explores methods to improve the mathematical reasoning abilities of language models through specialized fine-tuning techniques. The work focuses on developing parameter-efficient approaches that enhance step-by-step reasoning for solving mathematical problems, particularly using smaller models that can run with limited computational resources.

## Key Research Areas

- **Parameter-Efficient Fine-Tuning**: Implementation of LoRA (Low-Rank Adaptation) to train models efficiently without modifying all weights
- **Step-by-Step Reasoning Enhancement**: Training models to provide detailed chain-of-thought explanations when solving problems
- **TinyLlama Optimization**: Fine-tuning of the TinyLlama-1.1B-Chat model for mathematics
- **Larger Model Comparisons**: Investigating techniques for larger models and benchmarking against TinyLlama variants

## Implementation Details

### TinyLlama Fine-Tuning (`tinyllama_fine_tuning.ipynb`)

This notebook demonstrates:
- Fine-tuning the TinyLlama-1.1B-Chat-v1.0 model using LoRA
- Dataset preparation with GSM8K and NuminaMath datasets
- Implementation of chain-of-thought prompting techniques
- Evaluation on mathematical problem-solving benchmarks
- Quantitative analysis of reasoning steps and accuracy

### Larger Model Implementation (`large_model_math_reasoning.ipynb`) 

This notebook includes:
- Memory-optimized techniques for fine-tuning larger language models
- Advanced dataset combinations for improved training signal
- Comparative analysis with TinyLlama implementations
- Custom evaluation metrics for mathematical reasoning
- Ablation studies on different training approaches

## Datasets Used

The research utilizes multiple datasets:
- **PrimeIntellect/NuminaMath-QwQ-CoT-5M**: A large dataset of mathematical problems with step-by-step solutions
- **GSM8K**: Grade-school math problems with detailed solutions
- **reasoning-machines/gsm-hard**: Challenging problems focused on complex reasoning

## Evaluation Methodology

The evaluation methodology includes:
- Accuracy measurements on the GSM8K test set
- Analysis of reasoning coherence and correctness
- Comparison with baseline models
- Step-by-step solution tracing

## Key Findings

The research demonstrates:
1. Parameter-efficient fine-tuning can significantly improve mathematical reasoning even in smaller models
2. Chain-of-thought prompting combined with LoRA provides the best results for mathematical problem-solving
3. TinyLlama models can achieve competitive performance on mathematical reasoning tasks with proper fine-tuning
4. Larger models benefit from combined datasets and memory optimization techniques

## Setup Requirements

The notebooks are self-contained with installation instructions for required packages:
- transformers
- peft
- accelerate
- bitsandbytes
- datasets

## Citation

If you use this research in your work, please cite our paper. 