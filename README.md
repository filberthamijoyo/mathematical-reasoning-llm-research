# Mathematical Reasoning in Language Models

This repository contains advanced research and implementation of fine-tuning techniques for enhancing mathematical reasoning capabilities in large language models (LLMs).

## Overview

This research explores methods to improve the mathematical reasoning abilities of language models through specialized fine-tuning techniques. The work focuses on developing parameter-efficient approaches that enhance step-by-step reasoning for solving mathematical problems, particularly using smaller models that can run with limited computational resources.

## Research Methodology

The research employs a systematic approach to improving mathematical reasoning:

1. **Dataset Curation & Combination**: Multiple mathematical datasets are strategically combined to create a diverse training signal
2. **Parameter-Efficient Adaptation**: LoRA (Low-Rank Adaptation) is optimized with increased rank and alpha values
3. **Memory-Efficient Training**: Implementation of 4-bit quantization, gradient checkpointing, and CPU offloading
4. **Enhanced Prompting**: Structured prompts that guide the model through step-by-step reasoning processes
5. **Robust Evaluation**: Specialized metrics for mathematical accuracy and reasoning coherence

## Architecture Improvements

### LoRA Configuration

The research implements significant improvements to LoRA configuration:

- **Increased Rank (r=32)**: Expanded from traditional values of 8-16 to 32 for greater representational capacity
- **Higher Alpha (α=64)**: Increased from standard 16 to 64 for stronger weight updates
- **Enhanced Dropout (0.1)**: Added to improve generalization performance
- **Target Modules Expansion**: Attention components (q_proj, k_proj, v_proj, o_proj) and feed-forward components (gate_proj, up_proj, down_proj) are all targeted for adaptation

### Training Optimizations

- **Epoch-Based Training**: Train for 2 complete epochs rather than a fixed number of steps
- **Gradient Accumulation**: Set to 8 steps for creating effectively larger batch sizes
- **Learning Rate**: Optimized at 1e-4 with cosine scheduler for stability
- **Weight Decay**: Increased to 0.05 for enhanced regularization
- **Warmup Ratio**: Extended to 0.05 for more stable training initiation
- **Memory Optimizations**: Implementation of paged_adamw_32bit optimizer, gradient checkpointing, and 4-bit quantization

## Implementation Details

### TinyLlama Fine-Tuning (`tinyllama_fine_tuning.ipynb`)

This notebook demonstrates:
- Fine-tuning the TinyLlama-1.1B-Chat-v1.0 model using optimized LoRA
- Dataset preparation with GSM8K and NuminaMath datasets
- Implementation of chain-of-thought prompting techniques
- Evaluation on mathematical problem-solving benchmarks
- Quantitative analysis of reasoning steps and accuracy

Key technical features:
- Memory-efficient 4-bit quantization with `bnb_4bit_quant_type="nf4"`
- Enhanced tokenization with proper handling of padding tokens
- Template-based prompt formatting with explicit reasoning instructions
- Robust answer extraction with multiple fallback strategies

### Larger Model Implementation (`large_model_math_reasoning.ipynb`) 

This notebook includes:
- Memory-optimized techniques for fine-tuning larger language models (7B+ parameters)
- Advanced dataset combinations for improved training signal
- Comparative analysis with TinyLlama implementations
- Custom evaluation metrics for mathematical reasoning
- Ablation studies on different training approaches

Technical highlights:
- CPU and disk offloading strategies for large model training
- Robust error handling with multiple fallback paths for model loading
- Dynamic batch size adjustment based on available GPU memory
- Checkpoint selection based on quantitative loss metrics

## Datasets Used

The research utilizes multiple datasets with specific contributions:

- **PrimeIntellect/NuminaMath-QwQ-CoT-5M**: A large-scale dataset of mathematical problems with step-by-step solutions, providing 5,000 diverse examples
- **GSM8K**: Grade-school math problems with detailed solutions, contributing 3,000 training examples and serving as the primary evaluation benchmark
- **reasoning-machines/gsm-hard**: Challenging problems focused on complex reasoning, supplementing training with 1,000 difficult examples in the TinyLlama implementation

### Data Processing Pipeline

The implementation features a robust data loading pipeline with:
- Automatic fallback to alternative datasets if primary sources fail
- Seed-controlled shuffling for reproducibility
- Strategic sampling to balance different problem types
- Combined dataset creation with configurable size limits
- Data structure analysis for verification of format consistency

## Evaluation Methodology

The evaluation methodology includes:

- **GSM8K Test Set**: Primary benchmark using 100-200 examples from the GSM8K test set
- **Answer Extraction**: Multi-strategy approach for reliable answer identification:
  - Primary: Pattern matching for answers after #### markers
  - Secondary: Extraction of answers from "The answer is..." patterns
  - Tertiary: Identification of answers following "Therefore..." statements
  - Final fallback: Extraction of the last numerical value in the response
- **Numeric Comparison**: Robust comparison of extracted answers accounting for formatting differences
- **Statistical Analysis**: Calculation of overall accuracy, relative improvement percentages, and error patterns
- **Visualization**: Automated generation of comparison charts and example showcases

## Experimental Results

The research demonstrates significant improvements in mathematical reasoning ability:

1. **Parameter-efficient fine-tuning** can significantly improve mathematical reasoning even in smaller models, with TinyLlama showing improvements of 15-30% in absolute accuracy
2. **Chain-of-thought prompting combined with LoRA** provides the best results for mathematical problem-solving
3. **TinyLlama models** can achieve competitive performance on mathematical reasoning tasks with proper fine-tuning, making them viable for resource-constrained environments
4. **Larger models benefit from combined datasets** and memory optimization techniques, with Mistral-7B showing improvements of 10-20% in absolute accuracy

### Error Analysis

The implementation includes detailed error analysis capabilities:
- Identification of problem types where improvement is most significant
- Examples of corrected solutions after fine-tuning
- Visualization of performance differences between base and fine-tuned models
- Sample demonstration of reasoning improvements

## Setup Requirements

The notebooks are self-contained with installation instructions for required packages:
- transformers (latest GitHub version)
- peft (latest GitHub version for LoRA support)
- accelerate (for distributed training capabilities)
- bitsandbytes (for quantization)
- datasets (for data loading and processing)
- pytorch (with CUDA support)

## Usage Examples

The repository includes complete workflows for:
1. Loading and preparing mathematical datasets
2. Fine-tuning models with optimized LoRA configurations
3. Evaluating models on benchmark datasets
4. Analyzing and visualizing results
5. Testing models on custom mathematical problems

## Citation

If you use this research in your work, please cite our paper.
