# AlignEval-LLM

**SFT + DPO post-training pipeline for behavioral robustness**

---

## Overview
AlignEval-LLM is a practical post-training pipeline that combines Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to shape specific model behaviors — particularly reducing unwanted traits such as excessive persuasion and sycophancy — while preserving general capabilities.

## Key Capabilities
- Full SFT → DPO pipeline using modern open-source tooling
- Dataset construction for preference learning (synthetic + curated)
- Behavioral targeting: persuasion resistance, sycophancy reduction, calibrated confidence
- Alignment-style fine-tuning that changes specific behaviors without catastrophic forgetting

## Pipeline Stages
1. **Supervised Fine-Tuning (SFT)**  
   Instruction-following and capability preservation on high-quality data

2. **Preference Data Construction**  
   Generation of chosen / rejected pairs targeting specific failure modes

3. **Direct Preference Optimization (DPO)**  
   Preference optimization with KL regularization to stay close to the SFT model

4. **Evaluation**  
   Targeted behavioral probes + general capability benchmarks

## Design Principles
- Prefer precise behavioral change over broad capability alteration
- Maintain a strong reference model (SFT checkpoint) throughout DPO
- Use PEFT/LoRA for parameter-efficient updates when compute is limited
- Explicitly measure both the target behavior and side effects on other capabilities

## Tech Stack
- Hugging Face TRL
- DeepSpeed
- PEFT / LoRA
- Transformers
- Custom preference data generation and evaluation scripts

## Research Focus
- How to reduce sycophancy and over-persuasion without harming helpfulness
- Trade-offs between alignment strength and capability retention
- Efficient post-training under limited compute budgets

## Status
Active pipeline (updated August 2026). Used for experiments on controllable behavioral alignment of open-weight language models. Ongoing refinements focus on preference data quality and evaluation reliability.

---

**Author**: Gopar Beji  
**Related work**: RobustVision (vision adversarial robustness), EdgeGen (generative compression)
