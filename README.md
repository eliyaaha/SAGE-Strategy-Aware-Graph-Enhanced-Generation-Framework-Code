# SAGE: A Strategy-Aware Graph-Enhanced Generation Framework for Online Counseling

> This repository contains the implementation, experiments, and evaluations accompanying the paper "SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling", accepted at the **ACM UMAP 2026** conference.

## Overview
This work introduces **SAGE**, a **Strategy-Aware Graph-Enhanced** framework developed as a decision-support tool to assist mental health experts during online counseling sessions. By integrating **Heterogeneous Graph Transformers (HGT)** for strategy prediction with a dynamic graph-injection mechanism into LLMs, the framework generates recommended intervention messages that are both contextually and strategically aligned with clinical practices.

## Repository Structure
### Pipeline Overview:
1. `src/:` Houses the foundational logic of the pipeline.
   * `preprocessing.py`: Handles initial data cleaning and structural preparation.
   * `graph_utils.py`: Manages the construction of heterogeneous graph representations.
   * `prompt_utils.py`: Formats contextual data for the language model.
   * `gen_metrics.py` & `metrics.py`: Defines evaluation protocols for both **Next Strategy Classifier** and **Recommended Response Generator**.

2. `models/:` Defines the underlying architectures.
    * `hgt_model.py`: The Heterogeneous Graph Transformer for strategy prediction.
    * `sage_generator.py`: The hybrid architecture integrating graph embeddings into the LLM.

3. `scripts/:` The primary execution scripts for the SAGE framework.
    * `train_gnn.py`: Orchestrates the graph-based strategy training.
    * `train_llm_ga.py`: Manages the fine-tuning of the SAGE generator.
    * `run_generation.py`: Executes the full inference and evaluation pipeline.

### Artifacts & Storage:
* `config/:` Centralizes all variables.
* `models_checkpoints/` & `outputs/`: These directories are automatically created during runtime to store model artifacts, fine-tuning checkpoints, and final evaluation CSVs.
* `experiments/`: Contains standalone scripts for baseline comparisons and ablation testing.

## Getting Started
* Installation: `pip install -r requirements.txt`
* Hardware: NVIDIA RTX 6000 Ada.
* Pipeline Execution: Execute the following scripts in order to reproduce the SAGE framework results:
    ```
    python scripts/train_gnn.py
    python scripts/train_llm_ga.py
    python scripts/run_generation.py
    ```
    

## Dataset
Due to the sensitive nature of mental health counseling dialogues, the full dataset is not publicly available to protect participant privacy and comply with ethical standards. Access to anonymized subsets may be granted for non-commercial academic research purposes. To request access, please visit the [*Help-Seeking Corpus*](https://resources.nnlp-il.mafat.ai/?search=help-seeking-corpus) page and follow the application instructions provided there.

## Citation
If you find our work or code useful for your research, please cite our paper:
```bibtex
