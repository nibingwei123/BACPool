# BACPool

**BACPool: A Cross-Attention Dual-Branch Pooling Module for Molecular Property Prediction**

This repository provides the implementation of BACPool, a cross-attention dual-branch pooling module for molecular property prediction. It includes hyperparameter optimization, multi-seed training, model performance evaluation, and explainability analysis.

## Repository Structure

```text
BACPool/
│
├── dataset/                              # Datasets used in this study
│
├── trained_model/                        # Trained model checkpoints
│
├── environment.yml                       # Conda environment configuration
│
├── best_param_gcn.csv                    # Best hyperparameters for GCN
├── best_param_gatv2.csv                  # Best hyperparameters for GATv2
├── best_param_gin.csv                    # Best hyperparameters for GIN
│
├── model_gnns_grid (5).py                # Hyperparameter optimization
├── model_gnns_seed (7).py                # Training with 10 random seeds
├── model_gnns_evaluate_layernorm (1).py  # Model performance evaluation
├── explain_model_layernorm (1).py        # Explainability analysis
│
└── README.md
```

## Code Description

The repository contains four main Python scripts:

| Script | Description |
|---|---|
| [`model_gnns_grid (5).py`](https://github.com/nibingwei123/BACPool/blob/main/model_gnns_grid%20%285%29.py) | **Hyperparameter optimization.** Performs hyperparameter searches to identify the optimal configurations for different GNN backbones. |
| [`model_gnns_seed (7).py`](https://github.com/nibingwei123/BACPool/blob/main/model_gnns_seed%20%287%29.py) | **Multi-seed training.** Trains the models using 10 different random seeds to evaluate the robustness and stability of the results. |
| [`model_gnns_evaluate_layernorm (1).py`](https://github.com/nibingwei123/BACPool/blob/main/model_gnns_evaluate_layernorm%20%281%29.py) | **Model evaluation.** Evaluates the predictive performance of the trained models. |
| [`explain_model_layernorm (1).py`](https://github.com/nibingwei123/BACPool/blob/main/explain_model_layernorm%20%281%29.py) | **Explainability analysis.** Performs model interpretability analysis to investigate the contributions of molecular structures to model predictions. |

## Datasets

All datasets used in this study are provided in the [`dataset/`](https://github.com/nibingwei123/BACPool/tree/main/dataset) directory.

Users can directly access the datasets from this folder.

## Pre-trained Models

The trained model checkpoints are provided in the [`trained_model/`](https://github.com/nibingwei123/BACPool/tree/main/trained_model) directory.

These models can be used directly for model evaluation and explainability analysis without repeating the training process.

## Optimal Hyperparameters

The best hyperparameter configurations obtained through hyperparameter optimization are provided in the corresponding CSV files:

- [`best_param_gcn.csv`](https://github.com/nibingwei123/BACPool/blob/main/best_param_gcn.csv): Optimal hyperparameters for GCN.
- [`best_param_gatv2.csv`](https://github.com/nibingwei123/BACPool/blob/main/best_param_gatv2.csv): Optimal hyperparameters for GATv2.
- [`best_param_gin.csv`](https://github.com/nibingwei123/BACPool/blob/main/best_param_gin.csv): Optimal hyperparameters for GIN.

These files follow the naming convention `best_param_{backbone}.csv`, where `{backbone}` represents the corresponding GNN backbone. The optimal configurations can be used for subsequent multi-seed training experiments.

## Environment Setup

The Conda environment configuration is provided in [`environment.yml`](environment.yml), which specifies the Python packages and dependencies used in this project.

To create and activate the Conda environment, run:

```bash
conda env create -f environment.yml
conda activate BACPool
```

Please ensure that the environment is successfully configured before running the scripts.

## Experimental Workflow

The recommended workflow for reproducing the experiments is as follows:

**Step 1. Hyperparameter Optimization**

Run `model_gnns_grid (5).py` to perform hyperparameter searches and determine the optimal configurations for the evaluated GNN backbones. The best hyperparameters are saved in the corresponding `best_param_{backbone}.csv` files.

**Step 2. Multi-Seed Training**

Run `model_gnns_seed (7).py` to train the models using 10 different random seeds based on the selected optimal hyperparameters. The trained model checkpoints are saved in the `trained_model/` directory.

**Step 3. Model Performance Evaluation**

Run `model_gnns_evaluate_layernorm (1).py` to evaluate the predictive performance of the trained models.

**Step 4. Explainability Analysis**

Run `explain_model_layernorm (1).py` to conduct explainability analysis and investigate the molecular substructures and features contributing to model predictions.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{BACPool,
  title={BACPool: A Cross-Attention Dual-Branch Pooling Module for Molecular Property Prediction},
  author={},
  journal={},
  year={}
}
```

## Contact

For questions or further information about this repository, please feel free to contact the authors.
