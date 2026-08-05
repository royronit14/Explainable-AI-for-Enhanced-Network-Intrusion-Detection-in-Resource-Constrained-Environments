# Explainable AI for Enhanced Network Intrusion Detection in Resource-Constrained Environments

This repository contains a notebook-based machine learning workflow for network intrusion detection using the UNSW-NB15 dataset. The main focus is to compare models, train an XGBoost classifier, and explain the resulting predictions with SHAP-based feature importance.

## What is included

- `My_Project.ipynb`: the main notebook that loads the dataset, preprocesses the features, trains and compares models, and generates SHAP explanations.
- `Explanaible_IDS.ipynb`: a second notebook related to the intrusion detection workflow.
- `UNSW_NB15_training.csv`: the dataset used by the notebook.
- `UNSW-NB15_features.csv`: feature metadata for the dataset.
- `UNSW-NB15_LIST_EVENTS.csv`: event mapping / label reference file.

## Workflow

The notebook workflow shown in the workspace follows these steps:

1. Load the UNSW-NB15 training data.
2. Split the data into training and test sets.
3. Preprocess the features with scikit-learn transformers.
4. Train and compare candidate models.
5. Select XGBoost as the best-performing model.
6. Use SHAP to explain the model and identify the most important features.
7. Compare a full model against a lightweight feature subset.

## Requirements

Install the Python packages listed in `requirements.txt` before running the notebooks.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter or open the notebook in VS Code.
4. Run `My_Project.ipynb` from top to bottom.

## Expected libraries

The notebook imports and uses:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap
- lime
- eli5

## Notes

- The notebooks were not executed in this workspace, so the README is based on the visible notebook code and file layout.
- The project appears notebook-driven rather than packaged as a Python module.
