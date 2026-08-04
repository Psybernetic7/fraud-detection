# Fraud Detection Predictor

A machine learning pipeline that flags potentially fraudulent financial transactions, paired with an interactive Streamlit app for real-time predictions.

## Overview

Financial fraud is rare but costly, so in this dataset, fraudulent transactions make up just **0.13%** of all records (8,213 out of 6.36M). That imbalance is the core challenge: a model that predicts "not fraud" every time would be 99.87% accurate and completely useless. This project focuses on building a model that actually catches fraud (high recall) rather than chasing raw accuracy.

The pipeline:
1. Explores and cleans ~6.36M transaction records
2. Engineers features from transaction type and account balances
3. Trains a class-weighted Logistic Regression model inside a scikit-learn `Pipeline`
4. Serves predictions through a Streamlit web app

## Results

| Metric | Score |
|---|---|
| Accuracy | 94.4% |
| Recall (fraud class) | 95% |
| Precision (fraud class) | 2% |

Recall is prioritized over precision by design: in fraud detection, missing an actual fraud case is far more costly than flagging a legitimate transaction for review. The low precision is an expected tradeoff of `class_weight="balanced"` on a 750:1 imbalanced dataset, and would typically be handled downstream with a manual review queue.

## Dataset

[AIML Dataset / PaySim-style synthetic transaction data](https://www.kaggle.com/datasets/ealaxi/paysim1) — ~6.36M mobile money transactions with fields for transaction type, amount, and sender/receiver balances before and after each transaction.

> The dataset (~470MB) is not included in this repo due to size. Download it and place it in the project root as `AIML Dataset.csv` to reproduce the analysis/training notebook.

## Tech Stack

- **Python**
- **scikit-learn** — preprocessing pipeline (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) and Logistic Regression model
- **Pandas** — data wrangling
- **Seaborn / Matplotlib** — exploratory data analysis
- **Streamlit** — web app interface
- **joblib** — model serialization

## Project Structure

```
.
├── analysis_model.ipynb          # EDA, feature engineering, model training
├── fraud_detection.py            # Streamlit app for live predictions
├── fraud_detection_pipeline.pkl  # Trained, serialized model pipeline
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/Psybernetic7/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run fraud_detection.py
```

This launches a local web app where you can enter transaction details (type, amount, sender/receiver balances) and get a real-time fraud prediction from the trained model.

### Retraining the model

Open `analysis_model.ipynb` to walk through the EDA and retrain the pipeline from scratch (requires the dataset — see [Dataset](#dataset)). The final cell re-saves the trained pipeline to `fraud_detection_pipeline.pkl`.

## How It Works

- **Input features:** transaction type, amount, and old/new balances for both sender and receiver
- **Preprocessing:** numeric features are standardized; transaction type is one-hot encoded
- **Model:** Logistic Regression with `class_weight="balanced"` to counteract the severe class imbalance
- **Output:** a binary prediction (fraud / not fraud) surfaced in the app with a clear success/error message


