# Credit Card Fraud Detection & Risk Analytics System

A Streamlit machine-learning web application for predicting whether a credit-card transaction is likely genuine or fraudulent.

## Included
- `app.py` — Streamlit website
- `train_model.py` — retrain the XGBoost model
- `model/` — trained model, scaler, metrics and feature importance
- `data/demo_transactions.csv` — demo transactions
- `requirements.txt` — dependencies

## Run locally
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal.
4. Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

5. Chrome/browser will open the website.

## Retrain with the original dataset
Place your full `Credit_card.csv` in `data/` and run:

```bash
python train_model.py
streamlit run app.py
```

## Dataset note
The common Credit Card Fraud dataset has anonymized PCA features `V1`–`V28`, plus `Time`, `Amount`, and `Class`. `Class=0` is genuine and `Class=1` is fraud.
