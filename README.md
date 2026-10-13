# Customer Churn Prediction

![CI](https://github.com/mhamad1213/churn-prediction/actions/workflows/ci.yml/badge.svg)

Predicting which telecom customers are about to leave, explaining **why**, and deciding **who to contact**, served as a web app.

## Problem
A subscription business loses revenue every time a customer cancels. If we can rank customers by churn risk, a retention team can contact the riskiest ones first. Success is measured by **churners caught and profit**, not raw accuracy: on this data, predicting "nobody churns" is already 73.5% accurate.

## Data
[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d): 7,043 customers, 19 features (contract, services, billing, tenure), 26.5% churn. Cleaned and validated with a Pandera schema (`src/churn/data.py`).

## Results (held-out test set, 1,409 customers)

| Model | ROC-AUC | PR-AUC | Recall (churners) | Precision |
|---|---|---|---|---|
| Logistic regression baseline | 0.842 | 0.634 | 0.559 | 0.657 |
| **Tuned gradient boosting, class-balanced** | **0.846** | **0.667** | **0.799** | 0.507 |

- **Model comparison (5-fold CV):** logistic regression, Random Forest, gradient boosting and a PyTorch neural network all land at ROC-AUC 0.84–0.85. Tuned gradient boosting edges ahead on PR-AUC; the neural network trails (0.647), as is typical for tabular data.
- **Biggest gain:** handling class imbalance lifted recall from 56% to 80%.
- **Decision threshold** chosen by expected profit (offer £20, customer worth £300, 30% offer success): 0.43. Simulated campaign profit **£15,060** vs £12,450 for the baseline model (+21%) and £5,480 for contacting everyone.

## The app
`POST /predict` takes a customer's details and returns the churn probability, whether to contact them, and the top 3 reasons (SHAP). Input is validated twice (Pydantic, then Pandera). A web form is served at `/`, interactive API docs at `/docs`.

## Project structure
```
├── .github/workflows/ci.yml   # lint, retrain from scratch, test on every push
├── data/raw/                  # downloaded CSV (git-ignored)
├── models/                    # trained model + threshold (model git-ignored, rebuilt by training)
├── notebooks/                 # EDA and feature exploration
├── reports/                   # metrics JSON + figures
├── scripts/download_data.py
├── src/churn/
│   ├── data.py                # load, clean, Pandera schemas
│   ├── features.py            # feature engineering + preprocessing
│   ├── evaluate.py            # metrics incl. top-20% lift
│   ├── train_baseline.py      # logistic regression baseline
│   ├── train_models.py        # RF / gradient boosting comparison
│   ├── tune.py                # class imbalance + hyperparameter tuning
│   ├── threshold.py           # profit-based decision threshold
│   ├── neural_net.py          # PyTorch MLP as a scikit-learn estimator
│   ├── explain.py             # SHAP reasons per prediction
│   ├── api.py                 # FastAPI service
│   └── static/index.html      # web form
└── tests/                     # 17 pytest tests
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
export PYTHONPATH=src
python -m churn.train_baseline && python -m churn.tune && python -m churn.threshold
python -m pytest
uvicorn churn.api:app --reload --app-dir src     # then open http://127.0.0.1:8000
```

## Roadmap
- [x] Data validation, EDA, baseline
- [x] Feature engineering, ensembles, tuning, class imbalance
- [x] Profit-based threshold
- [x] PyTorch neural network comparison
- [x] FastAPI service with SHAP explanations
- [x] GitHub Actions CI
- [ ] Docker + deploy to Render