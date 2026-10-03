# 💳 Credit Card Fraud Detection — Machine Learning & Deep Learning

Capstone project: detecting fraudulent credit card transactions with classical machine-learning and deep-learning models, deployed as an interactive **Streamlit** web app.

🔗 **Live app:** `https://<your-app-name>.streamlit.app` *(replace with your Streamlit link)*
📓 **Notebook:** [`fraud_app/Credit_Card_Fraud_Detection.ipynb`](fraud_app/Credit_Card_Fraud_Detection.ipynb)

---

## 1. Problem Statement

Credit card fraud causes large financial losses every year. The goal is to build a binary classifier that flags a transaction as **Fraud (1)** or **Genuine (0)**.

The main challenge is **extreme class imbalance**: only about 0.17% of transactions are fraudulent. A model that predicts "genuine" for everything would be 99.8% accurate and completely useless, so the project focuses on **Precision–Recall based evaluation** instead of accuracy.

## 2. Dataset

| Item | Detail |
|---|---|
| Source | [Credit Card Cheating Detection (CCCD) – Kaggle](https://www.kaggle.com/datasets/arslanali4343/credit-card-cheating-detection-cccd) |
| Transactions (after removing duplicates) | 283,726 |
| Fraud rate | 0.167% |
| Features | 30: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed, anonymised) |
| Target | `Class` (1 = fraud, 0 = genuine) |

The dataset is downloaded directly in the notebook:

```python
import kagglehub
path = kagglehub.dataset_download("arslanali4343/credit-card-cheating-detection-cccd")
```

## 3. Methodology

```
Data ─► EDA ─► Cleaning & Split ─► Scaling ─► Imbalance handling ─► ML / DL models ─► Threshold tuning ─► Evaluation ─► Streamlit app
```

1. **Exploratory Data Analysis:** class distribution, transaction amount and hour-of-day by class, feature–target correlation, class-conditional feature densities.
2. **Preprocessing:** duplicate removal, median imputation, stratified **70 / 15 / 15** train / validation / test split, `StandardScaler` fitted on the training set only.
3. **Imbalance handling:**
   - **SMOTE** oversampling (training set only, to avoid data leakage) for Logistic Regression and Random Forest
   - **Cost-sensitive learning** (`scale_pos_weight`) for XGBoost
   - **Class weights** for the Deep MLP
   - **Unsupervised anomaly detection** for the Autoencoder, which is trained on genuine transactions only
4. **Threshold tuning:** each model's decision threshold is chosen to maximise F1 on the **validation** set, then applied unchanged to the **test** set.

## 4. Models

| Type | Model | Key settings |
|---|---|---|
| ML | Logistic Regression | Baseline, L2 regularisation (C = 0.1), SMOTE |
| ML | Random Forest | 150 trees, SMOTE |
| ML | XGBoost | 600 rounds max, early stopping on validation PR-AUC, `scale_pos_weight` |
| DL | Deep MLP | 64 → 32 → 16 → 1, ReLU, BatchNorm, Dropout 0.3, class weights, early stopping |
| DL | Autoencoder | 30 → 20 → 10 → **6** → 10 → 20 → 30; reconstruction error (MSE) used as the fraud score |

## 5. Results (held-out test set: 42,559 transactions, 71 frauds)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **XGBoost** 🏆 | **0.818** | **0.971** | 0.851 | **0.803** | **0.826** | 0.826 |
| Random Forest | 0.814 | 0.963 | **0.946** | 0.732 | 0.825 | **0.832** |
| Logistic Regression | 0.713 | 0.962 | 0.786 | 0.775 | 0.780 | 0.780 |
| Deep MLP | 0.684 | 0.971 | 0.800 | 0.789 | 0.794 | 0.794 |
| Autoencoder | 0.490 | 0.962 | 0.291 | 0.775 | 0.423 | 0.474 |

**Confusion matrices (test set)**

| Model | Frauds caught (TP) | Frauds missed (FN) | False alarms (FP) |
|---|---|---|---|
| XGBoost | 57 / 71 | 14 | 10 |
| Random Forest | 52 / 71 | 19 | 3 |
| Deep MLP | 56 / 71 | 15 | 14 |
| Logistic Regression | 55 / 71 | 16 | 15 |
| Autoencoder | 55 / 71 | 16 | 134 |

### Key findings
- **XGBoost is the best overall model.** It has the highest PR-AUC and F1, and catches 80% of frauds with only 10 false alarms out of more than 42,000 genuine transactions.
- **Random Forest is the most precise.** Only 3 false alarms, which suits a bank that wants to disturb customers as little as possible, though it misses more frauds.
- **ROC-AUC is misleading under heavy imbalance.** Every model scores about 0.96–0.97, while PR-AUC ranges from 0.49 to 0.82 and clearly separates them.
- **The Autoencoder needs no fraud labels.** It still reaches 77% recall, but at the cost of many false alarms. It is useful when labelled fraud data is scarce or new fraud patterns appear.
- **Most important features (XGBoost gain):** V14, V10, V4, V12, V19.

## 6. Streamlit Web App

The app (`fraud_app/app.py`) has three tabs:

1. **🔍 Single transaction:** pick a random test transaction (fraud or genuine) or enter feature values manually. Shows the prediction, fraud probability gauge, and the verdict of all five models side by side.
2. **📂 Batch prediction:** upload a CSV of transactions, get predictions and metrics (if labels are included), and download the results.
3. **📊 Model performance:** metrics table, ROC and Precision–Recall curves, confusion matrix, feature importance.

The sidebar lets you switch models and adjust the **decision threshold** to explore the precision–recall trade-off live.

## 7. Repository Structure

```
Credit-card-fraud-detection-model/
├── README.md
└── fraud_app/
    ├── Credit_Card_Fraud_Detection.ipynb   # full training notebook (Colab)
    ├── app.py                              # Streamlit application
    ├── requirements.txt
    └── artifacts/
        ├── scaler.joblib                   # fitted StandardScaler
        ├── logreg.joblib                   # Logistic Regression
        ├── rf.joblib                       # Random Forest
        ├── xgb.joblib                      # XGBoost
        ├── mlp.keras                       # Deep MLP
        ├── autoencoder.keras               # Autoencoder
        ├── meta.json                       # features, thresholds, metrics, curves
        └── test_sample.csv                 # test transactions for the demo
```

## 8. How to Run

**Train the models (Google Colab)**
1. Open `fraud_app/Credit_Card_Fraud_Detection.ipynb` in Google Colab.
2. Click **Runtime → Run all**. The dataset downloads automatically and the models are trained and saved to `artifacts/`.
3. The notebook also launches the app with a temporary public link.

**Run the app locally**
```bash
git clone https://github.com/jaypee83/Credit-card-fraud-detection-model.git
cd Credit-card-fraud-detection-model/fraud_app
pip install -r requirements.txt
streamlit run app.py
```

**Deploy on Streamlit Community Cloud:** repository `jaypee83/Credit-card-fraud-detection-model`, branch `main`, main file `fraud_app/app.py`, Python 3.11.

## 9. Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · TensorFlow / Keras · Matplotlib · Seaborn · Plotly · Streamlit · Google Colab

## 10. Future Work

- Hyperparameter optimisation (Optuna) and a stacked ensemble of XGBoost and Random Forest
- Cost-based threshold selection using the actual cost of a missed fraud versus a false alarm
- Model explainability with SHAP for individual predictions
- Sequence models (LSTM / Transformer) on per-card transaction histories
- Monitoring for concept drift as fraud patterns change

## 11. Author

**Jaypee** — Capstone Project, 2026
