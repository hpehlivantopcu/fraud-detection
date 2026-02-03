# 🚨 Fraud Detection & Anomaly Analysis

## What This Project Does
Flags fraudulent transactions using supervised classification. Key challenges solved: **class imbalance** (only ~4 % of transactions are fraud) handled with SMOTE, and a **PCA-projected decision-boundary scatter plot** shows how the model separates fraud from legitimate transactions.

---

## Files

| File | What It Is |
|---|---|
| `fraud_detection.py` | Full pipeline: EDA → SMOTE → 3 models → ROC + scatter viz |
| `README.md` | This file |

### Output Images (auto-generated)
- `eda_fraud_features.png`
- `roc_fraud.png`
- `decision_boundary_fraud.png`
- `confusion_matrices_fraud.png`

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
python fraud_detection.py
```

## Swap In Real Data
Find this line:
```python
df = generate_transaction_data()
```
Replace with:
```python
df = pd.read_csv("transactions.csv")
```
Make sure your CSV has a binary label column and update `"IsFraud"` references accordingly.

---

## Add to Your GitHub Portfolio
1. Create repo `fraud-detection` on GitHub
2. Push everything in this folder
3. Update the link in your `index.html` project card
