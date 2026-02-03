"""
=============================================================
Fraud Detection & Anomaly Analysis
=============================================================
Goal : Detect fraudulent transactions using supervised ML.
       Handles class imbalance with SMOTE, compares three
       classifiers, and visualizes the decision boundary.

Sections
--------
1. Imports & Synthetic Data
2. Exploratory Data Analysis
3. Class-Imbalance Handling (SMOTE)
4. Train / Test Split & Scaling
5. Model Training (Logistic Regression, Random Forest, SVM)
6. Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
7. Decision-Boundary Scatter Plot (PCA-projected)
8. Confusion Matrices

Dependencies
------------
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
=============================================================
"""

# ─── 1. IMPORTS & DATA ───────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.decomposition    import PCA
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
np.random.seed(42)


def generate_transaction_data(n_legit: int = 4800,
                              n_fraud: int = 200) -> pd.DataFrame:
    """
    Synthetic transaction dataset.
    Legitimate transactions cluster around (mean_amt=50, mean_freq=3).
    Fraudulent transactions cluster around (mean_amt=200, mean_freq=8).
    Replace with pd.read_csv(...) for real data.
    """
    # Legitimate transactions
    leg_amt   = np.random.normal(50,  20, n_legit).clip(5, 200)
    leg_freq  = np.random.normal(3,   1.5, n_legit).clip(0, 12)
    leg_loc   = np.random.normal(30,  10, n_legit)
    leg_hr    = np.random.normal(12,  5, n_legit).clip(0, 24)
    leg_label = np.zeros(n_legit, dtype=int)

    # Fraudulent transactions
    fr_amt   = np.random.normal(200, 60, n_fraud).clip(50, 500)
    fr_freq  = np.random.normal(8,   2,  n_fraud).clip(1, 20)
    fr_loc   = np.random.normal(70,  15, n_fraud)
    fr_hr    = np.random.normal(3,   2,  n_fraud).clip(0, 8)   # mostly late night
    fr_label = np.ones(n_fraud, dtype=int)

    df = pd.DataFrame({
        "TransactionAmount":   np.concatenate([leg_amt,  fr_amt]),
        "TransactionFrequency":np.concatenate([leg_freq, fr_freq]),
        "LocationScore":       np.concatenate([leg_loc,  fr_loc]),
        "HourOfDay":           np.concatenate([leg_hr,   fr_hr]),
        "IsFraud":             np.concatenate([leg_label, fr_label]),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


df = generate_transaction_data()
print("Dataset shape:", df.shape)
print(df["IsFraud"].value_counts())


# ─── 2. EDA ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Fraud vs Legitimate – Feature Distributions", fontsize=14)

features = ["TransactionAmount", "TransactionFrequency",
            "LocationScore",     "HourOfDay"]
colors   = {0: "#00e5a0", 1: "#ffa94d"}

for ax, feat in zip(axes.flat, features):
    for label, name in {0: "Legitimate", 1: "Fraud"}.items():
        ax.hist(df.loc[df["IsFraud"] == label, feat], bins=40,
                alpha=0.55, label=name, color=colors[label])
    ax.set_title(feat)
    ax.legend()
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_fraud_features.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 3. SMOTE – BALANCE CLASSES ──────────────────────────────
X = df.drop(columns=["IsFraud"])
y = df["IsFraud"]

smote   = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print(f"\nAfter SMOTE  →  0: {(y_bal==0).sum()}   1: {(y_bal==1).sum()}")


# ─── 4. SPLIT & SCALE ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─── 5. MODEL TRAINING ───────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    results[name] = {"clf": clf, "y_pred": y_pred, "y_prob": y_prob, "auc": auc}
    print(f"\n{'='*50}\n{name}  –  AUC: {auc:.4f}\n{'='*50}")
    print(classification_report(y_test, y_pred))


# ─── 6. ROC CURVES ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
palette = {"Logistic Regression": "#00e5a0",
           "Random Forest":       "#ffa94d",
           "SVM (RBF)":           "#b388ff"}
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=palette[name],
            label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Fraud Detection")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_fraud.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 7. DECISION-BOUNDARY SCATTER (PCA 2-D) ─────────────────
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test_sc)

fig, ax = plt.subplots(figsize=(9, 6))
for label, color, marker in [(0, "#00e5a0", "o"), (1, "#ffa94d", "^")]:
    mask = y_test.values == label
    ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
               c=color, marker=marker, s=40, alpha=0.6,
               label="Legitimate" if label == 0 else "Fraud", edgecolors="k", linewidths=0.4)

# Approximate decision boundary via grid
h  = 0.4
xx = np.arange(X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1, h)
yy = np.arange(X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1, h)
grid_2d = np.c_[xx.ravel(), yy.ravel()]
# Inverse-transform grid back to original feature space for best model
best_name = max(results, key=lambda k: results[k]["auc"])
grid_orig = pca.inverse_transform(grid_2d)
Z = results[best_name]["clf"].predict(grid_orig)
Z = Z.reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors=["#00bcd4"], linewidths=2, linestyles="--")

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title(f"Decision Boundary – {best_name} (PCA projection)")
ax.legend()
plt.tight_layout()
plt.savefig("decision_boundary_fraud.png", dpi=150, bbox_inches="tight")
plt.show()


# ─── 8. CONFUSION MATRICES ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (name, res) in zip(axes, results.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, res["y_pred"], ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig("confusion_matrices_fraud.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n─── DONE ───")
