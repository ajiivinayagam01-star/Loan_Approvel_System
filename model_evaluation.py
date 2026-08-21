import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)


# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("loans.csv")

# Remove ID
df = df.drop(columns=["Applicant_ID"])

# Features and target
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]


# ==========================================
# 2. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. Load saved best model
# ==========================================

model = joblib.load("model.pkl")

print("Best model loaded successfully!")


# ==========================================
# 4. Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 5. Classification Report
# ==========================================

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Rejected", "Approved"]
    )
)


# ==========================================
# 6. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Loan Approval Confusion Matrix")

plt.tight_layout()
plt.show()


# ==========================================
# 7. ROC Curve
# ==========================================

if hasattr(model, "predict_proba"):

    y_probability = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nROC-AUC Score: {roc_auc:.4f}")

else:
    print("\nROC curve skipped because the selected model does not provide probability estimates.")


print("\nModel evaluation completed successfully!")