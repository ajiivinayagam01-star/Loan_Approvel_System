import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD ORIGINAL DATASET
# =========================================================

df = pd.read_csv("loans.csv")

print("=" * 65)
print("LOAN APPROVAL - GENERALIZED MACHINE LEARNING TRAINING")
print("=" * 65)

print("\nDataset shape:", df.shape)


# =========================================================
# 2. REMOVE ONLY ID
# =========================================================

df = df.drop(columns=["Applicant_ID"])


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]


print("\nFeatures:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================================================
# 5. LOGISTIC REGRESSION PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# =========================================================
# 6. HYPERPARAMETER TUNING
# =========================================================

param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10
    ],

    "model__solver": [
        "liblinear",
        "lbfgs"
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)


print("\n" + "-" * 65)
print("HYPERPARAMETER TUNING")
print("-" * 65)


grid_search.fit(
    X_train,
    y_train
)


best_model = grid_search.best_estimator_


print("\nBest Parameters:")
print(grid_search.best_params_)

print(
    f"\nBest Cross-Validation F1: "
    f"{grid_search.best_score_:.4f}"
)


# =========================================================
# 7. TEST SET EVALUATION
# =========================================================

y_pred = best_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


print("\n" + "=" * 65)
print("FINAL TEST PERFORMANCE")
print("=" * 65)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# =========================================================
# 8. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ]
    )
)


# =========================================================
# 9. CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 10. SAVE MODEL
# =========================================================

joblib.dump(
    best_model,
    "model.pkl"
)


print("\n" + "=" * 65)
print("MODEL SAVED")
print("=" * 65)

print("File: model.pkl")
print("Model: Logistic Regression Pipeline")
print("Scaling: StandardScaler")
print("Cross-validation: 5-fold")
print("Class balancing: Enabled")