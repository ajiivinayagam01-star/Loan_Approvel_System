import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("loans.csv")

print("=" * 60)
print("LOAN APPROVAL - MACHINE LEARNING TRAINING")
print("=" * 60)


# =========================================================
# 2. REMOVE ID
# =========================================================

df = df.drop(columns=["Applicant_ID"])


# =========================================================
# 3. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]


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
# 5. CREATE MODELS
# =========================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("model", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=8
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}


# =========================================================
# 6. TRAIN AND EVALUATE MODELS
# =========================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0


for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training: {name}")
    print("-" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    # Display results
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Select best model using F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name


# =========================================================
# 7. MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="F1_Score",
        ascending=False
    ).to_string(index=False)
)


# =========================================================
# 8. BEST MODEL
# =========================================================

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print(f"F1 Score: {best_f1:.4f}")


# =========================================================
# 9. DETAILED EVALUATION OF BEST MODEL
# =========================================================

best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Rejected", "Approved"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, best_predictions))


# =========================================================
# 10. SAVE BEST MODEL
# =========================================================

joblib.dump(best_model, "model.pkl")

print("\nBest model saved successfully as:")
print("model.pkl")