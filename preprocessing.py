import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("loans.csv")

# -----------------------------------------
# 2. Remove Applicant_ID
# -----------------------------------------

df = df.drop(columns=["Applicant_ID"])

# -----------------------------------------
# 3. Separate features and target
# -----------------------------------------

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

# -----------------------------------------
# 4. Split data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# 5. Feature scaling
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------
# 6. Display results
# -----------------------------------------

print("Preprocessing completed successfully!")

print("\nFeatures:")
print(X.columns.tolist())

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nScaled training data:")
print(X_train_scaled[:5])