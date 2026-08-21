import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("loans.csv")

print("=" * 50)
print("LOAN APPROVAL DATASET - EDA")
print("=" * 50)


# --------------------------------------------------
# 2. Basic information
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. First 5 rows
# --------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. Duplicate values
# --------------------------------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# --------------------------------------------------
# 6. Data types
# --------------------------------------------------

print("\nData Types:")
print(df.dtypes)


# --------------------------------------------------
# 7. Statistical summary
# --------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------------------------
# 8. Loan approval distribution
# --------------------------------------------------

print("\nLoan Approval Distribution:")
print(df["Loan_Approved"].value_counts())


# --------------------------------------------------
# 9. Approval percentage
# --------------------------------------------------

approval_percentage = df["Loan_Approved"].value_counts(normalize=True) * 100

print("\nLoan Approval Percentage:")
print(approval_percentage)


# --------------------------------------------------
# 10. Correlation matrix
# --------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Loan approval count
# --------------------------------------------------

plt.figure(figsize=(6, 5))

sns.countplot(
    data=df,
    x="Loan_Approved"
)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Approved (0 = No, 1 = Yes)")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 12. Credit Score vs Loan Approval
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Loan_Approved",
    y="Credit_Score"
)

plt.title("Credit Score vs Loan Approval")
plt.xlabel("Loan Approved")
plt.ylabel("Credit Score")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Income vs Loan Approval
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Loan_Approved",
    y="Income"
)

plt.title("Income vs Loan Approval")
plt.xlabel("Loan Approved")
plt.ylabel("Income")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14. Loan Amount vs Loan Approval
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Loan_Approved",
    y="Loan_Amount"
)

plt.title("Loan Amount vs Loan Approval")
plt.xlabel("Loan Approved")
plt.ylabel("Loan Amount")

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 15. Employment Years vs Loan Approval
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Loan_Approved",
    y="Employment_Years"
)

plt.title("Employment Years vs Loan Approval")
plt.xlabel("Loan Approved")
plt.ylabel("Employment Years")

plt.tight_layout()
plt.show()


print("\nEDA completed successfully!")