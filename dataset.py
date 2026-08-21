# dataset.py

import numpy as np
import pandas as pd

# For reproducible results
np.random.seed(42)

# Number of applicants
n = 1000

# Generate applicant data
data = {
    "Applicant_ID": np.arange(1, n + 1),

    "Age": np.random.randint(21, 61, n),

    "Income": np.random.randint(20000, 150001, n),

    "Credit_Score": np.random.randint(300, 851, n),

    "Employment_Years": np.random.randint(0, 21, n),

    "Loan_Amount": np.random.randint(50000, 1000001, n),

    "Existing_Loans": np.random.randint(0, 6, n),

    "Loan_Term": np.random.choice([12, 24, 36, 48, 60], n)
}

df = pd.DataFrame(data)


# ---------------------------------------------------------
# Create an approval score based on multiple factors
# ---------------------------------------------------------

score = (
    (df["Credit_Score"] - 300) / 550 * 40
    + (df["Income"] - 20000) / 130000 * 25
    + df["Employment_Years"] / 20 * 15
    + (1 - df["Existing_Loans"] / 5) * 10
    + (1 - df["Loan_Amount"] / 1000000) * 10
)

# Add some randomness/noise
noise = np.random.normal(0, 8, n)

final_score = score + noise

# Convert score into binary target
df["Loan_Approved"] = (final_score >= 55).astype(int)


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

df.to_csv("loans.csv", index=False)


# ---------------------------------------------------------
# Display information
# ---------------------------------------------------------

print("Loan dataset generated successfully!")
print("\nDataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nApproval Distribution:")
print(df["Loan_Approved"].value_counts())

print("\nDataset Information:")
print(df.info())