import pandas as pd
import joblib


# ==========================================
# 1. Load trained model
# ==========================================

model = joblib.load("model.pkl")

print("Loan Approval Prediction System")
print("=" * 40)


# ==========================================
# 2. Get applicant details
# ==========================================

age = int(input("Enter Age: "))
income = float(input("Enter Annual Income: "))
credit_score = int(input("Enter Credit Score: "))
employment_years = float(input("Enter Employment Years: "))
loan_amount = float(input("Enter Loan Amount: "))
existing_loans = int(input("Enter Number of Existing Loans: "))
loan_term = int(input("Enter Loan Term (months): "))


# ==========================================
# 3. Create input DataFrame
# ==========================================

new_applicant = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Credit_Score": [credit_score],
    "Employment_Years": [employment_years],
    "Loan_Amount": [loan_amount],
    "Existing_Loans": [existing_loans],
    "Loan_Term": [loan_term]
})


# ==========================================
# 4. Make prediction
# ==========================================

prediction = model.predict(new_applicant)[0]


# ==========================================
# 5. Display result
# ==========================================

print("\n" + "=" * 40)

if prediction == 1:
    print("LOAN APPROVAL: APPROVED")
else:
    print("LOAN APPROVAL: REJECTED")


# ==========================================
# 6. Display probability
# ==========================================

if hasattr(model, "predict_proba"):

    probability = model.predict_proba(new_applicant)[0]

    approval_probability = probability[1] * 100

    print(f"Approval Probability: {approval_probability:.2f}%")

print("=" * 40)