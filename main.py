from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import numpy as np
import pandas as pd

import shap
from lime.lime_tabular import LimeTabularExplainer

import logging


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Loan Approval Predictor API",
    description="ML Loan Approval Prediction with SHAP and LIME XAI",
    version="2.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load("model.pkl")

    logger.info(
        "✓ Model loaded successfully"
    )

except Exception as e:

    logger.error(
        f"Model loading failed: {e}"
    )

    raise


# =========================================================
# FEATURE DEFINITIONS
# =========================================================

FEATURE_NAMES = [

    "Age",

    "Income",

    "Credit_Score",

    "Employment_Years",

    "Loan_Amount",

    "Existing_Loans",

    "Loan_Term"
]


# =========================================================
# REQUEST MODEL
# =========================================================

class LoanApplication(BaseModel):

    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Applicant age"
    )

    income: float = Field(
        ...,
        gt=0,
        description="Annual income"
    )

    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score"
    )

    employment_years: float = Field(
        ...,
        ge=0,
        description="Years of employment"
    )

    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan amount"
    )

    existing_loans: int = Field(
        ...,
        ge=0,
        description="Number of existing loans"
    )

    loan_term: int = Field(
        ...,
        gt=0,
        description="Loan term in months"
    )


# =========================================================
# RESPONSE MODEL
# =========================================================

class PredictionResponse(BaseModel):

    approval_status: str

    approval_probability: float

    confidence_percentage: float

    recommendation: str

    shap_explanation: list

    lime_explanation: list

    top_positive_factors: list

    top_negative_factors: list

    affordability: dict


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

# IMPORTANT:
# loans.csv is NOT modified.
# It is only used as representative data for LIME and SHAP.

training_df = pd.read_csv(
    "loans.csv"
)


training_X = training_df[
    FEATURE_NAMES
]


# =========================================================
# LIME EXPLAINER
# =========================================================

lime_explainer = LimeTabularExplainer(

    training_X.values,

    feature_names=FEATURE_NAMES,

    class_names=[
        "Rejected",
        "Approved"
    ],

    mode="classification",

    discretize_continuous=True,

    random_state=42
)


# =========================================================
# SHAP EXPLAINER
# =========================================================

scaler = model.named_steps["scaler"]

classifier = model.named_steps["model"]


background_scaled = scaler.transform(
    training_X
)


shap_explainer = shap.LinearExplainer(
    classifier,
    background_scaled
)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
async def root():

    return {

        "service":
            "Loan Approval Predictor API",

        "status":
            "running",

        "xai":
            [
                "SHAP",
                "LIME"
            ],

        "endpoints":
            {
                "health": "/health",
                "predict": "/predict",
                "docs": "/docs"
            }
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict_loan_approval(
    application: LoanApplication
):

    try:

        # =================================================
        # CREATE FEATURE ARRAY
        # =================================================

        features = np.array([[
            application.age,
            application.income,
            application.credit_score,
            application.employment_years,
            application.loan_amount,
            application.existing_loans,
            application.loan_term
        ]])


        # =================================================
        # ML PREDICTION
        # =================================================

        prediction = model.predict(
            features
        )[0]


        probabilities = model.predict_proba(
            features
        )[0]


        approval_probability = float(
            probabilities[1]
        )


        confidence_percentage = round(
            approval_probability * 100,
            2
        )


        # =================================================
        # SHAP EXPLANATION
        # =================================================

        scaled_features = scaler.transform(
            features
        )


        shap_values = shap_explainer(
            scaled_features
        )


        shap_vector = np.array(
            shap_values.values
        ).reshape(-1)


        shap_explanation = []


        for feature, value in zip(
            FEATURE_NAMES,
            shap_vector
        ):

            shap_explanation.append({

                "feature":
                    feature,

                "contribution":
                    round(
                        float(value),
                        6
                    ),

                "direction":
                    (
                        "positive"
                        if value > 0
                        else "negative"
                    )
            })


        # Sort by absolute contribution

        shap_explanation.sort(
            key=lambda x:
                abs(x["contribution"]),

            reverse=True
        )


        # =================================================
        # LIME EXPLANATION
        # =================================================

        lime_result = lime_explainer.explain_instance(

            features[0],

            model.predict_proba,

            num_features=len(
                FEATURE_NAMES
            )
        )


        lime_explanation = []


        for condition, contribution in (
            lime_result.as_list()
        ):

            lime_explanation.append({

                "feature":
                    condition,

                "contribution":
                    round(
                        float(contribution),
                        6
                    ),

                "direction":
                    (
                        "positive"
                        if contribution > 0
                        else "negative"
                    )
            })


        # =================================================
        # TOP SHAP FACTORS
        # =================================================

        positive_factors = [

            x

            for x in shap_explanation

            if x["contribution"] > 0
        ]


        negative_factors = [

            x

            for x in shap_explanation

            if x["contribution"] < 0
        ]


        positive_factors = sorted(

            positive_factors,

            key=lambda x:
                abs(x["contribution"]),

            reverse=True
        )


        negative_factors = sorted(

            negative_factors,

            key=lambda x:
                abs(x["contribution"]),

            reverse=True
        )


        # =================================================
        # AFFORDABILITY CALCULATION
        # =================================================

        # Annual income → monthly income

        monthly_income = (
            application.income / 12
        )


        # Simple estimated monthly repayment.
        #
        # NOTE:
        # This does not include interest.
        # It is used as a basic affordability indicator.

        estimated_payment = (

            application.loan_amount
            /
            application.loan_term
        )


        # Monthly payment relative to monthly income

        payment_to_income = (

            estimated_payment
            /
            monthly_income
        )


        # Loan amount relative to annual income

        loan_to_income = (

            application.loan_amount
            /
            application.income
        )


        # =================================================
        # AFFORDABILITY RISK
        # =================================================

        if payment_to_income > 0.50:

            affordability_status = "HIGH RISK"

        elif payment_to_income > 0.30:

            affordability_status = "MODERATE RISK"

        else:

            affordability_status = "LOW RISK"


        # =================================================
        # AFFORDABILITY RESPONSE
        # =================================================

        affordability = {

            "monthly_income":
                round(
                    monthly_income,
                    2
                ),

            "estimated_monthly_payment":
                round(
                    estimated_payment,
                    2
                ),

            "payment_to_income_ratio":
                round(
                    payment_to_income,
                    4
                ),

            "loan_to_income_ratio":
                round(
                    loan_to_income,
                    4
                ),

            "status":
                affordability_status
        }


        # =================================================
        # FINAL DECISION
        # =================================================
        #
        # IMPORTANT:
        #
        # The ML model can predict APPROVED.
        # However, if affordability is HIGH RISK,
        # the final system decision becomes MANUAL REVIEW.
        #
        # SHAP and LIME continue to explain the ML model.
        #
        # =================================================

        if affordability_status == "HIGH RISK":

            approval_status = "MANUAL REVIEW"

        elif prediction == 1:

            approval_status = "APPROVED"

        else:

            approval_status = "REJECTED"


        # =================================================
        # RECOMMENDATION
        # =================================================

        if affordability_status == "HIGH RISK":

            recommendation = (
                "High affordability risk - "
                "manual review required"
            )

        elif prediction == 1 and approval_probability >= 0.80:

            recommendation = (
                "Strong approval candidate"
            )

        elif prediction == 1 and approval_probability >= 0.60:

            recommendation = (
                "Likely to be approved"
            )

        elif prediction == 1:

            recommendation = (
                "Borderline approval - "
                "manual review recommended"
            )

        else:

            recommendation = (
                "High rejection risk"
            )


        # =================================================
        # LOG FINAL RESULT
        # =================================================

        logger.info(
            f"ML Prediction: "
            f"{'APPROVED' if prediction == 1 else 'REJECTED'} | "
            f"Probability: {confidence_percentage}% | "
            f"Affordability: {affordability_status} | "
            f"Final Decision: {approval_status}"
        )


        # =================================================
        # RETURN RESPONSE
        # =================================================

        return PredictionResponse(

            approval_status=
                approval_status,

            approval_probability=
                approval_probability,

            confidence_percentage=
                confidence_percentage,

            recommendation=
                recommendation,

            shap_explanation=
                shap_explanation,

            lime_explanation=
                lime_explanation,

            top_positive_factors=
                positive_factors[:5],

            top_negative_factors=
                negative_factors[:5],

            affordability=
                affordability
        )


    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except Exception as e:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(

            status_code=500,

            detail=
                f"Prediction failed: {str(e)}"
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health_check():

    return {

        "status":
            "healthy",

        "model_loaded":
            model is not None,

        "xai":
            [
                "SHAP",
                "LIME"
            ]
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )