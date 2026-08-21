from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Predictor API",
    description="ML API for predicting loan approval status",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
try:
    model = joblib.load("model.pkl")
    logger.info("✓ Model loaded successfully")

except FileNotFoundError as e:
    logger.error(f"Error loading model file: {e}")
    raise


# Request schema
class LoanApplication(BaseModel):
    age: int = Field(
        ..., ge=18, le=120,
        description="Applicant age"
    )

    income: float = Field(
        ..., gt=0,
        description="Annual income in USD"
    )

    credit_score: int = Field(
        ..., ge=300, le=850,
        description="Credit score (300-850)"
    )

    employment_years: float = Field(
        ..., ge=0,
        description="Years of employment"
    )

    loan_amount: float = Field(
        ..., gt=0,
        description="Requested loan amount in USD"
    )

    existing_loans: int = Field(
        ..., ge=0,
        description="Number of existing loans"
    )

    loan_term: int = Field(
        ..., gt=0,
        description="Loan term in months"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "income": 75000,
                "credit_score": 720,
                "employment_years": 5,
                "loan_amount": 25000,
                "existing_loans": 2,
                "loan_term": 36
            }
        }


# Response schema
class PredictionResponse(BaseModel):
    approval_status: str
    approval_probability: float
    confidence_percentage: float
    recommendation: str


@app.get("/")
async def root():
    return {
        "service": "Loan Approval Predictor API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_approval(application: LoanApplication):

    try:
        # IMPORTANT:
        # Feature order must exactly match train_model.py
        features = np.array([[
            application.age,
            application.income,
            application.credit_score,
            application.employment_years,
            application.loan_amount,
            application.existing_loans,
            application.loan_term
        ]])

        # model.pkl contains:
        # StandardScaler + Logistic Regression
        prediction = model.predict(features)[0]

        prediction_proba = model.predict_proba(features)[0]

        # Class 1 = Approved
        approval_probability = float(prediction_proba[1])

        confidence_percentage = round(
            approval_probability * 100,
            2
        )

        # Approval status
        approval_status = (
            "APPROVED"
            if prediction == 1
            else "DENIED"
        )

        # Recommendation
        if approval_probability >= 0.8:
            recommendation = "Strong approval candidate"

        elif approval_probability >= 0.6:
            recommendation = "Likely to be approved"

        elif approval_probability >= 0.4:
            recommendation = "Borderline case - may need review"

        else:
            recommendation = "High rejection risk"

        logger.info(
            f"Prediction made: {approval_status} "
            f"(confidence: {confidence_percentage}%)"
        )

        return PredictionResponse(
            approval_status=approval_status,
            approval_probability=approval_probability,
            confidence_percentage=confidence_percentage,
            recommendation=recommendation
        )

    except Exception as e:

        logger.error(
            f"Prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )