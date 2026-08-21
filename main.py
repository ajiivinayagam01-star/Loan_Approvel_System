from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
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

# Load trained model and scaler
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logger.info("✓ Model and scaler loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Error loading model files: {e}")
    raise


# Define request schema
class LoanApplication(BaseModel):
    """Loan applicant data schema"""
    income: float = Field(..., gt=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_years: float = Field(..., ge=0, description="Years of employment")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt-to-income ratio (0-1)")
    age: int = Field(..., ge=18, le=120, description="Applicant age")
    
    class Config:
        json_schema_extra = {
            "example": {
                "income": 75000,
                "credit_score": 720,
                "employment_years": 5,
                "loan_amount": 25000,
                "debt_to_income_ratio": 0.35,
                "age": 35
            }
        }


# Define response schema
class PredictionResponse(BaseModel):
    """API response schema"""
    approval_status: str
    approval_probability: float
    confidence_percentage: float
    recommendation: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Loan Approval Predictor API",
        "status": "running",
        "endpoints": {
            "health": "/",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_approval(application: LoanApplication):
    """
    Predict loan approval status for an applicant.
    
    Takes applicant financial data and returns approval prediction with confidence.
    """
    try:
        # Extract features in correct order (must match training data order)
        features = np.array([
            application.income,
            application.credit_score,
            application.employment_years,
            application.loan_amount,
            application.debt_to_income_ratio,
            application.age
        ]).reshape(1, -1)
        
        # Scale features using the trained scaler
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        # Extract probability of approval (class 1)
        approval_probability = float(prediction_proba[1])
        confidence_percentage = round(approval_probability * 100, 2)
        
        # Determine approval status
        approval_status = "APPROVED" if prediction == 1 else "DENIED"
        
        # Generate recommendation based on probability
        if approval_probability >= 0.8:
            recommendation = "Strong approval candidate"
        elif approval_probability >= 0.6:
            recommendation = "Likely to be approved"
        elif approval_probability >= 0.4:
            recommendation = "Borderline case - may need review"
        else:
            recommendation = "High rejection risk"
        
        logger.info(f"Prediction made: {approval_status} (confidence: {confidence_percentage}%)")
        
        return PredictionResponse(
            approval_status=approval_status,
            approval_probability=approval_probability,
            confidence_percentage=confidence_percentage,
            recommendation=recommendation
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
