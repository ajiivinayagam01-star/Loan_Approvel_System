import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# Page config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Header
st.title("💰 Loan Approval Predictor")
st.markdown("**AI-Powered Instant Loan Decision System**")
st.divider()

# Check API connectivity
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar - Info & Instructions
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        """
        This application uses machine learning to predict 
        loan approval decisions based on applicant financial data.
        
        **Features:**
        - Instant predictions
        - Confidence scoring
        - Financial recommendations
        """
    )
    
    st.divider()
    st.markdown("### 📋 Data Requirements")
    st.markdown("""
    - **Income**: Annual salary (USD)
    - **Credit Score**: 300-850
    - **Employment**: Years at current job
    - **Loan Amount**: Requested amount (USD)
    - **Debt Ratio**: Total debt / annual income
    - **Age**: 18-120 years
    """)
    
    st.divider()
    if check_api_health():
        st.success("✓ API Connected")
    else:
        st.error("✗ API Offline - Start main.py first")


# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Applicant Information")

# Create form
with st.form("loan_application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input(
            "Annual Income ($)",
            min_value=20000,
            max_value=500000,
            value=75000,
            step=5000,
            help="Gross annual income in USD"
        )
        
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=720,
            step=10,
            help="FICO credit score"
        )
        
        employment_years = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Years at current job"
        )
    
    with col2:
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=5000,
            max_value=500000,
            value=25000,
            step=5000,
            help="Requested loan amount in USD"
        )
        
        debt_to_income = st.slider(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Total monthly debt / monthly income"
        )
        
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=35,
            step=1,
            help="Applicant age in years"
        )
    
    st.divider()
    
    # Submit button
    submit_button = st.form_submit_button(
        "🔮 Predict Approval Status",
        use_container_width=True,
        type="primary"
    )

# Handle form submission
if submit_button:
    if not check_api_health():
        st.error(
            "❌ **API Connection Error**\n\n"
            "The backend server is not running. Please:\n"
            "1. Open a terminal\n"
            "2. Run: `python -m uvicorn main:app --reload --port 8000`\n"
            "3. Refresh this page"
        )
    else:
        # Prepare request payload
        payload = {
            "income": float(income),
            "credit_score": int(credit_score),
            "employment_years": float(employment_years),
            "loan_amount": float(loan_amount),
            "debt_to_income_ratio": float(debt_to_income),
            "age": int(age)
        }
        
        # Make API request
        with st.spinner("🔄 Processing application..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.divider()
                    st.subheader("📊 Prediction Results")
                    
                    # Display results based on approval status
                    if result["approval_status"] == "APPROVED":
                        st.markdown(
                            f"""
                            <div class="success-box">
                            <h3>✅ LOAN APPROVED</h3>
                            <p><strong>Confidence:</strong> {result['confidence_percentage']}%</p>
                            <p><strong>Recommendation:</strong> {result['recommendation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="danger-box">
                            <h3>❌ LOAN DENIED</h3>
                            <p><strong>Confidence:</strong> {result['confidence_percentage']}%</p>
                            <p><strong>Recommendation:</strong> {result['recommendation']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Display detailed metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Approval Probability",
                            f"{result['approval_probability']:.2%}"
                        )
                    
                    with col2:
                        st.metric(
                            "Decision",
                            result['approval_status']
                        )
                    
                    with col3:
                        st.metric(
                            "Confidence",
                            f"{result['confidence_percentage']}%"
                        )
                    
                    # Application Summary
                    st.divider()
                    st.subheader("📋 Application Summary")
                    
                    summary_data = {
                        "Metric": [
                            "Annual Income",
                            "Credit Score",
                            "Employment History",
                            "Requested Loan",
                            "Debt-to-Income Ratio",
                            "Age",
                            "Prediction Timestamp"
                        ],
                        "Value": [
                            f"${income:,.2f}",
                            credit_score,
                            f"{employment_years} years",
                            f"${loan_amount:,.2f}",
                            f"{debt_to_income*100:.1f}%",
                            f"{age} years",
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Insights
                    st.divider()
                    st.subheader("💡 Financial Insights")
                    
                    # Calculate and display insights
                    monthly_income = income / 12
                    monthly_debt = (debt_to_income * income) / 12
                    monthly_loan_payment = loan_amount / 60  # Assume 5-year loan
                    new_dti = (monthly_debt + monthly_loan_payment) / monthly_income
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(
                            f"**Current DTI Ratio:** {debt_to_income*100:.1f}%\n\n"
                            f"**Monthly Income:** ${monthly_income:,.2f}"
                        )
                    
                    with col2:
                        st.warning(
                            f"**Projected DTI (after loan):** {new_dti*100:.1f}%\n\n"
                            f"**Est. Monthly Payment (60mo):** ${monthly_loan_payment:,.2f}"
                        )
                    
                    # Export option
                    st.divider()
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 Download Report (JSON)"):
                            report = {
                                "prediction": result,
                                "application": payload,
                                "timestamp": datetime.now().isoformat()
                            }
                            st.download_button(
                                "Download JSON Report",
                                json.dumps(report, indent=2),
                                "loan_report.json",
                                "application/json"
                            )
                
                else:
                    st.error(f"❌ Prediction error: {response.json()}")
            
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Cannot Connect to API**\n\n"
                    "Make sure the FastAPI server is running:\n"
                    "`python -m uvicorn main:app --reload --port 8000`"
                )
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.caption(
    "💡 **Disclaimer:** This is a demonstration ML model. "
    "Real loan decisions should involve comprehensive financial review."
)
