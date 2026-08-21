import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

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
    """,
    unsafe_allow_html=True
)


# =========================================================
# API CONFIGURATION
# =========================================================

API_URL = "http://localhost:8000"


# =========================================================
# HEADER
# =========================================================

st.title("💰 Loan Approval Predictor")
st.markdown("**AI-Powered Instant Loan Decision System**")
st.divider()


# =========================================================
# CHECK API CONNECTIVITY
# =========================================================

def check_api_health():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=2
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


# =========================================================
# SIDEBAR
# =========================================================

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
    - **Age**: 18-120 years
    - **Income**: Annual salary (USD)
    - **Credit Score**: 300-850
    - **Employment**: Years at current job
    - **Loan Amount**: Requested amount (USD)
    - **Existing Loans**: Number of existing loans
    - **Loan Term**: Repayment period in months
    """)

    st.divider()

    if check_api_health():

        st.success("✓ API Connected")

    else:

        st.error("✗ API Offline - Start FastAPI first")


# =========================================================
# MAIN CONTENT
# =========================================================

col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("Enter Applicant Information")


    # =====================================================
    # APPLICATION FORM
    # =====================================================

    with st.form("loan_application_form"):

        col1, col2 = st.columns(2)


        # =================================================
        # LEFT COLUMN
        # =================================================

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=35,
                step=1,
                help="Applicant age in years"
            )


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


        # =================================================
        # RIGHT COLUMN
        # =================================================

        with col2:

            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=5000,
                max_value=500000,
                value=25000,
                step=5000,
                help="Requested loan amount in USD"
            )


            existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
                help="Number of existing loans"
            )


            loan_term = st.number_input(
                "Loan Term (Months)",
                min_value=6,
                max_value=120,
                value=36,
                step=6,
                help="Requested loan repayment period"
            )


        st.divider()


        # =================================================
        # SUBMIT BUTTON
        # =================================================

        submit_button = st.form_submit_button(
            "🔮 Predict Approval Status",
            use_container_width=True,
            type="primary"
        )


# =========================================================
# HANDLE FORM SUBMISSION
# =========================================================

if submit_button:

    # =====================================================
    # CHECK BACKEND
    # =====================================================

    if not check_api_health():

        st.error(
            "❌ **API Connection Error**\n\n"
            "The backend server is not running.\n\n"
            "Start it using:\n\n"
            "`python -m uvicorn main:app --reload --port 8000`"
        )

    else:

        # =================================================
        # PREPARE REQUEST PAYLOAD
        # =================================================

        payload = {

            "age": int(age),

            "income": float(income),

            "credit_score": int(credit_score),

            "employment_years": float(
                employment_years
            ),

            "loan_amount": float(
                loan_amount
            ),

            "existing_loans": int(
                existing_loans
            ),

            "loan_term": int(
                loan_term
            )
        }


        # =================================================
        # SEND REQUEST TO FASTAPI
        # =================================================

        with st.spinner(
            "🔄 Processing application..."
        ):

            try:

                response = requests.post(

                    f"{API_URL}/predict",

                    json=payload,

                    timeout=10
                )


                # =================================================
                # SUCCESS RESPONSE
                # =================================================

                if response.status_code == 200:

                    result = response.json()


                    st.divider()

                    st.subheader(
                        "📊 Prediction Results"
                    )


                    # =================================================
                    # APPROVED
                    # =================================================

                    if result["approval_status"] == "APPROVED":

                        st.markdown(
                            f"""
                            <div class="success-box">

                            <h3>✅ LOAN APPROVED</h3>

                            <p>
                            <strong>Confidence:</strong>
                            {result['confidence_percentage']}%
                            </p>

                            <p>
                            <strong>Recommendation:</strong>
                            {result['recommendation']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # =================================================
                    # DENIED
                    # =================================================

                    else:

                        st.markdown(
                            f"""
                            <div class="danger-box">

                            <h3>❌ LOAN DENIED</h3>

                            <p>
                            <strong>Confidence:</strong>
                            {result['confidence_percentage']}%
                            </p>

                            <p>
                            <strong>Recommendation:</strong>
                            {result['recommendation']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                    # =================================================
                    # DETAILED METRICS
                    # =================================================

                    col1, col2, col3 = st.columns(3)


                    with col1:

                        st.metric(
                            "Approval Probability",
                            f"{result['approval_probability']:.2%}"
                        )


                    with col2:

                        st.metric(
                            "Decision",
                            result["approval_status"]
                        )


                    with col3:

                        st.metric(
                            "Confidence",
                            f"{result['confidence_percentage']}%"
                        )


                    # =================================================
                    # APPLICATION SUMMARY
                    # =================================================

                    st.divider()

                    st.subheader(
                        "📋 Application Summary"
                    )


                    summary_data = {

                        "Metric": [

                            "Age",

                            "Annual Income",

                            "Credit Score",

                            "Employment History",

                            "Requested Loan",

                            "Existing Loans",

                            "Loan Term",

                            "Prediction Timestamp"
                        ],


                        "Value": [

                            f"{age} years",

                            f"${income:,.2f}",

                            credit_score,

                            f"{employment_years} years",

                            f"${loan_amount:,.2f}",

                            existing_loans,

                            f"{loan_term} months",

                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        ]
                    }


                    summary_df = pd.DataFrame(
                        summary_data
                    )


                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )


                    # =================================================
                    # FINANCIAL INSIGHTS
                    # =================================================

                    st.divider()

                    st.subheader(
                        "💡 Financial Insights"
                    )


                    monthly_income = income / 12


                    estimated_monthly_payment = (
                        loan_amount / loan_term
                    )


                    col1, col2 = st.columns(2)


                    with col1:

                        st.info(
                            f"""
                            **Monthly Income:**
                            ${monthly_income:,.2f}

                            **Existing Loans:**
                            {existing_loans}
                            """
                        )


                    with col2:

                        st.warning(
                            f"""
                            **Estimated Monthly Payment:**
                            ${estimated_monthly_payment:,.2f}

                            **Loan Term:**
                            {loan_term} months
                            """
                        )


                    # =================================================
                    # DOWNLOAD REPORT
                    # =================================================

                    st.divider()

                    st.subheader(
                        "📥 Download Report"
                    )


                    report = {

                        "prediction": result,

                        "application": payload,

                        "timestamp":
                            datetime.now().isoformat()
                    }


                    st.download_button(

                        label="Download JSON Report",

                        data=json.dumps(
                            report,
                            indent=2
                        ),

                        file_name="loan_report.json",

                        mime="application/json"
                    )


                # =====================================================
                # API ERROR
                # =====================================================

                else:

                    try:

                        error_details = response.json()

                    except:

                        error_details = response.text


                    st.error(
                        f"❌ Prediction error: "
                        f"{error_details}"
                    )


            # =========================================================
            # CONNECTION ERROR
            # =========================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ **Cannot Connect to API**\n\n"

                    "Make sure the FastAPI server is running:\n\n"

                    "`python -m uvicorn main:app "
                    "--reload --port 8000`"
                )


            # =========================================================
            # OTHER ERROR
            # =========================================================

            except Exception as e:

                st.error(
                    f"❌ Error: {str(e)}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "💡 **Disclaimer:** This is a demonstration ML model. "
    "Real loan decisions should involve comprehensive financial review."
)