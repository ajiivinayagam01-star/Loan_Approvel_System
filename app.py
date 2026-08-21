import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


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
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
}

.danger-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
}

.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
}

.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# API CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000"


# =========================================================
# HEADER
# =========================================================

st.title("💰 Loan Approval Predictor")

st.markdown(
    "**AI-Powered Loan Decision & Explainable AI System**"
)

st.markdown(
    """
    This system combines Machine Learning with **SHAP and LIME**
    to explain which applicant parameters influenced the prediction.
    """
)

st.divider()


# =========================================================
# API HEALTH CHECK
# =========================================================

def check_api_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=3
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
        ### Loan Approval Predictor

        This application uses Machine Learning to
        predict loan approval.

        **XAI Methods**

        🔵 SHAP  
        Explains feature contributions.

        🟣 LIME  
        Explains the individual prediction.

        💰 Affordability  
        Calculates repayment burden.
        """
    )

    st.divider()

    st.markdown("### 📋 Input Requirements")

    st.markdown("""
    **Age**
    - 18–120 years

    **Annual Income**
    - Positive value

    **Credit Score**
    - 300–850

    **Employment**
    - Years of employment

    **Loan Amount**
    - Requested amount

    **Existing Loans**
    - Number of current loans

    **Loan Term**
    - Repayment period in months
    """)

    st.divider()

    if check_api_health():

        st.success("✓ FastAPI Connected")

    else:

        st.error("✗ FastAPI Offline")

        st.caption(
            "Start FastAPI using:\n\n"
            "python -m uvicorn main:app "
            "--reload --port 8000"
        )


# =========================================================
# APPLICATION FORM
# =========================================================

st.subheader("👤 Applicant Information")


with st.form("loan_application_form"):

    left, right = st.columns(2)


    # =====================================================
    # LEFT COLUMN
    # =====================================================

    with left:

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
            min_value=1000,
            max_value=5000000,
            value=75000,
            step=5000,
            help="Applicant's annual income"
        )


        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=720,
            step=1,
            help="Credit score"
        )


        employment_years = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Years of employment"
        )


    # =====================================================
    # RIGHT COLUMN
    # =====================================================

    with right:

        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            max_value=5000000,
            value=25000,
            step=5000,
            help="Requested loan amount"
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
            min_value=1,
            max_value=360,
            value=36,
            step=6,
            help="Requested repayment period"
        )


    st.divider()


    submit_button = st.form_submit_button(
        "🔮 Predict Loan Decision",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# HANDLE SUBMISSION
# =========================================================

if submit_button:

    # =====================================================
    # CHECK API
    # =====================================================

    if not check_api_health():

        st.error(
            """
            ❌ **Backend API is not available.**

            Please start FastAPI:

            `python -m uvicorn main:app --reload --port 8000`
            """
        )

        st.stop()


    # =====================================================
    # PREPARE PAYLOAD
    # =====================================================

    payload = {

        "age": int(age),

        "income": float(income),

        "credit_score": int(credit_score),

        "employment_years":
            float(employment_years),

        "loan_amount":
            float(loan_amount),

        "existing_loans":
            int(existing_loans),

        "loan_term":
            int(loan_term)
    }


    # =====================================================
    # DISPLAY REQUEST DEBUG INFO
    # =====================================================

    with st.expander("🔧 Request Data", expanded=False):

        st.json(payload)


    # =====================================================
    # SEND REQUEST
    # =====================================================

    with st.spinner(
        "🤖 Machine Learning model is analyzing the application..."
    ):

        try:

            response = requests.post(

                f"{API_URL}/predict",

                json=payload,

                timeout=30
            )


            # =================================================
            # SUCCESS
            # =================================================

            if response.status_code == 200:

                result = response.json()


                # =================================================
                # GET RESULT VALUES
                # =================================================

                decision = result.get(
                    "approval_status",
                    "UNKNOWN"
                )

                probability = result.get(
                    "approval_probability",
                    0
                )

                confidence = result.get(
                    "confidence_percentage",
                    0
                )

                recommendation = result.get(
                    "recommendation",
                    "No recommendation available."
                )


                shap_data = result.get(
                    "shap_explanation",
                    []
                )

                lime_data = result.get(
                    "lime_explanation",
                    []
                )

                positive_factors = result.get(
                    "top_positive_factors",
                    []
                )

                negative_factors = result.get(
                    "top_negative_factors",
                    []
                )

                affordability = result.get(
                    "affordability",
                    {}
                )


                # =================================================
                # PREDICTION RESULT
                # =================================================

                st.divider()

                st.subheader(
                    "📊 Prediction Result"
                )


                # =================================================
                # APPROVED
                # =================================================

                if decision == "APPROVED":

                    st.markdown(
                        f"""
                        <div class="success-box">

                        <h2>✅ LOAN APPROVED</h2>

                        <p>
                        <strong>ML Approval Probability:</strong>
                        {probability:.2%}
                        </p>

                        <p>
                        <strong>Confidence:</strong>
                        {confidence}%
                        </p>

                        <p>
                        <strong>Recommendation:</strong>
                        {recommendation}
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # MANUAL REVIEW
                # =================================================

                elif decision == "MANUAL REVIEW":

                    st.markdown(
                        f"""
                        <div class="warning-box">

                        <h2>⚠️ MANUAL REVIEW</h2>

                        <p>
                        <strong>ML Approval Probability:</strong>
                        {probability:.2%}
                        </p>

                        <p>
                        <strong>Confidence:</strong>
                        {confidence}%
                        </p>

                        <p>
                        <strong>Recommendation:</strong>
                        {recommendation}
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # REJECTED
                # =================================================

                else:

                    st.markdown(
                        f"""
                        <div class="danger-box">

                        <h2>❌ LOAN REJECTED</h2>

                        <p>
                        <strong>ML Approval Probability:</strong>
                        {probability:.2%}
                        </p>

                        <p>
                        <strong>Confidence:</strong>
                        {confidence}%
                        </p>

                        <p>
                        <strong>Recommendation:</strong>
                        {recommendation}
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # METRICS
                # =================================================

                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Approval Probability",
                        f"{probability:.2%}"
                    )


                with col2:

                    st.metric(
                        "ML Decision",
                        decision
                    )


                with col3:

                    st.metric(
                        "Confidence",
                        f"{confidence}%"
                    )


                # =================================================
                # AFFORDABILITY ANALYSIS
                # =================================================

                st.divider()

                st.subheader(
                    "💰 Affordability Analysis"
                )


                if affordability:

                    monthly_income = affordability.get(
                        "monthly_income",
                        income / 12
                    )

                    estimated_payment = affordability.get(
                        "estimated_monthly_payment",
                        loan_amount / loan_term
                    )

                    payment_ratio = affordability.get(
                        "payment_to_income_ratio",
                        0
                    )

                    loan_ratio = affordability.get(
                        "loan_to_income_ratio",
                        0
                    )

                    affordability_status = affordability.get(
                        "status",
                        "UNKNOWN"
                    )


                    # =============================================
                    # FINANCIAL METRICS
                    # =============================================

                    col1, col2 = st.columns(2)


                    with col1:

                        st.metric(
                            "Monthly Income",
                            f"${monthly_income:,.2f}"
                        )

                        st.metric(
                            "Loan / Annual Income",
                            f"{loan_ratio:.2%}"
                        )


                    with col2:

                        st.metric(
                            "Estimated Monthly Payment",
                            f"${estimated_payment:,.2f}"
                        )

                        st.metric(
                            "Payment / Income",
                            f"{payment_ratio:.2%}"
                        )


                    # =============================================
                    # AFFORDABILITY STATUS
                    # =============================================

                    if affordability_status == "HIGH RISK":

                        st.error(
                            """
                            🔴 **HIGH AFFORDABILITY RISK**

                            The estimated monthly repayment is high
                            relative to the applicant's monthly income.
                            """
                        )


                    elif affordability_status == "MODERATE RISK":

                        st.warning(
                            """
                            🟠 **MODERATE AFFORDABILITY RISK**

                            The repayment burden requires additional
                            financial consideration.
                            """
                        )


                    else:

                        st.success(
                            """
                            🟢 **LOW AFFORDABILITY RISK**

                            The estimated repayment burden is relatively
                            manageable compared with monthly income.
                            """
                        )


                # =================================================
                # XAI SECTION
                # =================================================

                st.divider()

                st.subheader(
                    "🔍 Explainable AI (XAI)"
                )

                st.write(
                    """
                    XAI explains **which parameters influenced the
                    model's prediction** and whether they contributed
                    toward approval or rejection.
                    """
                )


                # =================================================
                # MAIN DECISION FACTORS
                # =================================================

                st.markdown(
                    "### 🎯 Main Decision Factors"
                )


                # =================================================
                # NEGATIVE FACTORS
                # =================================================

                if negative_factors:

                    st.markdown(
                        "#### 🔴 Factors Increasing Rejection Risk"
                    )


                    for factor in negative_factors:

                        feature = factor.get(
                            "feature",
                            "Unknown"
                        )

                        contribution = factor.get(
                            "contribution",
                            0
                        )


                        st.write(
                            f"🔴 **{feature}**  \n"
                            f"Contribution: `{contribution:.4f}`"
                        )


                else:

                    st.info(
                        "No negative SHAP factors were returned."
                    )


                # =================================================
                # POSITIVE FACTORS
                # =================================================

                if positive_factors:

                    st.markdown(
                        "#### 🟢 Factors Supporting Approval"
                    )


                    for factor in positive_factors:

                        feature = factor.get(
                            "feature",
                            "Unknown"
                        )

                        contribution = factor.get(
                            "contribution",
                            0
                        )


                        st.write(
                            f"🟢 **{feature}**  \n"
                            f"Contribution: `{contribution:.4f}`"
                        )


                else:

                    st.info(
                        "No positive SHAP factors were returned."
                    )


                # =================================================
                # SHAP CHART
                # =================================================

                st.markdown(
                    "### 📊 SHAP Feature Contributions"
                )


                if shap_data:

                    try:

                        shap_df = pd.DataFrame(
                            shap_data
                        )


                        shap_df = shap_df.sort_values(
                            by="contribution"
                        )


                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )


                        ax.barh(
                            shap_df["feature"],
                            shap_df["contribution"]
                        )


                        ax.axvline(
                            0,
                            linewidth=1
                        )


                        ax.set_xlabel(
                            "Contribution Toward Approval"
                        )


                        ax.set_ylabel(
                            "Parameter"
                        )


                        ax.set_title(
                            "SHAP Explanation"
                        )


                        plt.tight_layout()


                        st.pyplot(
                            fig
                        )


                        plt.close(fig)


                    except Exception as e:

                        st.warning(
                            f"Could not display SHAP chart: {e}"
                        )


                else:

                    st.info(
                        "SHAP explanation is not available."
                    )


                # =================================================
                # SHAP DATA TABLE
                # =================================================

                if shap_data:

                    with st.expander(
                        "📋 View SHAP Values"
                    ):

                        shap_df = pd.DataFrame(
                            shap_data
                        )

                        st.dataframe(
                            shap_df,
                            use_container_width=True,
                            hide_index=True
                        )


                # =================================================
                # LIME EXPLANATION
                # =================================================

                st.markdown(
                    "### 🧠 LIME Local Explanation"
                )


                st.write(
                    """
                    LIME explains the current applicant's prediction
                    by identifying the local feature conditions that
                    influenced the model.
                    """
                )


                if lime_data:

                    try:

                        lime_df = pd.DataFrame(
                            lime_data
                        )


                        lime_df = lime_df.sort_values(
                            by="contribution"
                        )


                        fig, ax = plt.subplots(
                            figsize=(8, 5)
                        )


                        ax.barh(
                            lime_df["feature"],
                            lime_df["contribution"]
                        )


                        ax.axvline(
                            0,
                            linewidth=1
                        )


                        ax.set_xlabel(
                            "LIME Contribution"
                        )


                        ax.set_ylabel(
                            "Feature / Condition"
                        )


                        ax.set_title(
                            "LIME Explanation for This Applicant"
                        )


                        plt.tight_layout()


                        st.pyplot(
                            fig
                        )


                        plt.close(fig)


                    except Exception as e:

                        st.warning(
                            f"Could not display LIME chart: {e}"
                        )


                else:

                    st.info(
                        "LIME explanation is not available."
                    )


                # =================================================
                # LIME DATA TABLE
                # =================================================

                if lime_data:

                    with st.expander(
                        "📋 View LIME Values"
                    ):

                        lime_df = pd.DataFrame(
                            lime_data
                        )

                        st.dataframe(
                            lime_df,
                            use_container_width=True,
                            hide_index=True
                        )


                # =================================================
                # HUMAN READABLE EXPLANATION
                # =================================================

                st.divider()

                st.subheader(
                    "💡 Why was this decision made?"
                )


                # =================================================
                # REJECTED EXPLANATION
                # =================================================

                if decision == "REJECTED":

                    if negative_factors:

                        top_negative = (
                            negative_factors[:3]
                        )


                        names = ", ".join(
                            [
                                str(
                                    item["feature"]
                                )

                                for item in top_negative
                            ]
                        )


                        st.error(
                            f"""
                            **Primary decision factors:**

                            The strongest factors pushing the model
                            toward rejection were:

                            **{names}**
                            """
                        )

                    else:

                        st.warning(
                            """
                            The model rejected the application,
                            but detailed SHAP factors were unavailable.
                            """
                        )


                # =================================================
                # APPROVED EXPLANATION
                # =================================================

                elif decision == "APPROVED":

                    if positive_factors:

                        top_positive = (
                            positive_factors[:3]
                        )


                        names = ", ".join(
                            [
                                str(
                                    item["feature"]
                                )

                                for item in top_positive
                            ]
                        )


                        st.success(
                            f"""
                            **Primary decision factors:**

                            The strongest factors supporting approval
                            were:

                            **{names}**
                            """
                        )

                    else:

                        st.info(
                            """
                            The model approved the application,
                            but detailed SHAP factors were unavailable.
                            """
                        )


                # =================================================
                # MANUAL REVIEW
                # =================================================

                else:

                    st.warning(
                        """
                        The application requires additional review.
                        Examine the affordability metrics and XAI
                        feature contributions before making a final
                        decision.
                        """
                    )


                # =================================================
                # APPLICATION SUMMARY
                # =================================================

                st.divider()

                st.subheader(
                    "📋 Application Summary"
                )


                summary_data = {

                    "Parameter": [

                        "Age",

                        "Annual Income",

                        "Credit Score",

                        "Employment Years",

                        "Loan Amount",

                        "Existing Loans",

                        "Loan Term",

                        "Prediction Time"
                    ],


                    "Value": [

                        f"{age} years",

                        f"${income:,.2f}",

                        f"{credit_score}",

                        f"{employment_years} years",

                        f"${loan_amount:,.2f}",

                        f"{existing_loans}",

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
                # DOWNLOAD REPORT
                # =================================================

                st.divider()

                st.subheader(
                    "📥 Download XAI Report"
                )


                report = {

                    "application":
                        payload,

                    "prediction":
                        result,

                    "timestamp":
                        datetime.now().isoformat()
                }


                st.download_button(

                    label=
                        "📄 Download JSON Report",

                    data=json.dumps(
                        report,
                        indent=2
                    ),

                    file_name=
                        "loan_xai_report.json",

                    mime=
                        "application/json",

                    use_container_width=True
                )


            # =====================================================
            # API ERROR
            # =====================================================

            else:

                try:

                    error_details = (
                        response.json()
                    )

                except Exception:

                    error_details = (
                        response.text
                    )


                st.error(
                    f"❌ Prediction error: "
                    f"{error_details}"
                )


        # =========================================================
        # CONNECTION ERROR
        # =========================================================

        except requests.exceptions.ConnectionError:

            st.error(
                """
                ❌ **Cannot connect to FastAPI.**

                Make sure FastAPI is running:

                `python -m uvicorn main:app --reload --port 8000`
                """
            )


        # =========================================================
        # TIMEOUT ERROR
        # =========================================================

        except requests.exceptions.Timeout:

            st.error(
                """
                ❌ **Request timed out.**

                SHAP and LIME explanations can take additional
                processing time. Please try again.
                """
            )


        # =========================================================
        # OTHER ERROR
        # =========================================================

        except Exception as e:

            st.error(
                f"❌ Unexpected error: {str(e)}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    """
    💡 Disclaimer: This is a demonstration Machine Learning system.
    SHAP and LIME explain model behavior; they do not guarantee
    loan repayment or constitute a real financial lending decision.
    """
)