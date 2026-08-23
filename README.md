# Lone_Approvel_System
Machine learning based loan approval prediction system with frontend and backend integration.


# Loan Approval Prediction System

This project is a machine learning based Loan Approval Prediction System developed as part of our college project. The main idea of the project is to predict whether a loan application will be approved or rejected based on details provided by the applicant.

For this project, we created a dataset of 1,000 loan applicants containing details such as age, income, credit score, employment years, loan amount, existing loans and loan term. We performed basic data analysis and preprocessing before training different supervised machine learning models.

We tested Logistic Regression, Decision Tree, Random Forest, KNN and SVM and compared their performance using accuracy, precision, recall and F1-score. Logistic Regression gave the best results among the models we tested, with an accuracy of 85.5% and an F1-score of 79.14%. The model was then saved and connected to a FastAPI backend for making predictions.

The project also includes a frontend where the user can enter the applicant details and get the loan approval prediction. Overall, this project helped us understand how a machine learning model can be developed, evaluated and integrated into a simple working application.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Streamlit / React
- GitHub

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM

## Project Flow

Dataset → Data Preprocessing → Model Training → Model Evaluation → Best Model Selection → FastAPI Backend → Frontend → Loan Prediction

## Result

Among the five models tested, Logistic Regression performed the best for our dataset.

**Accuracy:** 85.5%  
**F1-Score:** 79.14%  
**ROC-AUC:** 0.9530
