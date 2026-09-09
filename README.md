# 💳 Credit Score Analysis & Loan Default Prediction

## 📌 Project Overview

This project focuses on analyzing credit-related customer information and predicting the likelihood of loan default using Machine Learning.

The system uses customer financial and loan-related attributes to identify whether a loan applicant is likely to default or not.

The project includes data preprocessing, exploratory data analysis, feature engineering, machine learning model training, evaluation, and prediction.

---

## 🎯 Objectives

- Analyze customer credit and loan data.
- Perform data cleaning and preprocessing.
- Handle missing values and duplicate records.
- Identify important factors related to loan default.
- Train Machine Learning classification models.
- Evaluate model performance using multiple metrics.
- Predict loan default risk for new customer data.

---

## 📊 Dataset

The project uses a credit risk dataset containing **32,581 records and 12 attributes**.

### Features

- `person_age` – Applicant age
- `person_income` – Applicant income
- `person_home_ownership` – Home ownership status
- `person_emp_length` – Employment length
- `loan_intent` – Purpose of the loan
- `loan_grade` – Loan grade
- `loan_amnt` – Loan amount
- `loan_int_rate` – Loan interest rate
- `loan_status` – Loan default status
- `loan_percent_income` – Loan amount as percentage of income
- `cb_person_default_on_file` – Previous default history
- `cb_person_cred_hist_length` – Credit history length

### Target

```text
0 → No Default
1 → Default
