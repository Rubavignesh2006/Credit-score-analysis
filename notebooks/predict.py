import joblib
import pandas as pd

# Load saved Random Forest model
model = joblib.load("credit_scoring_model.pkl")

print("Model loaded successfully!")

print("\n===== NEW BORROWER PREDICTION =====")

age = float(input("Enter age: "))
income = float(input("Enter annual income: "))
emp_length = float(input("Enter employment length: "))
loan_amount = float(input("Enter loan amount: "))
interest_rate = float(input("Enter loan interest rate: "))
loan_percent_income = float(input("Enter loan percent income: "))
credit_history = float(input("Enter credit history length: "))

home_ownership = input("Enter home ownership (RENT/OWN/MORTGAGE/OTHER): ").upper()
loan_intent = input("Enter loan intent (PERSONAL/EDUCATION/MEDICAL/VENTURE/HOMEIMPROVEMENT/DEBTCONSOLIDATION): ").upper()
loan_grade = input("Enter loan grade (A/B/C/D/E/F/G): ").upper()
default_on_file = input("Any default on file? (Y/N): ").upper()

# Manual mappings (same as training)
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
default_map = {"N": 0, "Y": 1}

loan_grade_encoded = grade_map[loan_grade]
default_encoded = default_map[default_on_file]

# Engineered features
loan_to_income = loan_amount / income
income_per_loan = income / loan_amount

# Build input dict with all 21 features, default 0
input_data = {
    "person_age": age,
    "person_income": income,
    "person_emp_length": emp_length,
    "loan_grade": loan_grade_encoded,
    "loan_amnt": loan_amount,
    "loan_int_rate": interest_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": default_encoded,
    "cb_person_cred_hist_length": credit_history,
    "loan_to_income": loan_to_income,
    "income_per_loan": income_per_loan,
    "person_home_ownership_MORTGAGE": 0,
    "person_home_ownership_OTHER": 0,
    "person_home_ownership_OWN": 0,
    "person_home_ownership_RENT": 0,
    "loan_intent_DEBTCONSOLIDATION": 0,
    "loan_intent_EDUCATION": 0,
    "loan_intent_HOMEIMPROVEMENT": 0,
    "loan_intent_MEDICAL": 0,
    "loan_intent_PERSONAL": 0,
    "loan_intent_VENTURE": 0
}

# Set correct one-hot columns to 1
input_data[f"person_home_ownership_{home_ownership}"] = 1
input_data[f"loan_intent_{loan_intent}"] = 1

# Convert to DataFrame in EXACT column order model expects
input_df = pd.DataFrame([input_data])
input_df = input_df[model.feature_names_in_]

# Predict
prediction = model.predict(input_df)[0]

print("\n===== RESULT =====")
if prediction == 1:
    print("Prediction: Default (High Risk)")
else:
    print("Prediction: No Default (Low Risk)")