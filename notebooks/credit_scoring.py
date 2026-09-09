import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
df = pd.read_csv("data/credit_risk_dataset.csv")
print(df.head())
print(df.columns)
print(df.shape)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)
df["person_emp_length"] = df["person_emp_length"].fillna(
    df["person_emp_length"].median()
)
print(df.isnull().sum())
print(df.info())
print("Duplicate rows:", df.duplicated().sum())
print("Final dataset shape:", df.shape)
print(df.columns)
print(df["loan_status"].value_counts())
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print(df.select_dtypes(include="object").columns)
df["loan_to_income"] = df["loan_amnt"] / df["person_income"]

print(df[["loan_amnt", "person_income", "loan_to_income"]].head())
df["income_per_loan"] = df["person_income"] / df["loan_amnt"]

print(df[["person_income", "loan_amnt", "income_per_loan"]].head())
print(df[[
    "loan_amnt",
    "person_income",
    "loan_to_income",
    "income_per_loan"
]].head())
print(df[[
    "person_income",
    "loan_amnt",
    "loan_to_income",
    "income_per_loan"
]].describe())
print("Home ownership:")
print(df["person_home_ownership"].value_counts())

print("\nLoan intent:")
print(df["loan_intent"].value_counts())

print("\nLoan grade:")
print(df["loan_grade"].value_counts())

print("\nDefault history:")
print(df["cb_person_default_on_file"].value_counts())
df["loan_grade"] = df["loan_grade"].map({
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
})

df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map({
    "N": 0,
    "Y": 1
})

print(df[["loan_grade", "cb_person_default_on_file"]].head())
df = pd.get_dummies(
    df,
    columns=["person_home_ownership"],
    dtype=int
)

print(df.columns)
df = pd.get_dummies(
    df,
    columns=["loan_intent"],
    dtype=int
)

print(df.columns)
print("Columns after encoding:", df.shape)
print(df.columns)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(df.isnull().sum())
print(df.isnull().sum()[df.isnull().sum() > 0])
df["loan_int_rate"] = df["loan_int_rate"].fillna(
    df["loan_int_rate"].median()
)

print("Missing loan_int_rate:", df["loan_int_rate"].isnull().sum())
print("Total missing values:", df.isnull().sum().sum())
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled:", X_train_scaled.shape)
print("X_test_scaled:", X_test_scaled.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

print("Logistic Regression training completed!")
y_pred = model.predict(X_test_scaled)

print("Predictions:", y_pred[:20])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=8
)

dt_model.fit(X_train, y_train)

print("Decision Tree training completed!")
dt_pred = dt_model.predict(X_test)

print("Decision Tree Predictions:", dt_pred[:20])
from sklearn.metrics import accuracy_score

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Accuracy percentage:", dt_accuracy * 100, "%")
print(classification_report(y_test, dt_pred))
print("Model Comparison")
print("-----------------")
print("Logistic Regression Accuracy:", accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
from sklearn.metrics import recall_score

logistic_recall = recall_score(y_test, y_pred)
dt_recall = recall_score(y_test, dt_pred)

print("Logistic Regression Recall:", logistic_recall)
print("Decision Tree Recall:", dt_recall)
from sklearn.metrics import f1_score

logistic_f1 = f1_score(y_test, y_pred)
dt_f1 = f1_score(y_test, dt_pred)

print("Logistic Regression F1:", logistic_f1)
print("Decision Tree F1:", dt_f1)
from sklearn.metrics import precision_score

logistic_precision = precision_score(y_test, y_pred)
dt_precision = precision_score(y_test, dt_pred)

print("Logistic Regression Precision:", logistic_precision)
print("Decision Tree Precision:", dt_precision)
print("\n===== MODEL COMPARISON =====")

print("\nLogistic Regression")
print("Accuracy :", accuracy)
print("Precision:", logistic_precision)
print("Recall   :", logistic_recall)
print("F1-score :", logistic_f1)

print("\nDecision Tree")
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1-score :", dt_f1)
# Step 58 — Random Forest Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# 1. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 2. Train the model
rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# 3. Make predictions
rf_pred = rf_model.predict(X_test)

print("\nFirst 20 Predictions:")
print(rf_pred[:20])

# 4. Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# 5. Display results
print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1-score :", rf_f1)

# 6. Detailed classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, rf_pred))
import pandas as pd

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10))
model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        dt_accuracy,
        rf_accuracy
    ],
    "Precision": [
        logistic_precision,
        dt_precision,
        rf_precision
    ],
    "Recall": [
        logistic_recall,
        dt_recall,
        rf_recall
    ],
    "F1-Score": [
        logistic_f1,
        dt_f1,
        rf_f1
    ]
})

print(model_comparison)
import joblib

joblib.dump(rf_model, "credit_scoring_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Random Forest model saved successfully!")
print("Scaler saved successfully!")
import os

print("Model exists:", os.path.exists("credit_scoring_model.pkl"))
print("Scaler exists:", os.path.exists("scaler.pkl"))
import joblib

loaded_model = joblib.load("credit_scoring_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

print("Saved model loaded successfully!")
print("Saved scaler loaded successfully!")
sample = X_test.iloc[[0]]

prediction = loaded_model.predict(sample)

print("Actual loan status:", y_test.iloc[0])
print("Predicted loan status:", prediction[0])

if prediction[0] == 0:
    print("Prediction: No Default")
else:
    print("Prediction: Default")

    # Step 65 - Test an actual Default borrower

default_sample = X_test[y_test == 1].iloc[[0]]

default_prediction = loaded_model.predict(default_sample)

print("Actual loan status: 1")
print("Predicted loan status:", default_prediction[0])

if default_prediction[0] == 0:
    print("Prediction: No Default")
else:
    print("Prediction: Default")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 66 - Confusion Matrix

y_pred = loaded_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Default", "Default"]
)

disp.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()
# Step 68 - Feature Importance

import pandas as pd
import matplotlib.pyplot as plt

importance = loaded_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

feature_importance.plot(
    x="Feature",
    y="Importance",
    kind="bar",
    legend=False
)

plt.title("Random Forest - Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Step 69 - Save Feature Importance

feature_importance.to_csv(
    "../data/feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
import joblib

# Step 70 - Save Final Decision Tree Model

joblib.dump(dt_model, "../data/final_credit_model.pkl")

print("Final Decision Tree model saved successfully!")
import joblib

# Check saved model
loaded_model = joblib.load("../data/model.pkl")

print("Saved model loaded successfully!")
print("Model type:", type(loaded_model))
import joblib

# Load existing saved model
loaded_model = joblib.load("../data/model.pkl")

print("Saved model loaded successfully!")
print("Model type:", type(loaded_model))
print(df.columns)