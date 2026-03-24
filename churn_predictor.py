import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw churn data.
    This version is tailored for the common Telco Customer Churn dataset.
    """
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove customerID if it exists because it does not help prediction
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges if it exists and has blank spaces
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert target column Churn from Yes/No to 1/0
    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in dataset.")

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower()
    df["Churn"] = df["Churn"].map({"yes": 1, "no": 0})

    if df["Churn"].isna().any():
        raise ValueError("Target column 'Churn' contains unexpected values.")

    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print useful exploratory information.
    """
    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== COLUMN TYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["Churn"].value_counts())
    print("\n===== TARGET DISTRIBUTION (PROPORTION) =====")
    print(df["Churn"].value_counts(normalize=True))


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build preprocessing + model pipeline.
    """
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """
    Split data, train model, evaluate performance, and return trained pipeline.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline(X)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Plot ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.show()

    return pipeline


def save_model(model: Pipeline, output_path: str) -> None:
    """
    Save trained model to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")


def predict_new_customer(model: Pipeline, sample_input: pd.DataFrame) -> None:
    """
    Predict churn for a new customer sample.
    """
    prediction = model.predict(sample_input)[0]
    probability = model.predict_proba(sample_input)[0][1]

    label = "Will Churn" if prediction == 1 else "Will Stay"

    print("\n===== NEW CUSTOMER PREDICTION =====")
    print(f"Prediction: {label}")
    print(f"Churn Probability: {probability:.4f}")


def main() -> None:
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_path = "models/churn_model.joblib"

    # Load and clean data
    df = load_data(data_path)
    df = clean_data(df)

    # Explore
    explore_data(df)

    # Train and evaluate
    trained_model = train_and_evaluate(df)

    # Save model
    save_model(trained_model, model_path)

    # Example new customer prediction
    sample_customer = pd.DataFrame([
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 5,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 420.75
        }
    ])

    predict_new_customer(trained_model, sample_customer)


if __name__ == "__main__":
    main()