# score.py
# Scoring script for Customer Churn Prediction Model
# Usage: python score.py

import pandas as pd
import numpy as np
import joblib
import os

# ── 1. Load model and feature columns ──────────────────────────────────────
MODEL_PATH = '../models/xgb_churn_model.pkl'
FEATURES_PATH = '../models/feature_columns.pkl'

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

print("Model loaded successfully.")
print(f"Expected features: {len(feature_columns)}")

# ── 2. Define scoring function ─────────────────────────────────────────────
def preprocess(df_input):
    """
    Preprocess raw customer data to match training format.
    Applies the same cleaning and encoding steps used during training.
    """
    df = df_input.copy()

    # Drop identifier columns if present
    df = df.drop(columns=['Customer ID', 'Customer Status',
                           'Churn Category', 'Churn Reason',
                           'City', 'Zip Code', 'Latitude', 'Longitude',
                           'Churn'], errors='ignore')

    # Fill missing values
    service_columns = [
        'Online Backup', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Premium Tech Support',
        'Device Protection Plan', 'Online Security', 'Internet Type'
    ]
    for col in service_columns:
        if col in df.columns:
            df[col] = df[col].fillna('No')

    df['Multiple Lines'] = df['Multiple Lines'].fillna('No')
    df['Unlimited Data'] = df['Unlimited Data'].fillna('No')
    df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(0)

    if 'Avg Monthly Long Distance Charges' in df.columns:
        df['Avg Monthly Long Distance Charges'] = df[
            'Avg Monthly Long Distance Charges'
        ].fillna(df['Avg Monthly Long Distance Charges'].median())

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training features
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def score(df_input):
    """
    Generate churn predictions for new customer data.
    Returns original dataframe with Churn_Probability and Churn_Prediction columns.
    """
    df_processed = preprocess(df_input)
    probabilities = model.predict_proba(df_processed)[:, 1]
    predictions = model.predict(df_processed)

    df_output = df_input.copy()
    df_output['Churn_Probability'] = probabilities.round(3)
    df_output['Churn_Prediction'] = predictions
    df_output['Churn_Prediction_Label'] = df_output['Churn_Prediction'].map(
        {1: 'Churned', 0: 'Stayed'}
    )

    return df_output


# ── 3. Example usage ───────────────────────────────────────────────────────
if __name__ == '__main__':

    # Load sample data from raw file
    sample_path = '../data_raw/telecom_customer_churn.csv'

    if os.path.exists(sample_path):
        df_raw = pd.read_csv(sample_path)

        # Use 5 sample customers
        df_sample = df_raw.head(5).copy()

        results = score(df_sample)

        print("\nScoring Results:")
        print(results[['Customer ID', 'Churn_Probability',
                        'Churn_Prediction', 'Churn_Prediction_Label']])
    else:
        print(f"Sample file not found at {sample_path}")
        print("Please provide a valid path to customer data.")