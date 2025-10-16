import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from notebooks.generate_sample_data import generate_sample_data


def prepare_data(df):
    """Prepare data for training"""
    # Create a copy
    data = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['contract_type', 'payment_method']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le
    
    # Select features (excluding customer_id)
    feature_columns = [col for col in data.columns if col not in ['customer_id', 'churn']]
    
    X = data[feature_columns]
    y = data['churn']
    
    return X, y, encoders, feature_columns

def train_model(X, y):
    """Train a Random Forest model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test, y_test

if __name__ == "__main__":
    # Generate sample data (replace with your data loading)
    df = generate_sample_data(1000)
    
    # Prepare data
    X, y, encoders, feature_columns = prepare_data(df)
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model and encoders
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/churn_model.pkl')
    joblib.dump(encoders, '../models/encoders.pkl')
    joblib.dump(feature_columns, '../models/feature_columns.pkl')
    
    print("\nModel saved successfully!")