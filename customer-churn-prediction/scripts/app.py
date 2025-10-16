from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders
try:
    model = joblib.load('../models/churn_model.pkl')  # Added ../
    encoders = joblib.load('../models/encoders.pkl')  # Added ../
    feature_columns = joblib.load('../models/feature_columns.pkl')  # Added ../
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first!")
    exit()

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        # Get data from request
        data = request.json
        
        # Create DataFrame
        input_data = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in encoders.items():
            if col in input_data.columns:
                # Handle unseen labels
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                input_data[col] = encoder.transform(input_data[col])
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # or appropriate default value
        
        # Reorder columns to match training
        input_data = input_data[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'churn_risk': 'High' if probability > 0.5 else 'Low'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)