# Customer Churn Prediction System

A machine learning system that predicts customer churn probability using behavioral and demographic data. This system helps businesses identify at-risk customers and take proactive retention measures.

## 🚀 Features

- **Predictive Analytics**: ML model to forecast customer churn probability
- **REST API**: Flask-based API for real-time predictions
- **Feature Engineering**: Advanced customer behavior features
- **Model Management**: Version control and experiment tracking
- **Monitoring**: Prediction logging and performance monitoring

## 📁 Project Structure

```
customer-churn-prediction/
├── models/                  # Saved models and artifacts
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Python scripts for training and serving
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Tech Stack

- **ML Framework**: Scikit-learn, XGBoost
- **Backend**: Flask, Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

## ⚡ Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv churn_env
   source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🏃‍♂️ Running the System

#### Step 1: Prepare Data
```bash
cd scripts
python process_data.py
```
This generates sample data and creates processed datasets for training.

#### Step 2: Train Model
```bash
python train_model.py
```
Trains the churn prediction model and saves it to `models/` directory.

#### Step 3: Start Prediction API
```bash
python app.py
```
Starts the Flask server on `http://localhost:5000`

### 📊 Making Predictions

Use the API to get churn predictions:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5,
    "monthly_charges": 75.0,
    "total_charges": 375.0,
    "contract_type": "Month-to-month",
    "paperless_billing": 1,
    "payment_method": "Electronic check",
    "monthly_usage_gb": 350.5,
    "support_calls": 3,
    "account_age_days": 150
  }'
```

**Example Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.72,
  "churn_risk": "High"
}
```

## 🔧 API Endpoints

### `POST /predict`
Get churn prediction for a customer

**Input:**
```json
{
  "tenure": 12,
  "monthly_charges": 65.5,
  "total_charges": 786.0,
  "contract_type": "Month-to-month",
  "paperless_billing": 1,
  "payment_method": "Electronic check", 
  "monthly_usage_gb": 250.0,
  "support_calls": 2,
  "account_age_days": 365
}
```

### `GET /health`
Check API status
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /features`
Get expected feature columns
```json
{
  "feature_columns": ["tenure", "monthly_charges", ...],
  "categorical_encoders": ["contract_type", "payment_method"]
}
```

## 📈 Model Features

The model uses these customer attributes:

### Demographic Features
- `tenure`: Months with the company
- `account_age_days`: Total account age in days
- `contract_type`: Contract duration (Month-to-month, One year, Two year)

### Financial Features
- `monthly_charges`: Monthly service cost
- `total_charges`: Total amount charged
- `payment_method`: Payment type

### Behavioral Features
- `monthly_usage_gb`: Data usage per month
- `support_calls`: Number of support calls
- `paperless_billing`: Paperless billing preference

### Engineered Features
- `tenure_to_age_ratio`: Engagement intensity
- `cost_per_gb`: Value for money indicator
- `calls_per_month`: Support frequency

## 🎯 Risk Categories

- **Low Risk**: < 30% churn probability
- **Medium Risk**: 30-70% churn probability  
- **High Risk**: > 70% churn probability

## 🔄 Retraining

To retrain the model with new data:

1. Place new data in `data/raw/`
2. Run data processing:
   ```bash
   python process_data.py
   ```
3. Retrain model:
   ```bash
   python train_model.py
   ```
4. Restart the API

## 📊 Monitoring

The system logs all predictions to `predictions_log.csv` with timestamps for monitoring and analysis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the example notebooks

## 🗺️ Roadmap

- [ ] Database integration for real-time data
- [ ] Model performance dashboard
- [ ] Automated retraining pipeline
- [ ] A/B testing framework
- [ ] Customer segmentation features

---

**Note**: This is a prototype system. For production use, consider adding:
- Authentication and authorization
- Rate limiting
- Comprehensive error handling
- Database persistence
- Containerization with Docker