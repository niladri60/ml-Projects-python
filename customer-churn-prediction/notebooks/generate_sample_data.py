import numpy as np
import pandas as pd

def generate_sample_data(n_customers=1000):
    np.random.seed(42)
    data = {
        'customer_id': range(n_customers),
        'tenure': np.random.randint(1, 72, n_customers),
        'monthly_charges': np.random.uniform(20, 100, n_customers),
        'total_charges': np.random.uniform(50, 5000, n_customers),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'paperless_billing': np.random.choice([0, 1], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
        'monthly_usage_gb': np.random.uniform(50, 500, n_customers),
        'support_calls': np.random.randint(0, 10, n_customers),
        'account_age_days': np.random.randint(30, 365*3, n_customers)
    }

    df = pd.DataFrame(data)

    churn_prob = (
        0.1 +
        0.3 * (df['tenure'] < 12) +
        0.2 * (df['monthly_charges'] > 70) +
        0.15 * (df['support_calls'] > 5) +
        0.1 * (df['contract_type'] == 'Month-to-month') -
        0.1 * (df['tenure'] > 24)
    )

    df['churn'] = (np.random.random(n_customers) < churn_prob).astype(int)
    return df
