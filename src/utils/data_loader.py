import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """Generic function to load data from a CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def load_heart_disease_data():
    """Load heart disease data and return X, y splits."""
    # Try processed data first, then raw data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_path = os.path.join(project_root, 'data', 'processed', 'heart_disease_clean.csv')
    raw_path = os.path.join(project_root, 'data', 'raw', 'heart_disease.csv')
    
    if os.path.exists(processed_path):
        file_path = processed_path
    elif os.path.exists(raw_path):
        file_path = raw_path
    else:
        raise FileNotFoundError("Heart disease dataset not found in data/processed or data/raw")
    
    data = pd.read_csv(file_path)
    
    # Assuming the target column is named 'target' or similar
    target_cols = ['target', 'Target', 'HeartDisease', 'heart_disease']
    target_col = None
    
    for col in target_cols:
        if col in data.columns:
            target_col = col
            break
    
    if target_col is None:
        # Use the last column as target if no obvious target column found
        target_col = data.columns[-1]
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    return X.values, y.values

def load_cardiovascular_risk_data():
    """Load cardiovascular risk score data for regression analysis."""
    # Try processed data first, then raw data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_path = os.path.join(project_root, 'data', 'processed', 'cardiovascular_risk.csv')
    raw_path = os.path.join(project_root, 'data', 'raw', 'heart_disease.csv')
    
    if os.path.exists(processed_path):
        file_path = processed_path
    elif os.path.exists(raw_path):
        file_path = raw_path
        # Generate cardiovascular risk scores from heart disease data
        data = pd.read_csv(file_path)
        # Create continuous risk scores (0-100) based on heart disease features
        risk_scores = create_cardiovascular_risk_scores(data)
        data['risk_score'] = risk_scores
        target_col = 'risk_score'
    else:
        raise FileNotFoundError("Heart disease dataset not found in data/processed or data/raw")
    
    if 'risk_score' not in data.columns:
        data = pd.read_csv(file_path)
        risk_scores = create_cardiovascular_risk_scores(data)
        data['risk_score'] = risk_scores
        target_col = 'risk_score'
    else:
        target_col = 'risk_score'
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    return X.values, y.values

def create_cardiovascular_risk_scores(data):
    """Create continuous cardiovascular risk scores from heart disease features."""
    # Normalize features and create weighted risk score
    import numpy as np
    
    # Clinical weights based on medical literature
    weights = {
        'age': 0.15,
        'sex': 0.20,  # Male = higher risk
        'cp': 0.25,   # Chest pain type
        'trestbps': 0.15,  # Blood pressure
        'chol': 0.10,      # Cholesterol
        'exang': 0.15,     # Exercise angina
        'oldpeak': 0.20,   # ST depression
        'ca': 0.25,        # Vessel blockage
        'thal': 0.30       # Thalassemia
    }
    
    risk_score = np.zeros(len(data))
    
    # Age risk (normalized to 0-1)
    if 'age' in data.columns:
        age_norm = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())
        risk_score += weights.get('age', 0.15) * age_norm
    
    # Gender risk
    if 'sex' in data.columns:
        risk_score += weights.get('sex', 0.20) * data['sex']
    
    # Chest pain (asymptomatic = highest risk)
    if 'cp' in data.columns:
        cp_risk = np.where(data['cp'] == 0, 1.0, 0.5)  # Asymptomatic = high risk
        risk_score += weights.get('cp', 0.25) * cp_risk
    
    # Blood pressure risk
    if 'trestbps' in data.columns:
        bp_risk = np.where(data['trestbps'] > 140, 1.0, data['trestbps'] / 140)
        risk_score += weights.get('trestbps', 0.15) * bp_risk
    
    # Cholesterol risk
    if 'chol' in data.columns:
        chol_risk = np.where(data['chol'] > 240, 1.0, data['chol'] / 240)
        risk_score += weights.get('chol', 0.10) * chol_risk
    
    # Exercise angina
    if 'exang' in data.columns:
        risk_score += weights.get('exang', 0.15) * data['exang']
    
    # ST depression
    if 'oldpeak' in data.columns:
        oldpeak_norm = np.clip(data['oldpeak'] / 6.0, 0, 1)  # Normalize to 0-1
        risk_score += weights.get('oldpeak', 0.20) * oldpeak_norm
    
    # Number of vessels
    if 'ca' in data.columns:
        ca_risk = data['ca'] / 3.0  # Normalize to 0-1
        risk_score += weights.get('ca', 0.25) * ca_risk
    
    # Thalassemia
    if 'thal' in data.columns:
        thal_risk = np.where(data['thal'] == 2, 1.0, 0.3)  # Fixed defect = high risk
        risk_score += weights.get('thal', 0.30) * thal_risk
    
    # Scale to 0-100 range
    risk_score = (risk_score / np.max(risk_score)) * 100
    
    # Add some medical noise for realism
    noise = np.random.normal(0, 5, len(data))
    risk_score = np.clip(risk_score + noise, 0, 100)
    
    return risk_score

def load_medical_data(file_path=None):
    """Load medical dataset for analysis."""
    if file_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(project_root, 'data', 'raw', 'heart_disease.csv')
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The medical dataset file {file_path} does not exist.")

def load_processed_medical_data(file_path=None):
    """Load processed medical dataset for SVM analysis."""
    if file_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(project_root, 'data', 'processed', 'heart_disease_analysis.csv')
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"The processed medical dataset file {file_path} does not exist.")

def load_processed_data(file_path):
    """Load processed data from a specified file path."""
    return load_data(file_path)

def save_processed_data(data, file_path):
    """Save processed data to a specified file path."""
    data.to_csv(file_path, index=False)