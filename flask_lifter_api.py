from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Constants for model files
MODEL_DIR = 'models'
SNATCH_MODEL_PATH = os.path.join(MODEL_DIR, 'snatch_model.joblib')
CLEAN_JERK_MODEL_PATH = os.path.join(MODEL_DIR, 'clean_jerk_model.joblib')
SNATCH_SCALER_PATH = os.path.join(MODEL_DIR, 'snatch_scaler.joblib')
CLEAN_JERK_SCALER_PATH = os.path.join(MODEL_DIR, 'clean_jerk_scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
STATS_PATH = os.path.join(MODEL_DIR, 'model_stats.joblib')

def calculate_age(row):
    """Calculate age based on competition date and birthday."""
    comp_date = pd.to_datetime(row['date'])
    birth_date = pd.to_datetime(row['birthday'])
    return (comp_date - birth_date).days / 365.25

def calculate_model_stats(X, y, model, feature_names):
    """Calculate detailed statistics for a model."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Calculate feature importance
    feature_importance = {}
    for idx, feature in enumerate(feature_names):
        feature_importance[feature] = abs(model.coef_[idx])
    
    # Calculate prediction intervals
    residuals = y - y_pred
    std_residuals = np.std(residuals)
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'feature_importance': feature_importance,
        'std_residuals': std_residuals,
        'mean_absolute_error': np.mean(np.abs(residuals)),
        'sample_size': len(y)
    }

def train_models():
    """Train models if they don't exist."""
    # Check if models already exist
    if (os.path.exists(SNATCH_MODEL_PATH) and 
        os.path.exists(CLEAN_JERK_MODEL_PATH) and
        os.path.exists(SNATCH_SCALER_PATH) and
        os.path.exists(CLEAN_JERK_SCALER_PATH) and
        os.path.exists(LABEL_ENCODER_PATH) and
        os.path.exists(STATS_PATH)):
        print("Models already exist. Skipping training.")
        return

    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv('merged_lifters_20250315_212804.csv')
    
    # Calculate age
    df['age'] = df.apply(calculate_age, axis=1)
    
    # Convert numeric columns
    numeric_cols = ['bodyweight', 'snatch', 'clean_jerk']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Encode sex
    label_encoder = LabelEncoder()
    df['sex_encoded'] = label_encoder.fit_transform(df['sex'])
    
    # Save the label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    # Remove rows with missing values
    df = df.dropna(subset=['age', 'bodyweight', 'snatch', 'clean_jerk', 'sex_encoded'])
    
    # Train Clean & Jerk Model
    clean_jerk_features = ['snatch', 'bodyweight', 'age', 'sex_encoded']
    X_clean_jerk = df[clean_jerk_features]
    y_clean_jerk = df['clean_jerk']
    
    # Scale features for clean_jerk model
    clean_jerk_scaler = StandardScaler()
    X_clean_jerk_scaled = clean_jerk_scaler.fit_transform(X_clean_jerk)
    
    # Train clean_jerk model
    clean_jerk_model = LinearRegression()
    clean_jerk_model.fit(X_clean_jerk_scaled, y_clean_jerk)
    
    # Train Snatch Model
    snatch_features = ['clean_jerk', 'bodyweight', 'age', 'sex_encoded']
    X_snatch = df[snatch_features]
    y_snatch = df['snatch']
    
    # Scale features for snatch model
    snatch_scaler = StandardScaler()
    X_snatch_scaled = snatch_scaler.fit_transform(X_snatch)
    
    # Train snatch model
    snatch_model = LinearRegression()
    snatch_model.fit(X_snatch_scaled, y_snatch)
    
    # Calculate and save model statistics
    model_stats = {
        'clean_jerk': calculate_model_stats(X_clean_jerk_scaled, y_clean_jerk, clean_jerk_model, clean_jerk_features),
        'snatch': calculate_model_stats(X_snatch_scaled, y_snatch, snatch_model, snatch_features)
    }
    
    # Save models, scalers and stats
    joblib.dump(clean_jerk_model, CLEAN_JERK_MODEL_PATH)
    joblib.dump(snatch_model, SNATCH_MODEL_PATH)
    joblib.dump(clean_jerk_scaler, CLEAN_JERK_SCALER_PATH)
    joblib.dump(snatch_scaler, SNATCH_SCALER_PATH)
    joblib.dump(model_stats, STATS_PATH)
    
    print("Models trained and saved successfully.")

def load_models():
    """Load trained models and scalers."""
    clean_jerk_model = joblib.load(CLEAN_JERK_MODEL_PATH)
    snatch_model = joblib.load(SNATCH_MODEL_PATH)
    clean_jerk_scaler = joblib.load(CLEAN_JERK_SCALER_PATH)
    snatch_scaler = joblib.load(SNATCH_SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    model_stats = joblib.load(STATS_PATH)
    return clean_jerk_model, snatch_model, clean_jerk_scaler, snatch_scaler, label_encoder, model_stats

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics about the models."""
    try:
        _, _, _, _, _, model_stats = load_models()
        return jsonify({
            'model_statistics': model_stats,
            'interpretation': {
                'r2_score': 'Proportion of variance explained by the model (0-1, higher is better)',
                'rmse': 'Root Mean Square Error (in kg)',
                'feature_importance': 'Absolute importance of each feature (higher values indicate stronger influence)',
                'std_residuals': 'Standard deviation of prediction errors (in kg)',
                'mean_absolute_error': 'Average absolute prediction error (in kg)',
                'sample_size': 'Number of samples used to train the model'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.
    Expects JSON with: bodyweight, age, sex, and either snatch or clean_jerk
    """
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug print
        
        # Validate required fields
        required_fields = ['bodyweight', 'age', 'sex']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f'Missing required fields: {", ".join(missing_fields)}'
            print("Validation error:", error_msg)  # Debug print
            return jsonify({'error': error_msg}), 400
            
        if 'snatch' not in data and 'clean_jerk' not in data:
            error_msg = 'Either snatch or clean_jerk must be provided'
            print("Validation error:", error_msg)  # Debug print
            return jsonify({'error': error_msg}), 400
            
        # Load models
        clean_jerk_model, snatch_model, clean_jerk_scaler, snatch_scaler, label_encoder, model_stats = load_models()
        
        # Encode sex
        try:
            sex_encoded = label_encoder.transform([data['sex']])[0]
        except ValueError as e:
            error_msg = f'Invalid sex value. Must be one of: {list(label_encoder.classes_)}'
            print("Sex encoding error:", error_msg)  # Debug print
            return jsonify({'error': error_msg}), 400
        
        result = {}
        prediction_stats = {}
        
        # Predict Clean & Jerk
        if 'snatch' in data and 'clean_jerk' not in data:
            print("Predicting Clean & Jerk")  # Debug print
            features = pd.DataFrame([[
                data['snatch'],
                data['bodyweight'],
                data['age'],
                sex_encoded
            ]], columns=['snatch', 'bodyweight', 'age', 'sex_encoded'])
            
            features_scaled = clean_jerk_scaler.transform(features)
            clean_jerk_pred = clean_jerk_model.predict(features_scaled)[0]
            result['clean_jerk'] = float(clean_jerk_pred)
            
            # Add confidence interval
            std_residuals = model_stats['clean_jerk']['std_residuals']
            prediction_stats['clean_jerk'] = {
                'prediction_interval': {
                    'lower_bound': float(clean_jerk_pred - 2 * std_residuals),
                    'upper_bound': float(clean_jerk_pred + 2 * std_residuals)
                },
                'model_accuracy': {
                    'r2_score': model_stats['clean_jerk']['r2_score'],
                    'rmse': model_stats['clean_jerk']['rmse']
                }
            }
            
        # Predict Snatch
        elif 'clean_jerk' in data and 'snatch' not in data:
            print("Predicting Snatch")  # Debug print
            features = pd.DataFrame([[
                data['clean_jerk'],
                data['bodyweight'],
                data['age'],
                sex_encoded
            ]], columns=['clean_jerk', 'bodyweight', 'age', 'sex_encoded'])
            
            features_scaled = snatch_scaler.transform(features)
            snatch_pred = snatch_model.predict(features_scaled)[0]
            result['snatch'] = float(snatch_pred)
            
            # Add confidence interval
            std_residuals = model_stats['snatch']['std_residuals']
            prediction_stats['snatch'] = {
                'prediction_interval': {
                    'lower_bound': float(snatch_pred - 2 * std_residuals),
                    'upper_bound': float(snatch_pred + 2 * std_residuals)
                },
                'model_accuracy': {
                    'r2_score': model_stats['snatch']['r2_score'],
                    'rmse': model_stats['snatch']['rmse']
                }
            }
            
        print("Prediction successful:", result)  # Debug print
        return jsonify({
            'predictions': result,
            'prediction_statistics': prediction_stats,
            'input_data': data
        })
        
    except Exception as e:
        print("Error in prediction:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """Serve the main page."""
    return render_template('index.html')

if __name__ == '__main__':
    # Train models if they don't exist
    train_models()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False for security in production 