from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
models = {}
weights = {}

def load_all_models():
    """Load all trained models"""
    global models, weights
    model_dir = Path("models")
    
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            category = model_file.stem.replace("_model", "")
            try:
                with open(model_file, 'rb') as f:
                    models[category] = pickle.load(f)
                print(f"✅ Loaded model: {category}")
            except Exception as e:
                print(f"❌ Error loading {category}: {e}")
    
    # Load weights
    weights_file = Path("optimized_weights.pkl")
    if weights_file.exists():
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
        print(f"✅ Loaded feature weights")

# Load models at startup
load_all_models()

def convert_value_to_rating(value):
    """Convert predicted value to 1-10 rating scale"""
    if value < 100:
        return 1
    elif value < 500:
        return 2
    elif value < 1000:
        return 3
    elif value < 2500:
        return 4
    elif value < 5000:
        return 5
    elif value < 10000:
        return 6
    elif value < 25000:
        return 7
    elif value < 50000:
        return 8
    elif value < 100000:
        return 9
    else:
        return 10

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Collectivault Rating System API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/categories': 'List available categories',
            '/predict': 'Make prediction (POST)',
            '/batch_predict': 'Batch predictions (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/categories', methods=['GET'])
def get_categories():
    """Get list of available categories"""
    return jsonify({
        'categories': list(models.keys()),
        'count': len(models)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict rating for a single item
    
    Expected JSON format:
    {
        "category": "cricket_bats",
        "features": {
            "age": 10,
            "condition": 8,
            "rarity": 7,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        category = data.get('category')
        features = data.get('features')
        
        if not category or not features:
            return jsonify({'error': 'Missing category or features'}), 400
        
        if category not in models:
            return jsonify({'error': f'Model not available for category: {category}'}), 404
        
        # Prepare features
        feature_df = pd.DataFrame([features])
        
        # Apply feature engineering
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (feature_df[col] > 0).all():
                feature_df[f'{col}_log'] = np.log1p(feature_df[col])
        
        # Make prediction
        model = models[category]
        predicted_value = model.predict(feature_df)[0]
        rating = convert_value_to_rating(predicted_value)
        
        # Return results
        return jsonify({
            'success': True,
            'category': category,
            'rating': int(rating),
            'estimated_value': float(predicted_value),
            'currency': 'INR',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict ratings for multiple items
    
    Expected JSON format:
    {
        "items":
