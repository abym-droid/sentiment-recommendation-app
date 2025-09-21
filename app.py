from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Decompress models if needed (for Heroku)
try:
    print("Checking if decompression is needed...")
    import decompress_models
    decompress_models.decompress_models()
    print("Decompression completed.")
except Exception as e:
    print(f"Decompression not needed or failed: {e}")

# Import model AFTER decompression
from model import SentimentRecommendationModel

# Initialize the model
print("Initializing the recommendation model...")
model = SentimentRecommendationModel()
print("Model initialized successfully!")

@app.route('/')
def index():
    """Render the main page with user input form"""
    try:
        # Get list of available users for dropdown
        available_users = model.get_available_users()
        return render_template('index.html', users=available_users)
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/recommend', methods=['POST'])
def recommend():
    """Process recommendation request"""
    try:
        username = request.form.get('username')
        
        if not username:
            return jsonify({'error': 'Please provide a username'})
        
        # Get recommendations
        recommendations = model.get_top5_recommendations(username)
        
        if recommendations is None:
            return jsonify({'error': f'User {username} not found or no recommendations available'})
        
        # Convert DataFrame to dict for JSON response
        recommendations_dict = recommendations.to_dict('records')
        
        return jsonify({
            'success': True,
            'username': username,
            'recommendations': recommendations_dict
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)