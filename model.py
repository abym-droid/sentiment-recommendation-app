import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import warnings
import os
import gzip
warnings.filterwarnings('ignore')

class SentimentRecommendationModel:
    def __init__(self):
        """Initialize the model by loading pickled models and data"""
        self.load_models()
        self.load_data()
    
    def load_compressed_pickle(self, filepath):
        """Load a compressed pickle file"""
        if os.path.exists(filepath + '.gz'):
            # Load compressed version
            with gzip.open(filepath + '.gz', 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(filepath):
            # Load uncompressed version
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Neither {filepath} nor {filepath}.gz found")
    
    def load_models(self):
        """Load the trained models"""
        try:
            # Load sentiment analysis model (Random Forest)
            self.rf_model = self.load_compressed_pickle('models/best_sentiment_rf_model.pkl')
            
            # Load TF-IDF vectorizer
            self.tfidf = self.load_compressed_pickle('models/tfidf_vectorizer.pkl')
            
            # Load user recommendation matrix
            self.user_final_rating = self.load_compressed_pickle('models/user_final_rating.pkl')
                
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_data(self):
        """Load the preprocessed data"""
        try:
            # Load preprocessed NLP data
            self.df_nlp_prep = self.load_compressed_pickle('data/df_nlp_prep.pkl')
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_available_users(self):
        """Get list of available users"""
        return list(self.user_final_rating.index)
    
    def get_top5_recommendations(self, user_name):
        """
        Get top 5 product recommendations for a user based on sentiment analysis
        
        Args:
            user_name (str): Username to get recommendations for
            
        Returns:
            pandas.DataFrame: Top 5 recommended products with sentiment scores
        """
        try:
            # Check if user exists
            if user_name not in self.user_final_rating.index:
                print(f"The User {user_name} does not exist. Please provide a valid user name")
                return None
            
            # Get top 20 recommended products from the recommendation model
            top20_recommended_products = list(
                self.user_final_rating.loc[user_name]
                .sort_values(ascending=False)[0:20].index
            )
            
            # Get only the recommended products from the prepared dataframe
            df_top20_products = self.df_nlp_prep[
                self.df_nlp_prep.id.isin(top20_recommended_products)
            ].copy()
            
            # Check if we have any products
            if df_top20_products.empty:
                print(f"No products found for user {user_name}")
                return None
            
            # Transform reviews using TF-IDF vectorizer
            X = self.tfidf.transform(df_top20_products["reviews_lemmatized"].values.astype(str))
            
            # Predict sentiment using Random Forest model
            df_top20_products['predicted_sentiment'] = self.rf_model.predict(X)
            
            # Create binary positive sentiment column
            df_top20_products['positive_sentiment'] = df_top20_products['predicted_sentiment'].apply(
                lambda x: 1 if x == "Positive" else 0
            )
            
            # Group by product name and calculate sentiment metrics
            pred_df = df_top20_products.groupby(by='name')['positive_sentiment'].sum().to_frame()
            pred_df.columns = ['positive_sentiment_count']
            
            # Calculate total sentiment count
            pred_df['total_sentiment_count'] = df_top20_products.groupby(by='name')['predicted_sentiment'].count()
            
            # Calculate positive sentiment percentage
            pred_df['positive_sentiment_percentage'] = np.round(
                pred_df['positive_sentiment_count'] / pred_df['total_sentiment_count'] * 100, 2
            )
            
            # Return top 5 products sorted by positive sentiment percentage
            result = pred_df.sort_values(by='positive_sentiment_percentage', ascending=False)[:5]
            
            # Reset index to include product name as a column
            result = result.reset_index()
            
            return result
            
        except Exception as e:
            print(f"Error in get_top5_recommendations: {e}")
            return None
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Predicted sentiment (Positive/Negative)
        """
        try:
            # Transform text using TF-IDF
            X = self.tfidf.transform([text])
            
            # Predict sentiment
            sentiment = self.rf_model.predict(X)[0]
            
            return sentiment
            
        except Exception as e:
            print(f"Error in predict_sentiment: {e}")
            return "Unknown"