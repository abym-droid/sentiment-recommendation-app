## Problem Statement
The e-commerce business is quite popular today. Here product vendors do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.

A part of the Capstone project, I'm expected to be pretending to work as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.

With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.

As a senior ML Engineer, I'm asked to build a model that will improve the recommendations given to the users given their past reviews and ratings.

In order to do this, I'm planning to build a sentiment-based product recommendation system that will recommend top 5 products for a given user
High level steps that need to be performed

- Data sourcing and sentiment analysis
- Building a recommendation system
- Improving the recommendations using the sentiment analysis model
- Deploying the end-to-end project with a user interface
In order to proceed towards the implementation, I need to perfom below steps

- Importing necessary libraries
- Perform Data Loading
- Perform Exploratory Data Analysis
- Data Cleaning
- Text Preprocessing
- Feature Extraction
- Training a Text Classification Model
  - Build at least 3 models from Logistic Regression, Random Forest, XGBoost and Naive Bayes and pick one based on evaluation metrics
- Build a Recommendation system and select one of them based on best relevance and performance
  - User Based Recommendation
  - Item Based Recommendation
- Use Sentiment based model to improve Recommendation system

## Live Demo

**Heroku Deployment:** [https://senti-product-reco-system-97abd5400519.herokuapp.com](https://senti-product-reco-system-97abd5400519.herokuapp.com)

## Features

- **Sentiment Analysis**: Uses a Random Forest classifier to predict sentiment from product reviews (chosen out of 4 models and their hyperparameter tuned iterations)
- **Collaborative Filtering**: Implements user-based recommendation system using cosine similarity
- **Smart Recommendations**: Combines recommendation scores with sentiment analysis for top 5 product suggestions
- **Web Interface**: Flask web application
- **Real-time Processing**: Instant recommendations based on user input

## Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **NLP**: NLTK, spaCy, TF-IDF Vectorization
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Heroku
- **Model Storage**: Compressed pickle files

## Project Structure

```
sentiment-recommendation-app/
├── app.py                          # Main Flask application
├── model.py                        # ML model wrapper class
├── decompress_models.py            # Model decompression utility
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment configuration
├── runtime.txt                     # Python version specification
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── templates/
│   └── index.html                  # Main web interface
├── models/                         # ML models (compressed)
│   ├── best_sentiment_rf_model.pkl.gz
│   ├── tfidf_vectorizer.pkl.gz
│   └── user_final_rating.pkl.gz
└── data/                          # Preprocessed data (compressed)
    └── df_nlp_prep.pkl.gz
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/sentiment-recommendation-app.git
   cd sentiment-recommendation-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Install spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:5001`

8. **Recommendation Process**
1. Get top 20 products from user-based collaborative filtering
2. Extract reviews for these products
3. Predict sentiment for each review using the trained model
4. Calculate positive sentiment percentage for each product
5. Return top 5 products ranked by positive sentiment

### Web Interface
1. Visit the web application
2. Select a username from the dropdown
3. Click "Get Recommendations"
4. View top 5 recommended products with sentiment scores

## Acknowledgments

- UpGrad for the capstone project framework and learning modules
- scikit-learn community for excellent ML libraries
- Flask team for the lightweight web framework
- Heroku for free hosting platform