# IMDb Movie Reviews Sentiment Analysis

## Problem Statement
Sentiment analysis of movie reviews is a key application of natural language processing (NLP) that helps in understanding audience opinions. The goal of this project is to classify IMDb movie reviews as either positive or negative using machine learning techniques. This analysis can assist businesses, filmmakers, and marketers in gauging audience sentiment, improving recommendations, and enhancing user engagement.

## Expected Outcomes

* Develop a robust machine learning model to classify IMDb reviews as positive or negative.

* Gain insights into the common words and patterns associated with each sentiment.

* Evaluate different machine learning models and compare their performance.

* Deploy a sentiment analysis model for real-world applications.


## Project Structure
```
IMDB_Sentiment_Analysis/
│-- artifacts/                 # Dataset and preprocessed data
|   |-- cleaned_data.csv
|   |-- data.csv
|   |-- model.pkl               # Trained models and saved versions 
|   |-- vectorizer.pkl
|--data/
|   |--IMDB-Dataset.csv
│-- notebooks/              # Jupyter notebooks for analysis and experimentation               
|-- src                     # Source code for data processing and model training
|   |-- components
|   |   |-- data_ingestion.py
|   |   |-- data_preprocessing.py
|   |   |-- feature_extraction.py
|   |   |-- model_trainer.py
|   |-- pipeline
|   |   |-- prediction_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   |--utils.py
<!-- │-- static/          # CSS, JS, and other static files for web app -->
│-- templates/            # HTML templates for Flask app
|   |-- home.html
|   |-- predict.html
|-- .gitignore
│-- application.py        # Flask web application
|-- Procfile
│-- README.md             # Project documentation
│-- requirements.txt      # Dependencies
|-- setup.py
```

## Project Workflow
### 1. Data Collection
* Dataset: IMDB Dataset of Movies Reviews

* Source: Downloaded from Kaggle( https://www.kaggle.com/datasets/crisbam/imdb-dataset-of-65k-movie-reviews-and-translation )

### 2. Data Preprocessing
* Removed HTML tags and special characters.

* Converted text to lowercase.

* Tokenization and stopword removal.

* Applied stemming and lemmatization.

* Vectorized text using TF-IDF and Bag of Words (BoW).

### 3. Exploratory Data Analysis (EDA)
* Word cloud visualization for positive and negative reviews.

* Distribution of word counts in reviews.

* Most frequently occurring words in positive and negative reviews.

### 4. Model Training

* Implemented Logistic Regression, Naive Bayes, and Random Forest.

* Hyperparameter tuning for performance optimization.

* Evaluated models using accuracy, precision, recall, and F1-score.

### 5. Model Evaluation

* Best-performing model: Logistic Regression (Accuracy: 88%).

* Confusion matrix analysis for false positives and false negatives.

* ROC curve and AUC score comparison.

### 6. Deployment

* Developed a Flask web application for real-time review classification.

* Initially deployed the model on an AWS EC2 instance using SSH for connection.

* Due to cost concerns, migrated the deployment to Render (cost-free hosting).

* Users can input a review and get sentiment predictions instantly through a Flask web application.

live demo : https://imdb-reviews-sentiment-analysis-q27f.onrender.com

## Results

* Achieved an 88% accuracy using Logistic Regression with TF-IDF features.

* Word clouds showed that words like "great," "amazing," and "best" were frequent in positive reviews, while words like "worst," "boring," and "waste" appeared in negative reviews.

* The model generalizes well on unseen IMDb reviews with minimal overfitting.

## Future Enhancements

* Implement deep learning models like LSTM or BERT for improved accuracy.

* Incorporate sentiment intensity scoring instead of binary classification.

* Deploy on a cloud-based API for wider accessibility.
