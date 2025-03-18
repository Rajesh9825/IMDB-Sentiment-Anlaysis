from flask import Flask, request, render_template, jsonify
import sys
import joblib
import pandas as pd
from src.logger import logging
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
import numpy as np

# Flask Application Setup
application = Flask(__name__)
app = application

# Logger Configuration
logging.info("Application started successfully...")

# -------------------- Utility Functions --------------------

def predictfunc(review):
    """
    Predict the sentiment of a given review using the prediction pipeline.
    """
    try:
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(review)
        print(prediction)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative' if prediction[0] == 0 else 'Wrong'
        print(sentiment)
        return prediction[0], sentiment
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise CustomException(e, sys)

# -------------------- Route Definitions --------------------

@app.route('/')
def index():
    """
    Render the homepage.
    """
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    try:
        if request.method == 'POST':
            content = request.form.get('review')
            
            if not content:
                logging.info("Empty review content received.")
                return render_template("predict.html", pred="N/A", sent="Please provide a review.")

            custom_data = CustomData()
            review = custom_data.get_data_as_data_frame(content)
            print(review)
            # Logging the processed data for better visibility
            logging.info(f"Processed review data: {review}")

            prediction, sentiment = predictfunc(review)

            return render_template("predict.html", pred=prediction, sent=sentiment)

    except Exception as e:
        raise CustomException(e,sys)
        #return render_template("predict.html", pred="N/A", sent="Error in prediction")

# -------------------- Main Execution --------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
