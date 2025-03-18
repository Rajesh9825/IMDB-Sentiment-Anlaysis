import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,data):
        try:
            model_path = 'artifacts/model.pkl'
            vectorizer_path ='artifacts/vectorizer.pkl'
            model = load_object(file_path = model_path)
            vectorizer = load_object(file_path=vectorizer_path)
            vectorized_data = vectorizer.transform(data)
            preds = model.predict(vectorized_data)
            print(preds)
            pred_proba = model.predict_proba(vectorized_data)
            print(pred_proba)
            return preds
    
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self):
        pass

    def get_data_as_data_frame(self,Reviews):
        try:
            custom_data_input_dict = {
                "Reviews" : [Reviews]
            }

            df = pd.DataFrame(custom_data_input_dict)
            df.reset_index(drop=True, inplace=True)
            return df['Reviews']

        except Exception as e:
            raise CustomException(e,sys)

