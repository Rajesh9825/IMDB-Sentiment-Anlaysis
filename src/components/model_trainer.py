import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models,save_object

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test):
        try :
            logging.info("split training and test input data")

            models = {
                "Logistic Regression" : LogisticRegression()
            }
            logging.info("Model Training Started...")
            model_report : dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test=y_test,models=models)
            
            logging.info("Model training completed")
            print(model_report)
            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]

            print(best_model_name)
            
            if best_model_score < 0.6:
                return "No best model found"
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            prec_score = precision_score(y_test,predicted)
            print(f"Recall : {recall_score(y_test,predicted)}")
            print(f"F1 score : {f1_score(y_test,predicted,average='weighted')}")
            print(f"accuracy : {accuracy_score(y_test,predicted)}")
            print(f"Confusion Matrix : {confusion_matrix(y_test,predicted)}")

            return prec_score
        
        except Exception as e:
            raise CustomException(e,sys)