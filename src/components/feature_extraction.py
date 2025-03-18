import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class VectorizerConfig:
    vectorized_model_file_path = os.path.join("artifacts","vectorizer.pkl")

class TextVectorizer:
    def __init__(self):
        self.vectorizer_config = VectorizerConfig()
        
    
    def initiate_vectorization(self,cleaned_data):
        try :
            df = pd.read_csv(cleaned_data)
            df.dropna(subset=['Reviews'], inplace=True)
            X_train,X_test,y_train,y_test = train_test_split(df['Reviews'],df['Label'],test_size=0.1,random_state=42)
        
            tfid_vectorizer = TfidfVectorizer(ngram_range=(1,3),min_df=10,max_features=2000)
            print(X_train.isnull().sum())
            logging.info("Vectorization Started...")
            X_train = tfid_vectorizer.fit_transform(X_train).toarray()
            X_test = tfid_vectorizer.transform(X_test).toarray()
            logging.info("Vectorization done")  
            
            # coverting Dataframe to array
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            save_object(
                    file_path=self.vectorizer_config.vectorized_model_file_path,
                    obj=tfid_vectorizer
                )
            logging.info('Vectorizer pickle file saved')


            return X_train,X_test,y_train,y_test

            
        except Exception as e:
            raise CustomException(e,sys)


