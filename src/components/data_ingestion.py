import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_preprocessing import DataPreprocessingConfig,DataPreprocessing
from src.components.feature_extraction import TextVectorizer
from src.components.model_trainer import ModelTrainerConfig,ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_Ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df = pd.read_csv('data/IMDB-Dataset.csv')
            df['Label'] = df['Ratings'].apply(lambda x:1 if x >= 7 else (0 if x <= 4 else 2))
            df= df[df['Label']<2]
            data = df[['Reviews','Label']]

            print(df.isnull().sum())
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            # train_set, test_set = train_test_split(data,test_size=0.2,random_state=42)

            #train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            #test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.raw_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys) 



if __name__=="__main__":
    obj = DataIngestion()
    data = obj.initiate_data_Ingestion()
    data_preprocessing = DataPreprocessing()
    cleaned_data = data_preprocessing.Initiate_preprocessing(data)
    vectorizer = TextVectorizer()
    X_train,X_test,y_train,y_test = vectorizer.initiate_vectorization(cleaned_data)
    model_train = ModelTrainer()
    print(model_train.initiate_model_trainer(X_train,X_test,y_train,y_test))



