import re
import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = stopwords.words('english')
new_stopwords = ['mario','la','blah','saturday','monday','sunday','morning','evening','friday','would','shall','could','might']
stop_words.extend(new_stopwords)
stop_words.remove('not')
stop_words = set(stop_words)

# Removing Special Character
def remove_special_character(content):
    return re.sub('\[[^&@#!]]*\]',' ', content) # re.sub('\[[^&@#!]]*\]', '', content)

# Removing URL's
def remove_url(content):
    return re.sub(r'http\S+',' ',content)

# Removing the stopwords from text
def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)


# Expansion of english contractions
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    return content
 

# Stemming
def lemmatization(content):
    wordnetlemma = WordNetLemmatizer()
    content = [wordnetlemma.lemmatize(word,) for word in word_tokenize(content)]
    return " ".join(content)
    
# Data Preprocessing

def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    content = remove_stopwords(content)
    content = lemmatization(content)
    return content



@dataclass
class DataPreprocessingConfig:
    cleaned_data_file_path = os.path.join('artifacts','cleaned_data.csv')
    preprocessed_data_file_path = os.path.join('artifacts','preprocessed_data.csv')


class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config = DataPreprocessingConfig()

    def Initiate_preprocessing(self,data_path):
        try:
            df = pd.read_csv(data_path)
            logging.info("Data Cleaning Initiated")
            df['Reviews'] = df['Reviews'].apply(data_cleaning)
            logging.info("Cleaning done")

            df.to_csv(self.data_preprocessing_config.cleaned_data_file_path,index=False,header=True)

            return self.data_preprocessing_config.cleaned_data_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
    


# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wordnetlemma = WordNetLemmatizer()
#         self.data_preprocessing_config = DataPreprocessingConfig()

#     def Lemmatization(self,cleaned_data):
#         try:
#             df = pd.read_csv(cleaned_data)
#             df['Reviews'] = [self.wordnetlemma.lemmatize(word) for word in word_tokenize(df['Reviews'])]
#             df.to_csv(self.data_preprocessing_config.cleaned_data_file_path,index=False,header=True)
#             return df
#         except Exception as e:
#             raise CustomException(e,sys)





