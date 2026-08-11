import sys
import pandas as pd 
import numpy as np
from src.utils import load_object
from src.exception import CustomException
from pydantic import BaseModel
import os
from src.logger import logging

class StudentInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int 


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts","model.pkl")
        self.preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
        
        except Exception as e:
            logging.info("Exception occured in prediction pipeline")
            raise CustomException(e, sys)