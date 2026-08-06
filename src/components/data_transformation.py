import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer # ColumnTransformer help us doing different preprocessing steps to different columns of our dataset
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path:str = os.path.join("artifact","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns =  ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps= [
                    ("Imputer", SimpleImputer(strategy= "median")),
                    ("Scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy=("most_frequent"))),
                    ("one hot encoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical data encoding completed")
            logging.info("Numerical data standard scaling completed")


            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.info("Exception occurred while getting data transformation object")
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the test and train dataframes")
            logging.info("Obtaining the preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column = "math_score"
            numerical_columns = ["reading_score", "writing_score"]
            
            input_feature_train_df = train_df.drop(columns = [target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = [target_column])
            target_feature_test_df = test_df[target_column]
            logging.info("Applying transformation on train and test datasets")

            inpute_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            inpute_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_array = np.c_[input_feature_train_df, np.array(input_feature_train_df)]
            test_array = np.c_[input_feature_test_df, np.array(input_feature_test_df)]

            logging.info("Saved preprocessing object")

            # we need to save the pkl file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logging.info("Error occured while initiating data transformation")
            raise CustomException(e, sys)