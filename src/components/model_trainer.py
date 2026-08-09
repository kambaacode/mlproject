import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evalaute_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_model_trainer(self, trainArray, testArray, preprocessor_path):
        try:
            logging.info("fetching training and test input data")

            x_train,x_test,y_train,y_test = (
                trainArray[:,:-1], # x_train
                testArray[:,:-1], # x_test
                trainArray[:,-1], # y_train
                testArray[:,-1], # y_test
                )
        
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor()
            }

            model_report:dict = evalaute_models(x_train, y_train, x_test, y_test, models)
            self.print_models_report(model_report)

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score < 0.7:
                raise CustomException("Best model score is less than 70%", sys)
            
            
            save_object(
                self.model_trainer.train_model_file_path,
                obj= best_model
            )

            logging.info("Best model saved successfully")

            predicted = best_model.predict(x_test)
            score = r2_score(y_test, predicted)
            
            return score
            
        except Exception as e:
            logging.info("Exception occured while initiating model trainer")
            raise CustomException(e, sys)


    def print_models_report(self, models_report):
        try:
            for name,score in models_report.items():
                print(name, " test accuraccy score: " , score)
        except Exception as e:
            logging.info("Exception occured while printing models report")
            raise CustomException(e, sys)