import os
import sys
from dataclasses import dataclass
from importlib import import_module

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor#

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing inputdata")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "linear_regression": LinearRegression(),
                "decision_tree": DecisionTreeRegressor(random_state=42),
                "random_forest": RandomForestRegressor(random_state=42),
                "ada_boost": AdaBoostRegressor(random_state=42),
                "gradient_boosting": GradientBoostingRegressor(random_state=42),
                "k_neighbors": KNeighborsRegressor(),
            }
            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            # to get the best model name from dict
            best_model_name = list(model_report.keys())[list(
                model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with score greater than 0.6", sys)

            logging.info(
                "Best found model on both training and testing dataset")

            preprocessing_obj = DataTransformation().get_data_transformer_object()

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predictions = best_model.predict(x_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square

        except Exception as error:
            raise CustomException(error, sys) from error
