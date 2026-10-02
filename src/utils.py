import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split




from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, x_test, y_test, models):
    try:
        model_report: dict = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)

            # Predict testing data
            y_test_pred = model.predict(x_test)

            # Get r2 score for test data
            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score

        return model_report

    except Exception as e:
        raise CustomException(e, sys)
    try:
      
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)

            y_train_pred = model.predict(X)
            train_model_score = r2_score(y, y_train_pred)

            report[list(models.keys())[i]] = train_model_score
        for model_name, model in models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            score = r2_score(y, y_pred)
            model_report[model_name] = score

        return model_report

    except Exception as e:
        raise CustomException(e, sys)