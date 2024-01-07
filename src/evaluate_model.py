from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation(ABC):
    """
    Abstract class definition for model evaluation
    """
    @abstractmethod
    def calculate_scores(self, y_true: pd.Series, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    """
    Mean Squared Error Evaluation Strategy
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        try:
            logging.info(f"Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE")
            raise e


class MAE(Evaluation):
    """
    Mean Absolute Error Evaluation Strategy
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        try:
            logging.info(f"Calculating MSE")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info(f"MAE: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error while calculating MAE")
            raise e


class R2(Evaluation):
    """
    R-squared Evaluation Strategy
    """
    def calculate_scores(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        try:
            logging.info(f"Calculating R-Squared")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R-squared: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error while calculating R-Squared")
            raise e




