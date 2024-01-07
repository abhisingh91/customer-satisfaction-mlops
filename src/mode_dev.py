import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
import pandas as pd


class Model(ABC):
    """
    Abstract Class for all Models
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass


class LinearRegressionModel(Model):
    """
    Creates a linear regression model
    """
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Trains the model
        """
        try:
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            logging.info(f"Model training completed!")
            return regressor
        except Exception as e:
            logging.error(f"Error while creating the model")
            raise e