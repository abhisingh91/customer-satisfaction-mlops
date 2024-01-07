import logging
import pandas as pd

from zenml import step
from zenml.client import Client
import mlflow

from src.mode_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

expreriment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=expreriment_tracker.name)
def train_model(X_train: pd.DataFrame, y_train: pd.Series, config: ModelNameConfig) -> RegressorMixin:
    """
    Train a model on the training data
    """
    try:
        model = None
        if config.model == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model} not supported")
    except Exception as e:
        logging.error(f"Error in training the model")
        raise e
    
