import logging
import pandas as pd

from zenml import step
from zenml.client import Client
import mlflow

from src.evaluate_model import MSE, MAE, R2
from sklearn.base import RegressorMixin
from typing import Tuple, Annotated

expreriment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=expreriment_tracker.name)
def evaluate_model(
    model: RegressorMixin, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "mse"], 
    Annotated[float, "mae"],
    Annotated[float, "r2_score"]
]:
    """
    Performs model evaluation based on the given metrics
    """
    try:
        y_pred = model.predict(X_test)
        mse_class, mae_class, r2_class = MSE(), MAE(), R2()
        mse = mse_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        mae = mae_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mae", mae)

        r2_score = r2_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("r2_score", r2_score)


        return mse, mae, r2_score
    except Exception as e:
        logging.error(f"Error while evaluating the model")
        raise e

