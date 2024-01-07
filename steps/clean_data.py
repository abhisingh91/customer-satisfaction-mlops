import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessingStrategy

from zenml import step
from typing import Annotated, Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Clean the data as per the needs

    Args:
        df: Raw DataFrame of the data
    Returns:
        X_train: Train features df
        X_test: Test Features df
        y_train: Train lables series
        y_test: Test labels series
    """
    try:
        strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, strategy)
        processed_data = data_cleaning.handle_data()

        strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data cleaned and splitted into train and test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error occurred while cleaning the data")
        raise e

