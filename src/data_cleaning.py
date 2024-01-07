import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling the data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing the data 
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data

        Args:
            data: df to be preprocessed
        """
        try:
            data = data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
                "customer_zip_code_prefix",
                "order_item_id"
            ],
            axis=1)

            # replace the null values in these cols with their median
            num_fill_cols = ['product_weight_g', 'product_height_cm', 'product_length_cm', 'product_width_cm']
            for col in num_fill_cols:
                data[col].fillna(data[col].median(), inplace=True)
            
            # replace the null values as "No review" for simplicity
            data['review_comment_message'].fillna("No review", inplace=True)

            # only consider numeric columns for demonstration purposes
            data = data.select_dtypes(include=[np.number])
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data")
            raise e
        

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing the data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X, y = data.iloc[:, :-1], data.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error while diving the data into training and testing sets")
            raise e
    

class DataCleaning:
    """
    Preprocess data and provide train test split
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data while cleaning")
            raise e