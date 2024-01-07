import logging

import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path of the data file
        """
        self.data_path = data_path
    
    def get_data(self):
        """
        returns the data frame after ingesting the data from the data path
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path, index_col=None, parse_dates=True)
    

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting data from the data path

    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data")
        raise e
