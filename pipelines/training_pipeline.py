from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.config import ModelNameConfig

@pipeline
def train_pipeline(data_path: str):
    df = ingest_df(data_path)

    # clean and split the data into features and label
    X_train, X_test, y_train, y_test = clean_data(df)
    
    # get the model configuration and use it to train the model
    trained_model = train_model(X_train, y_train)

    # evaluate the model
    mse, mae, r2_score = evaluate_model(trained_model, X_test, y_test)
