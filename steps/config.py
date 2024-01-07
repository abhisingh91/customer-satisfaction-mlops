from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configurations"""
    model: str = "LinearRegression"