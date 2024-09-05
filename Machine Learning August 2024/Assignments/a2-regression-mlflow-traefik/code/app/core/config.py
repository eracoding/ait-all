import os

from dotenv import load_dotenv

from pydantic import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI Template"
    API_V1_STR: str = "/api/v1"

    # MODEL_PATH: str = "app/ml_models/rf_carPrice_5_feature_3rd_attempt.pkl"
    MODEL_PATH: str = os.getenv('ML_MODEL_PATH')

    MODEL_PATH2: str = os.getenv('ML_MODEL_PATH2')

    X_scaler_path: str = os.getenv('X_scaler_path') 
    Y_scaler_path: str = os.getenv('Y_scaler_path')
    rf_scaler_path: str = os.getenv('rf_scaler_path')

    class Config:
        case_sensitive = True


settings = Settings()
