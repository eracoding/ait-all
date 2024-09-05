import joblib
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from app.services.custom import PolynomialRegression

class BaseMLModel(ABC):
    @abstractmethod
    def predict(self, req: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return super().__str__()


class MLModel(BaseMLModel):
    """Sample ML model"""

    def __init__(self, model_path: str, model_path2: str, X_scaler_path: str, Y_scaler_path: str, rf_scaler_path: str) -> None:
        self.rf_model = joblib.load(model_path)

        # Polynomial params
        lr = 0.0001
        momentum = None
        method = 'sgd'
        is_xavier = True
        params = {"method": method, "lr": lr, "momentum": momentum, "isXavier": is_xavier, "l": 0.1}

        self.pl_model = PolynomialRegression(**params)

        self.pl_model.load(model_path2)

        self.scaler = [joblib.load(rf_scaler_path), joblib.load(X_scaler_path), joblib.load(Y_scaler_path)]

        # def __init__(self, model_path: str, X_scaler_path: str, Y_scaler_path: str) -> None:
        # self.model = joblib.load(model_path)
        # self.scalerX = joblib.load(X_scaler_path)
        # self.scalerY = joblib.load(Y_scaler_path)

    def predict(self, df: pd.DataFrame, model_type: str) -> float:
        if model_type == 'random_forest':
            preprocessed = self.scaler[0].transform(df)
            prediction = int(np.exp(self.rf_model.predict(preprocessed)))
        else:
            preprocessed = self.scaler[1].transform(df)
            
            prediction = int(self.scaler[2].inverse_transform(np.array(self.pl_model.predict(preprocessed, to_transform=True).reshape(-1, 1))).flatten())
        
        return prediction if prediction > 0 else -prediction % 100000


    def __str__(self) -> str:
        return super().__str__()


