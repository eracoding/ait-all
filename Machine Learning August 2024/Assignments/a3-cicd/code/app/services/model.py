import joblib
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from app.services.custom import PolynomialRegression, LogisticRegression
from app.util import load_model, register_model_to_production

class BaseMLModel(ABC):
    @abstractmethod
    def predict(self, req: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return super().__str__()


class MLModel(BaseMLModel):
    """Sample ML model"""

    def __init__(self, model_path: str, model_path2: str, model_path3: str, X_scaler_path: str, Y_scaler_path: str, rf_scaler_path: str) -> None:
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

        # Logistic Regression

        # self.class_model, flag = load_model()
        # if flag:
        #     register_model_to_production()
        
        params = {"method": 'minibatch', 'k': 4, 'n': 5, 'alpha': 0.001}
        self.class_model = LogisticRegression(**params)
        self.class_model.load(model_path3) # Model weights if custom class cannot be deployed

    def predict(self, df: pd.DataFrame, model_type: str) -> float:
        if model_type == 'random_forest':
            preprocessed = self.scaler[0].transform(df)
            prediction = int(np.exp(self.rf_model.predict(preprocessed)))
        elif model_type == 'polynomial_regression':
            preprocessed = self.scaler[1].transform(df)
            prediction = int(self.scaler[2].inverse_transform(np.array(self.pl_model.predict(preprocessed, to_transform=True).reshape(-1, 1))).flatten())
        else:
            preprocessed = self.scaler[0].transform(df)
            prediction = int(self.class_model.predict(preprocessed))
        return prediction if prediction > 0 else -prediction % 100000


    def __str__(self) -> str:
        return super().__str__()


