import joblib
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Any
from sklearn.preprocessing import StandardScaler


class BaseMLModel(ABC):
    @abstractmethod
    def predict(self, req: Any) -> Any:
        raise NotImplementedError

    def __str__(self) -> str:
        return super().__str__()


class MLModel(BaseMLModel):
    """Sample ML model"""

    def __init__(self, model_path: str) -> None:
        self.model = joblib.load(model_path)
        self.scaler = StandardScaler()

    def predict(self, df: pd.DataFrame) -> float:
        return int(np.exp(self.model.predict(df)))


    def __str__(self) -> str:
        return super().__str__()
