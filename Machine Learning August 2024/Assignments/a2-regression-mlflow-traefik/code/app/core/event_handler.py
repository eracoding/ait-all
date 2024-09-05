from typing import Callable

from fastapi import FastAPI

from app.services.model import MLModel
from app.services.custom import PolynomialRegression

import pickle


def _startup_model(app: FastAPI, model_path: str, model_path2: str, X_scaler_path: str, Y_scaler_path: str, rf_scaler_path: str) -> None:
    model_instance = MLModel(model_path, model_path2, X_scaler_path, Y_scaler_path, rf_scaler_path)
    # with open(model_path2, 'rb') as file:
    #     test = pickle.load(file)
    app.state.model = model_instance
    app.state.scaler = model_instance.scaler


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None
    app.state.scaler = None


def start_app_handler(app: FastAPI, model_path: str, model_path2: str, X_scaler_path: str, Y_scaler_path: str, rf_scaler_path: str) -> Callable:
    def startup() -> None:
        _startup_model(app, model_path, model_path2, X_scaler_path, Y_scaler_path, rf_scaler_path)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
