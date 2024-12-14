from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import HTMLResponse

from app.models.models import PredictRequest, PredictResponse
from app.utils.preprocess import process_input


api_router = APIRouter()


@api_router.get("/", response_class=HTMLResponse)
async def get_form():
    with open("app/static/form.html", "r") as file:
        return HTMLResponse(content=file.read())


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_data = payload.dict()

    processed_data = process_input(input_data).to_numpy()

    model_type = input_data.get('model')
    
    model = request.app.state.model

    prediction = str(model.predict(processed_data, model_type))

    if model_type == 'logistic_regression':
        prediction = f"class {prediction}\n\n class 0: [<260000]\n class 1: [260000-450000]\n class 2: [450000-680000]\n class 3: [680000>]"

    return PredictResponse(result=prediction)
