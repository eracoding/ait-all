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

    prediction = model.predict(processed_data, model_type)

    return PredictResponse(result=prediction)
