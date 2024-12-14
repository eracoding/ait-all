from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    brand: Optional[str] = Field(default='Unknown')
    year: Optional[str] = Field(default='2000')
    km_driven: Optional[str] = Field(default='0')
    fuel: Optional[str] = Field(default='Petrol')
    seller_type: Optional[str] = Field(default='Dealer')
    transmission: Optional[str] = Field(default='manual')
    owner: Optional[str] = Field(default='First')
    mileage: Optional[str] = Field(default='0')
    engine: Optional[str] = Field(default='Standard')
    max_power: Optional[str] = Field(default='0')
    seats: Optional[str] = Field(default='4')
    model: Optional[str] = Field(default='random_forest')


class PredictResponse(BaseModel):
    result: str = Field(..., title="result", description="Predict value", example=0.9)
