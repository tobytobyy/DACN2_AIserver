from typing import List

from pydantic import BaseModel, Field


class FoodPrediction(BaseModel):
    label: str
    score: float


class FoodImageRequest(BaseModel):
    image_url: str


class FoodImageResponse(BaseModel):
    status: str
    is_food: bool
    message: str
    predictions: List[FoodPrediction] = Field(default_factory=list)
