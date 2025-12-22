from dataclasses import dataclass
from typing import Optional, List

from PIL import Image

from app.core.config import settings
from app.infra.clip_router import ClipRouter

LABELS: List[str] = [
    "a photo of delicious food",
    "medicine pills or blister pack",
    "medical report document",
    "human face",
    "generic object",
]

FOOD_LABEL = "a photo of delicious food"


@dataclass
class RouteDecision:
    is_food: bool
    food_score: float
    best_label: Optional[str]
    best_score: float


class VisionRouterService:
    def __init__(self):
        self.clip = ClipRouter()

    def route(self, image: Image.Image) -> RouteDecision:
        scores = self.clip.score(image, LABELS)
        best_label, best_score = scores[0]

        food_score = 0.0
        for label, sc in scores:
            if label == FOOD_LABEL:
                food_score = sc
                break

        is_food = (best_label == FOOD_LABEL) and (food_score > settings.FOOD_THRESHOLD)

        return RouteDecision(
            is_food=is_food,
            food_score=food_score,
            best_label=best_label,
            best_score=best_score,
        )
