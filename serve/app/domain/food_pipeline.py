from typing import Dict, Any

from PIL import Image

from app.core.state import model_state
from app.infra.predict_food import predict_with


class FoodPipeline:
    def __init__(self):
        if (
            model_state.model is None
            or model_state.preprocess is None
            or model_state.classes is None
        ):
            raise RuntimeError("Food model not loaded")

        self.model = model_state.model
        self.preprocess = model_state.preprocess
        self.classes = model_state.classes

    def analyze(self, image: Image.Image, top_k: int = 3) -> Dict[str, Any]:
        preds = predict_with(
            self.model, self.preprocess, self.classes, image, top_k=top_k
        )

        normalized = []
        for i, p in enumerate(preds, start=1):
            normalized.append(
                {
                    "rank": i,
                    "label": p["label"],
                    "score": float(p["score"]),
                    "source": "food_model_v1",
                }
            )

        detected_items = [p["label"] for p in preds]
        food_top1_score = float(preds[0]["score"]) if preds else None

        return {
            "detected_items": detected_items,
            "food_top1_score": food_top1_score,
            "food_predictions": normalized,
        }
