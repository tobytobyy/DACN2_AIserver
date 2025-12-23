from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image

from app.core.config import settings
from app.core.state import model_state

# Canonical label keys (stable)
FOOD_KEY = "food"
MEDICINE_KEY = "medicine"
MED_REPORT_KEY = "medical_document"
FACE_KEY = "face"
GENERIC_KEY = "generic_object"

# Consistent CLIP prompts (same template)
LABEL_PROMPTS: Dict[str, str] = {
    FOOD_KEY: "a photo of delicious food",
    MEDICINE_KEY: "a photo of medicine pills or a blister pack",
    MED_REPORT_KEY: "a photo of a medical report document",
    FACE_KEY: "a photo of a human face",
    GENERIC_KEY: "a photo of a generic object",
}


@dataclass
class RouteDecision:
    """
    best_key is a STABLE KEY (e.g., 'medicine'), not the raw prompt.
    """

    is_food: bool
    food_score: float
    best_key: Optional[str]
    best_score: float


class VisionRouterService:
    """
    CLIP-based router: determines whether an image should go to Food model or Health pipeline.

    Fixes:
    - Avoid the "food_score high but not top1" bug by using:
        is_food = food_score > threshold AND food_score >= max_non_food_score - margin
    - Uses consistent prompt templates for CLIP.
    """

    def __init__(self) -> None:
        if model_state.clip_model is None or model_state.clip_processor is None:
            raise RuntimeError("CLIP router is not loaded")

        self.model = model_state.clip_model
        self.processor = model_state.clip_processor
        self.device = model_state.device

        # Make sure the model is in eval mode
        self.model.eval()

    @torch.inference_mode()
    def route(self, image: Image.Image) -> RouteDecision:
        keys: List[str] = list(LABEL_PROMPTS.keys())
        texts: List[str] = [LABEL_PROMPTS[k] for k in keys]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, num_labels)
        probs = logits_per_image.softmax(dim=1)[0].detach().cpu().tolist()

        score_by_key: Dict[str, float] = {k: float(p) for k, p in zip(keys, probs)}
        sorted_scores: List[Tuple[str, float]] = sorted(
            score_by_key.items(), key=lambda x: x[1], reverse=True
        )

        best_key, best_score = sorted_scores[0]
        food_score = score_by_key.get(FOOD_KEY, 0.0)

        # Compute the best non-food score
        max_non_food_score = max(
            (sc for k, sc in score_by_key.items() if k != FOOD_KEY),
            default=0.0,
        )

        threshold_ok = food_score > settings.FOOD_THRESHOLD
        margin = getattr(settings, "FOOD_MARGIN", 0.02)  # safe default

        # Route to food if:
        # - food_score passes threshold
        # - food_score is not meaningfully lower than the best non-food score
        is_food = threshold_ok and (food_score >= max_non_food_score - margin)

        return RouteDecision(
            is_food=is_food,
            food_score=food_score,
            best_key=best_key,
            best_score=float(best_score),
        )
