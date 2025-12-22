from typing import List, Tuple

import torch
from PIL import Image

from app.core.state import model_state


class ClipRouter:
    def __init__(self):
        if model_state.clip_model is None or model_state.clip_processor is None:
            raise RuntimeError("CLIP model/processor not loaded")

        self.model = model_state.clip_model
        self.processor = model_state.clip_processor
        self.device = model_state.device
        self.model.eval()

    @torch.inference_mode()
    def score(self, image: Image.Image, labels: List[str]) -> List[Tuple[str, float]]:
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        # move tensors to device (CPU or GPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = logits.softmax(dim=0)

        scored = [(labels[i], float(probs[i].item())) for i in range(len(labels))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
