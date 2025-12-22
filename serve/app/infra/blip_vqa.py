from typing import List, Dict

import torch
from PIL import Image

from app.core.state import model_state


# BLIP-VQA inference class
class BlipVQA:
    # Constructor
    def __init__(self):
        if model_state.blip_vqa_model is None or model_state.blip_vqa_processor is None:
            raise RuntimeError("BLIP-VQA not loaded")
        self.model = model_state.blip_vqa_model
        self.processor = model_state.blip_vqa_processor
        self.device = model_state.device
        self.model.eval()

    # Inference method for multiple questions
    @torch.inference_mode()
    def ask_many(
        self, image: Image.Image, questions: List[str], max_new_tokens: int = 16
    ) -> Dict[str, str]:
        answers: Dict[str, str] = {}
        for q in questions:
            inputs = self.processor(images=image, text=q, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            ans = self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
            answers[q] = ans
        return answers
