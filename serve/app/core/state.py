from dataclasses import dataclass
from typing import Any, Optional, List


@dataclass
class ModelState:
    # food model
    model: Optional[Any] = None
    preprocess: Optional[Any] = None
    classes: Optional[List[str]] = None

    # clip router
    clip_model: Optional[Any] = None
    clip_processor: Optional[Any] = None

    # blip vqa
    blip_vqa_model: Optional[Any] = None
    blip_vqa_processor: Optional[Any] = None

    # device
    device: str = "cpu"

    error: Optional[str] = None


model_state = ModelState()
