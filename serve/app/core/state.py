from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelState:
    model: Optional[Any] = None
    classes: Optional[list[str]] = None
    preprocess: Optional[Any] = None
    model_version: str = "unknown"


model_state = ModelState()
