import os
from pathlib import Path

from pydantic import BaseModel

# Read token from ENV variable: INTERNAL_TOKEN

# parents[2] => serve/
BASE_DIR = Path(__file__).resolve().parents[2]
artifacts_dir: str = str(BASE_DIR.parent / "artifacts")  # DACN2_AIserver/artifacts


class Settings(BaseModel):
    # ENV variables
    ENV: str = os.getenv("ENV", "dev").lower()  # dev | staging | prod
    INTERNAL_TOKEN: str = os.getenv("INTERNAL_TOKEN", "")

    # CLIP
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    FOOD_THRESHOLD: float = float(os.getenv("FOOD_THRESHOLD", "0.35"))

    # BLIP VQA
    BLIP_VQA_MODEL_NAME: str = os.getenv(
        "BLIP_VQA_MODEL_NAME", "Salesforce/blip-vqa-base"
    )
    VQA_MAX_QUESTIONS: int = int(os.getenv("VQA_MAX_QUESTIONS", "6"))

    # Device
    DEVICE: str = os.getenv("DEVICE", "auto").lower()  # auto | cuda | cpu

    # Ollama LLM
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT_S: float = float(os.getenv("OLLAMA_TIMEOUT_S", "30"))

    # Artifacts dir
    artifacts_dir: str = artifacts_dir

    @property
    def is_dev(self) -> bool:
        return self.ENV in {"dev", "local"}


settings = Settings()
