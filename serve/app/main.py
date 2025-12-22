from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import BlipForQuestionAnswering, BlipProcessor
from transformers import CLIPModel, CLIPProcessor

from app.api.v1.routes.inference import router as inference_router
from app.core.config import settings
from app.core.state import model_state
from app.domain.food_pipeline import FoodPipeline
from app.domain.health_pipeline import HealthPipeline
from app.domain.llm_engine import LLMEngine
from app.domain.vision_router_service import VisionRouterService
from app.inference import load_artifacts, build_model, build_preprocess


def pick_device() -> str:
    if settings.DEVICE in {"cpu", "mps"}:
        return settings.DEVICE
    # auto
    return "mps" if torch.backends.mps.is_available() else "cpu"

    # if settings.DEVICE == "cuda":
    #     return "cuda"
    # if settings.DEVICE == "cpu":
    #     return "cpu"
    # # auto
    # return "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = pick_device()
    try:
        # 1) load food model
        cfg, classes = load_artifacts()
        model = build_model(cfg)
        model.to(device)
        preprocess = build_preprocess(cfg)
        model_state.model = model
        model_state.preprocess = preprocess
        model_state.classes = classes
        model_state.device = device

        # 2) load CLIP router
        model_state.clip_processor = CLIPProcessor.from_pretrained(
            settings.CLIP_MODEL_NAME
        )
        model_state.clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
        model_state.clip_model.eval()
        model_state.clip_model.to(device)

        # 3) load BLIP VQA
        model_state.blip_vqa_processor = BlipProcessor.from_pretrained(
            settings.BLIP_VQA_MODEL_NAME
        )
        model_state.blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
            settings.BLIP_VQA_MODEL_NAME
        )
        model_state.blip_vqa_model.eval()
        model_state.blip_vqa_model.to(device)

        # 4) initialize domain services after model load
        app.state.food_pipeline = FoodPipeline()
        app.state.health_pipeline = HealthPipeline()
        app.state.vision_router = VisionRouterService()
        app.state.llm_engine = LLMEngine()

        model_state.error = None
    except Exception as e:
        model_state.error = str(e)
    yield


app = FastAPI(title="AI Inference Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_state.model is not None,
        "num_classes": len(model_state.classes) if model_state.classes else 0,
        "device": model_state.device,
        "error": model_state.error,
    }


app.include_router(inference_router)
