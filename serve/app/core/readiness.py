from fastapi import HTTPException, Request, status

from app.core.state import model_state
from app.infra.blip_vqa import ensure_blip_loaded


def _startup_error_suffix() -> str:
    return f" Startup error: {model_state.error}" if model_state.error else ""


def require_llm_ready(request: Request) -> None:
    # LLM engine is stored in app.state in lifespan
    if (
        not hasattr(request.app.state, "llm_engine")
        or request.app.state.llm_engine is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM engine is not ready." + _startup_error_suffix(),
        )


def require_clip_ready() -> None:
    # CLIP is required by ClipRouter constructor
    if model_state.clip_model is None or model_state.clip_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CLIP router is not ready." + _startup_error_suffix(),
        )


def require_food_ready() -> None:
    # FoodPipeline constructor checks these
    if (
        model_state.model is None
        or model_state.preprocess is None
        or model_state.classes is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Food model is not ready." + _startup_error_suffix(),
        )


async def require_blip_ready() -> None:
    try:
        await ensure_blip_loaded()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"BLIP-VQA failed to initialize: {str(e)}" + _startup_error_suffix(),
        ) from e

    if model_state.blip_vqa_model is None or model_state.blip_vqa_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BLIP-VQA is not ready." + _startup_error_suffix(),
        )
