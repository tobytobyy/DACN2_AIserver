from fastapi import APIRouter, Depends, Request, HTTPException, status

from app.core.fetch_image import fetch_image_from_url
from app.core.security import verify_internal_token
from app.core.state import model_state
from app.domain.actions import build_suggested_actions
from app.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceData,
    VisionAnalysis,
)

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("/chat", response_model=InferenceResponse)
async def chat(
    req: InferenceRequest, request: Request, _=Depends(verify_internal_token)
):
    if (
        model_state.error
        or model_state.clip_model is None
        or model_state.blip_vqa_model is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not ready"
        )

    # fetch image if provided
    decision = None
    details = {}
    detected_items = []
    confidence = None

    if req.image_url:
        image = await fetch_image_from_url(req.image_url)

        router_service = request.app.state.vision_router

        decision = router_service.route(image)

        if decision.is_food:
            food_pipeline = request.app.state.food_pipeline
            fp = food_pipeline.analyze(image, top_k=3)

            detected_items = fp["detected_items"]
            confidence = fp["confidence"]

            details = {"food_predictions": fp.get("food_predictions", [])}

        else:
            health_pipeline = request.app.state.health_pipeline
            hp = health_pipeline.analyze(
                image, clip_best_label=decision.best_label or ""
            )
            details = {
                "structured_context": hp["structured_context"],
                "vqa_answers": hp["vqa_answers"],
            }

    analyzed = VisionAnalysis(
        is_food=decision.is_food if decision else False,
        detected_items=detected_items,
        nutrition_facts={},  # chưa làm
        confidence=(
            confidence
            if decision and decision.is_food
            else (decision.food_score if decision else None)
        ),
        detected_label=decision.best_label if decision else None,
        details=details,
    )

    actions = build_suggested_actions(
        is_food=analyzed.is_food,
        session_id=req.session_id,
        food_predictions=(analyzed.details or {}).get("food_predictions"),
    )

    llm = request.app.state.llm_engine

    try:
        intent, text = await llm.generate_intent_and_text(
            user_message=req.message,
            user_context=req.user_context.model_dump(),
            analyzed_image=analyzed.model_dump(),
        )
    except Exception:
        # Nếu LLM local chết / Ollama chưa chạy => 503
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM not available"
        )

    # stub text_response
    data = InferenceData(
        text_response=text,
        intent_detected=intent,
        analyzed_image=analyzed,
        suggested_actions=actions,
    )
    return InferenceResponse(status="success", data=data)
