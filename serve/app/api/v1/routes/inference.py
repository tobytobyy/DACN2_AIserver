from fastapi import APIRouter, Depends, Request

from app.core.fetch_image import fetch_image_from_url
from app.core.security import verify_internal_token
from app.domain.actions import build_suggested_actions
from app.domain.guardrails import apply_guardrails
from app.domain.routing_hint import to_routing_hint
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
    # defaults (important to avoid UnboundLocalError)
    decision = None
    detected_items = []
    details = {}

    router_food_score = None
    router_best_label = None
    router_best_label_score = None
    routing_hint = "no_image"  # deafault if no image

    # if req.image_url is available, do vision analysis
    # if False, skip vision analysis and go to LLM directly
    if req.image_url:
        image = await fetch_image_from_url(req.image_url)

        router_service = request.app.state.vision_router
        decision = router_service.route(image)

        router_food_score = decision.food_score
        router_best_label = decision.best_label
        router_best_label_score = decision.best_score
        routing_hint = to_routing_hint(
            is_food=decision.is_food, best_label=router_best_label
        )

        if decision.is_food:
            fp = request.app.state.food_pipeline.analyze(image, top_k=3)
            detected_items = fp["detected_items"]
            details.update(
                {
                    "food_predictions": fp["food_predictions"],
                    "food_top1_score": fp["food_top1_score"],
                }
            )
        else:
            hp = request.app.state.health_pipeline.analyze(
                image, clip_best_label=router_best_label or ""
            )
            details.update(
                {
                    "structured_context": hp["structured_context"],
                    "vqa_answers": hp["vqa_answers"],
                }
            )

    # populate details
    details.update(
        {
            "routing_hint": routing_hint,
            "router_food_score": router_food_score,
            "router_best_label_score": router_best_label_score,
        }
    )

    analyzed = VisionAnalysis(
        is_food=bool(decision.is_food) if decision else False,
        detected_items=detected_items,
        nutrition_facts={},
        confidence=router_food_score,  # theo chuẩn B bạn chốt
        detected_label=router_best_label,
        details=details,
    )

    # call LLM to generate intent and text response
    llm = request.app.state.llm_engine
    intent, text = await llm.generate_intent_and_text(
        user_message=req.message,
        user_context=req.user_context.model_dump(),
        analyzed_image=analyzed.model_dump(),
    )
    text = apply_guardrails(intent=intent, user_message=req.message, text_response=text)

    actions = build_suggested_actions(
        is_food=analyzed.is_food,
        session_id=req.session_id,
        food_predictions=(analyzed.details or {}).get("food_predictions"),
    )

    data = InferenceData(
        text_response=text,
        intent_detected=intent,
        analyzed_image=analyzed,
        suggested_actions=actions,
    )
    return InferenceResponse(status="success", data=data)
