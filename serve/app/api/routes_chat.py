from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional

from PIL import Image
from app.core.groq_client import call_groq_chat
from app.core.prompts import SYSTEM_PROMPT
from app.core.state import model_state
from app.core.uploads import validate_content_type, read_upload_limited
from app.inference import predict_with
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter()


# ---------------------------
# Helpers: JSON parsing
# ---------------------------


def _safe_json_loads(value: str, field_name: str) -> Any:
    """
    Accepts:
      - "" or None -> None
      - valid JSON string -> parsed object
    Raises 400 for invalid JSON.
    """
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON for '{field_name}': {str(e)}"
        )


def _normalize_history(raw_history: Any) -> Optional[List[Dict[str, str]]]:
    """
    Normalize history into a list of dicts: {"role": "...", "content": "..."}
    Accepts:
      - None
      - list[{"role": "...", "content": "..."}]
      - list[{"role": "...", "message": "..."}]  (common variant)
    Filters invalid entries and trims to safe shape.
    """
    if raw_history is None:
        return None
    if not isinstance(raw_history, list):
        raise HTTPException(
            status_code=400, detail="Field 'history' must be a JSON array"
        )

    out: List[Dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", item.get("message"))
        if role not in ("system", "user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content.strip()})

    return out or None


# ---------------------------
# Helpers: Prompt building
# ---------------------------


def _build_messages(
    *,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    conversation_summary: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    image_findings: Optional[Dict[str, Any]] = None,
    locale: Optional[str] = "vi",
) -> List[Dict[str, str]]:
    """
    Construct OpenAI/Groq messages with guardrails.
    """
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Optional: steer language
    if locale:
        msgs.append({"role": "system", "content": f"Respond in locale: {locale}."})

    # Optional summary from Spring
    if conversation_summary and conversation_summary.strip():
        msgs.append(
            {
                "role": "system",
                "content": (
                    "Conversation summary (provided by server): "
                    f"{conversation_summary.strip()}"
                ),
            }
        )

    # History provided by Spring (already trimmed on Spring side ideally)
    if history:
        msgs.extend(history)

    # Append structured context to the current user message
    ctx_parts: List[str] = []
    if user_context:
        ctx_parts.append(f"User Health Snapshot (JSON, from server): {user_context}")
    if image_findings:
        ctx_parts.append(f"Image findings (from AIserver pipeline): {image_findings}")

    user_text = (message or "").strip()
    if not user_text:
        user_text = "Hãy hỗ trợ tôi."

    if ctx_parts:
        user_text = user_text + "\n\n" + "\n".join(ctx_parts)

    msgs.append({"role": "user", "content": user_text})
    return msgs


# ---------------------------
# Endpoint: JSON text chat
# ---------------------------


@router.post("/chat/message", response_model=ChatMessageResponse)
def chat_message(req: ChatMessageRequest) -> ChatMessageResponse:
    normalized_history = _normalize_history(req.history) if req.history else None

    messages = _build_messages(
        message=req.message,
        history=normalized_history,
        conversation_summary=req.conversation_summary,
        user_context=req.user_context,
        image_findings=None,
        locale=req.locale or "vi",
    )

    answer, llm_meta = call_groq_chat(messages=messages)

    meta: Dict[str, Any] = {
        **llm_meta,
        "session_id": req.session_id,
        "user_id": req.user_id,
        "used_history": len(normalized_history or []),
        "has_user_context": bool(req.user_context),
        "has_summary": bool(req.conversation_summary),
        "mode": "text",
    }

    return ChatMessageResponse(session_id=req.session_id, answer=answer, meta=meta)


# ---------------------------
# Endpoint: multipart image chat
# ---------------------------


@router.post("/chat/image", response_model=ChatMessageResponse)
async def chat_image(
    user_id: str = Form(...),
    session_id: str = Form(...),
    message: str = Form(""),
    user_context: str = Form(""),  # JSON string
    history: str = Form(""),  # JSON string (array)
    conversation_summary: str = Form(""),  # plain string
    locale: str = Form("vi"),
    image: UploadFile = File(...),
) -> ChatMessageResponse:
    # Parse JSON fields safely
    uc = _safe_json_loads(user_context, "user_context")
    hist_raw = _safe_json_loads(history, "history")
    hist = _normalize_history(hist_raw)

    # Decode image
    t_img0 = time.time()
    try:
        validate_content_type(image)
        img_bytes = await read_upload_limited(image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # MVP image analysis using existing food model (but from model_state)
    try:
        if (
            model_state.model is None
            or model_state.preprocess is None
            or model_state.classes is None
        ):
            raise RuntimeError("Model not loaded")

        preds = predict_with(
            model_state.model, model_state.preprocess, model_state.classes, img, top_k=3
        )
        image_findings: Dict[str, Any] = {
            "type": "food_classification",
            "food_predictions": preds,
            "image_latency_ms": int((time.time() - t_img0) * 1000),
        }
    except Exception as e:
        image_findings = {
            "type": "food_classification",
            "error": f"food_predict_failed: {str(e)}",
            "image_latency_ms": int((time.time() - t_img0) * 1000),
        }

    messages = _build_messages(
        message=message or "Hãy phân tích ảnh và đưa gợi ý.",
        history=hist,
        conversation_summary=conversation_summary or None,
        user_context=uc,
        image_findings=image_findings,
        locale=locale or "vi",
    )

    answer, llm_meta = call_groq_chat(messages=messages)

    meta: Dict[str, Any] = {
        **llm_meta,
        "session_id": session_id,
        "user_id": user_id,
        "used_history": len(hist or []),
        "has_user_context": bool(uc),
        "has_summary": bool(conversation_summary),
        "mode": "image",
        "image_findings": image_findings,
        "upload": {"filename": image.filename, "content_type": image.content_type},
    }

    return ChatMessageResponse(session_id=session_id, answer=answer, meta=meta)
