import io
import json

from PIL import Image
from app.core.groq_client import groq_client
from app.core.prompts import SYSTEM_PROMPT
from app.core.settings import settings
from app.inference import predict  # dùng food model hiện có (MVP)
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter()


def build_messages(req: ChatMessageRequest, image_findings=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    if req.conversation_summary:
        msgs.append(
            {
                "role": "system",
                "content": f"Conversation summary: {req.conversation_summary}",
            }
        )

    if req.history:
        msgs.extend([{"role": m.role, "content": m.content} for m in req.history])

    ctx_parts = []
    if req.user_context:
        ctx_parts.append(f"User Health Snapshot (JSON): {req.user_context}")
    if image_findings:
        ctx_parts.append(f"Image findings: {image_findings}")

    user_content = req.message
    if ctx_parts:
        user_content += "\n\n" + "\n".join(ctx_parts)

    msgs.append({"role": "user", "content": user_content})
    return msgs


@router.post("/chat:message", response_model=ChatMessageResponse)
def chat_message(req: ChatMessageRequest):
    client = groq_client()
    messages = build_messages(req)

    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    return ChatMessageResponse(
        session_id=req.session_id,
        answer=answer,
        meta={
            "used_history": len(req.history or []),
            "has_user_context": bool(req.user_context),
        },
    )


@router.post("/chat:image", response_model=ChatMessageResponse)
async def chat_image(
    user_id: str = Form(...),
    session_id: str = Form(...),
    message: str = Form(""),
    user_context: str = Form(""),
    history: str = Form(""),
    conversation_summary: str = Form(""),
    image: UploadFile = File(...),
):
    # Parse JSON fields (Spring gửi string)
    uc = json.loads(user_context) if user_context else None
    hist = json.loads(history) if history else None

    req = ChatMessageRequest(
        user_id=user_id,
        session_id=session_id,
        message=message or "Hãy phân tích ảnh và đưa gợi ý.",
        user_context=uc,
        history=hist,
        conversation_summary=conversation_summary or None,
    )

    # MVP: dùng food classifier có sẵn
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    preds = predict(img, top_k=3)
    image_findings = {"food_predictions": preds}

    client = groq_client()
    messages = build_messages(req, image_findings=image_findings)

    resp = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    return ChatMessageResponse(
        session_id=session_id, answer=answer, meta={"image_findings": image_findings}
    )
