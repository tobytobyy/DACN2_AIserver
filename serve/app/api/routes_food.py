import io
import time

from PIL import Image
from app.core.state import model_state
from app.inference import predict_with
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()


@router.post("/food:predict")
async def food_predict(image: UploadFile = File(...), top_k: int = 3):
    t0 = time.time()

    # Giữ đúng validate như repo hiện tại (1..10)
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    if (
        model_state.model is None
        or model_state.preprocess is None
        or model_state.classes is None
    ):
        raise HTTPException(status_code=503, detail="Model not loaded")

    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    preds = predict_with(
        model_state.model, model_state.preprocess, model_state.classes, img, top_k=top_k
    )

    return {
        "model_version": model_state.model_version,
        "predictions": preds,
        "latency_ms": int((time.time() - t0) * 1000),
    }
