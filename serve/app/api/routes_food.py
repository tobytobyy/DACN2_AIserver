import io
import time

from PIL import Image
from app.inference import predict
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()


@router.post("/food:predict")
async def food_predict(image: UploadFile = File(...), top_k: int = 3):
    t0 = time.time()
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")

    img_bytes = await image.read()
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    preds = predict(img, top_k=top_k)
    return {
        "model_version": "food101-resnet50-v1",
        "predictions": preds,
        "latency_ms": int((time.time() - t0) * 1000),
    }
