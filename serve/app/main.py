import io
import time

from PIL import Image
from app.inference import predict
from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="DACN2_AIserver")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/food:predict")
async def food_predict(image: UploadFile = File(...), top_k: int = 3):
    t0 = time.time()
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    preds = predict(img, top_k=top_k)
    return {
        "model_version": "food101-resnet50-v1",
        "predictions": preds,
        "latency_ms": int((time.time() - t0) * 1000),
    }
