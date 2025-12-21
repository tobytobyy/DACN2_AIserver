from app.api.routes_chat import router as chat_router
from app.api.routes_food import router as food_router
from app.core.state import model_state
from app.inference import load_artifacts, build_model, build_preprocess
from fastapi import FastAPI

app = FastAPI(title="DACN2_AIserver")


@app.on_event("startup")
def _startup():
    cfg, classes = load_artifacts()
    model = build_model(cfg)
    preprocess = build_preprocess(cfg)

    model_state.model = model
    model_state.preprocess = preprocess
    model_state.classes = classes

    # model_version: lấy từ cfg nếu có, fallback về arch
    model_state.model_version = str(cfg.get("version") or cfg.get("arch") or "unknown")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_state.model is not None,
        "num_classes": len(model_state.classes) if model_state.classes else 0,
        "model_version": model_state.model_version,
    }


app.include_router(food_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
