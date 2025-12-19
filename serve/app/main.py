from app.api.routes_food import router as food_router
from fastapi import FastAPI

app = FastAPI(title="DACN2_AIserver")


@app.get("/health")
def health():
    return {"status": "ok"}


# Giữ nguyên đường dẫn public: /v1/food:predict
app.include_router(food_router, prefix="/v1")
