from app.api.routes_chat import router as chat_router
from app.api.routes_food import router as food_router
from fastapi import FastAPI

app = FastAPI(title="DACN2_AIserver")


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(food_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")
