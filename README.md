serve/app/
main.py # giữ lifespan load food model (đang tốt)
api/
routes_food.py # giữ /v1/food/predict (dev/debug)
routes_chat.py # giữ demo endpoints (dev/debug)
routes_inference.py # NEW: /api/v1/inference/chat (canonical)
schemas/
chat.py # đang có (demo)
inference.py # NEW: request/response theo spec
core/
state.py # giữ model_state
prompts.py # giữ SYSTEM_PROMPT, bổ sung system prompts theo spec
security.py # NEW: verify X-Internal-Token
fetch_image.py # NEW: download image_url async + validate
domain/
orchestrator.py # NEW: router -> pipeline -> llm -> response
vision_router.py # NEW: CLIP route
food_pipeline.py # NEW: reuse predict_with + enrich nutrition
health_pipeline.py # NEW: BLIP/VQA caption
llm_engine.py # NEW: local-first + fallback groq
guardrails.py # NEW: keyword filter + disclaimer + triage
infra/
spring_client.py # NEW: httpx async
clip_router.py # NEW
blip_vqa.py # NEW
llm_runtime.py # NEW: ollama/llama-cpp wrapper
inference.py # giữ (food model)