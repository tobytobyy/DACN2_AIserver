serve/app/
main.py

api/
v1/
routes/
inference.py # POST /api/v1/inference/chat

core/
config.py # ONE settings system (ENV, tokens, thresholds, ollama, artifacts_dir)
security.py # verify X-Internal-Token (dev bypass)
fetch_image.py # image_url fetch + 5MB + content-type
locale.py # detect_vi(), choose_locale()
errors.py # centralized error codes mapping (nếu bạn đang dùng)
logging.py # (optional) request_id

domain/
food_pipeline.py
health_pipeline.py
vision_router_service.py
llm_engine.py
guardrails.py
actions.py
routing_hint.py
vqa_questions.py
llm_intents.py

infra/
clip_router.py
blip_vqa.py
llm_ollama.py
translator_ollama.py # option

schemas/
inference.py