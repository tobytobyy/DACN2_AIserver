import json
import re
from typing import Any, Dict, List, Tuple

from app.domain.llm_intents import ALLOWED_INTENTS
from app.infra.llm_ollama import OllamaClient

SYSTEM_PROMPT = """You are a health & nutrition assistant.
Safety rules:
- Do NOT claim a certain medical diagnosis.
- Do NOT prescribe dangerous medication.
- If urgent symptoms are possible, advise seeing a clinician or emergency care.

Output rules (VERY IMPORTANT):
- Reply with ONLY a valid JSON object.
- The JSON must have exactly these keys:
  - "intent_detected": one of: food_inquiry, food_logging, medication_question, medical_document, symptom_check, general_health, general_chat, unknown
  - "text_response": a helpful response in Vietnamese.
- No extra keys. No markdown. No explanations outside JSON.
"""


def build_messages(
    user_message: str, user_context: Dict[str, Any], analyzed_image: Dict[str, Any]
) -> List[Dict[str, str]]:
    hint = (analyzed_image.get("details") or {}).get("routing_hint", "generic")

    if analyzed_image.get("is_food"):
        preds = (analyzed_image.get("details") or {}).get("food_predictions", [])
        vision_context = f"ROUTING_HINT={hint}; IMAGE_TYPE=FOOD; predictions={preds}"
    else:
        sc = (analyzed_image.get("details") or {}).get("structured_context", "")
        vision_context = (
            f"ROUTING_HINT={hint}; IMAGE_TYPE=NON_FOOD; structured_context_en={sc}"
        )

    user_block = f"""USER_MESSAGE: {user_message}

USER_CONTEXT:
{user_context}

VISION_CONTEXT:
{vision_context}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]


def _extract_json_object(s: str) -> str:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM output")
    return m.group(0)


def parse_llm_json(raw: str) -> Tuple[str, str]:
    obj = json.loads(_extract_json_object(raw))

    intent = str(obj.get("intent_detected", "unknown")).strip()
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    text = str(obj.get("text_response", "")).strip()
    if not text:
        text = "Mình chưa đủ thông tin để trả lời. Bạn có thể mô tả rõ hơn giúp mình không?"

    return intent, text


class LLMEngine:
    def __init__(self):
        self.client = OllamaClient()

    async def generate_intent_and_text(
        self,
        user_message: str,
        user_context: Dict[str, Any],
        analyzed_image: Dict[str, Any],
    ) -> Tuple[str, str]:
        messages = build_messages(user_message, user_context, analyzed_image)
        raw = await self.client.chat(messages, temperature=0.2)

        try:
            return parse_llm_json(raw)
        except Exception:
            safe_text = raw.strip()
            if not safe_text:
                safe_text = "Mình chưa đủ thông tin để trả lời. Bạn có thể mô tả rõ hơn giúp mình không?"
            return "unknown", safe_text
