import json
import re
from typing import Any, Dict, List, Tuple

from app.domain.llm_intents import ALLOWED_INTENTS
from app.infra.llm_ollama import OllamaClient

SYSTEM_PROMPT = """You are an AI Health & Nutrition Assistant.

### CORE SAFETY & BEHAVIOR RULES:
1. **NO DIAGNOSIS/RX:** Do NOT provide definitive medical diagnoses. Do NOT prescribe medications.
2. **EMERGENCY PROTOCOL:** If the user mentions symptoms like chest pain, severe shortness of breath, fainting, confusion, heavy bleeding, or paralysis ("URGENCY_HINTS"), you MUST set "severity_level" to "emergency" and advise immediate medical attention.
3. **PERSONALIZATION:** If a "User Health Snapshot" is provided, use it to tailor advice (e.g., considering diabetes for food queries).
4. **LANGUAGE:** The "text_response" must be in the SAME language as the user's input.

### OUTPUT FORMAT RULES (STRICT):
- Reply with ONLY a valid JSON object.
- Do NOT wrap the output in markdown code blocks (do not use ```json).
- No extra text outside the JSON.
- The JSON must have exactly the following keys:
{
  "intent_detected": "String. One of: [symptom_check, food_inquiry, food_logging, medication_question, medical_document, general_health, general_chat, unknown]",
  "severity_level": "String. One of: [emergency, urgent, monitor, none]. (Use 'none' for general chat/food logging)",  
  "text_response": "String. The natural language reply to the user.
     - If intent is 'symptom_check' or 'medical_document', structure the text as:
       1) Summary & Assessment
       2) Actionable Next Steps
       3) Clarifying Questions (if needed)
       4) Safety Disclaimer"
}
- No extra keys. No explanations outside JSON.
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
        text = (
            "I don't have enough information yet. Could you describe it in more detail?"
        )

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
