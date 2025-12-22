URGENCY_HINTS = [
    "khó thở",
    "đau ngực",
    "ngất",
    "co giật",
    "chảy máu nhiều",
    "sưng nhanh",
    "phát ban toàn thân",
    "sốc",
    "lơ mơ",
]


def apply_guardrails(intent: str, user_message: str, text_response: str) -> str:
    msg_lower = (user_message or "").lower()

    # warning chung cho y tế/thuốc
    if intent in {
        "medication_question",
        "symptom_check",
        "general_health",
        "medical_document",
    }:
        tail = "\n\nLưu ý: Mình chỉ hỗ trợ tham khảo, không thay thế tư vấn của bác sĩ."

        # warning “khẩn cấp” nếu có từ khoá
        if any(k in msg_lower for k in URGENCY_HINTS):
            tail += " Nếu bạn có dấu hiệu nặng (khó thở/đau ngực/ngất/chảy máu nhiều...), hãy đi khám hoặc gọi cấp cứu ngay."

        # tránh append trùng
        if "không thay thế tư vấn của bác sĩ" not in (text_response or ""):
            return (text_response or "").rstrip() + tail

    return (text_response or "").strip()
