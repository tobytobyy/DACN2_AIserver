from app.domain.vision_router_service import MEDICINE_KEY, MED_REPORT_KEY, FACE_KEY


def to_routing_hint(is_food: bool, best_key: str | None) -> str:
    if is_food:
        return "FOOD_KEY"
    if not best_key:
        return "generic"

    if best_key == MEDICINE_KEY:
        return "medicine"
    if best_key == MED_REPORT_KEY:
        return "medical_document"
    if best_key == FACE_KEY:
        return "face"
    return "generic"
