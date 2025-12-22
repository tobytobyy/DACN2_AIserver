def to_routing_hint(is_food: bool, best_label: str | None) -> str:
    if is_food:
        return "food"
    if not best_label:
        return "generic"
    if best_label == "medicine pills or blister pack":
        return "medicine"
    if best_label == "medical report document":
        return "medical_document"
    if best_label == "human face":
        return "face"
    return "generic"
