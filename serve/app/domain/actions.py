from typing import Any, Dict, List


def build_suggested_actions(
    *,
    is_food: bool,
    session_id: str,
    food_predictions: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    # Suggested action to add food to nutrition log
    # Only if the image is classified as food
    # if not is_food or not food_predictions:
    #     ️ return []
    if not is_food:
        return []

    return [
        {
            "type": "BUTTON",
            "label": "Thêm vào nhật ký dinh dưỡng",
            "action_api": "/api/user/nutrition/log",
            "payload": {
                "session_id": session_id,
                "predictions": food_predictions or [],
            },
        }
    ]
