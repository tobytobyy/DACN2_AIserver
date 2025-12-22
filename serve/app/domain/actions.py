from typing import Any, Dict, List


def build_suggested_actions(
    *,
    is_food: bool,
    session_id: str,
    food_predictions: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    if not is_food:
        return []

    return [
        {
            "type": "BUTTON",
            "label": "Add to nutrition log",
            "action_api": "/api/user/nutrition/log",
            "payload": {
                "session_id": session_id,
                "predictions": food_predictions or [],
            },
        }
    ]
