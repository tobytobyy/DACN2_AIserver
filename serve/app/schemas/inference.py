from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request: session_id, message, image_url, user_context (required)
# Response: status, data.text_response, intent_detected, analyzed_image, suggested_actions
class UserContext(BaseModel):
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    activity_level: Optional[str] = None
    medical_conditions: List[str] = Field(default_factory=list)


class InferenceRequest(BaseModel):
    session_id: str
    message: str
    image_url: Optional[str] = None
    user_context: UserContext  # required


class VisionAnalysis(BaseModel):
    is_food: bool
    detected_items: List[str] = Field(default_factory=list)
    nutrition_facts: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    detected_label: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class SuggestedAction(BaseModel):
    type: str
    label: str
    action_api: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class InferenceData(BaseModel):
    text_response: str
    intent_detected: Optional[str] = None
    analyzed_image: VisionAnalysis
    suggested_actions: List[SuggestedAction] = Field(default_factory=list)


class InferenceResponse(BaseModel):
    status: str
    data: InferenceData
