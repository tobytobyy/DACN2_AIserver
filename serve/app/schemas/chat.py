from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class HistoryMsg(BaseModel):
    role: Role
    content: str


class ChatMessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    user_context: Optional[Dict[str, Any]] = None
    history: Optional[List[HistoryMsg]] = None
    conversation_summary: Optional[str] = None
    locale: Optional[str] = "vi"


class ChatMessageResponse(BaseModel):
    session_id: str
    answer: str
    meta: Dict[str, Any] = {}
