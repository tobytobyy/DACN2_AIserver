from typing import Any, Dict, List

import httpx

from app.core.config import settings


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip(
            "/"
        )  # for example, http://127.0.0.1:11434
        self.model = settings.OLLAMA_MODEL
        self.timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT_S)

    async def chat(
        self, messages: List[Dict[str, str]], temperature: float = 0.2
    ) -> str:
        """
        Ollama /api/chat:
        payload: {"model": "...", "messages": [...], "stream": false, "options": {...}}
        return: {"message": {"role":"assistant","content":"..."} , ...}
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        # Ollama return: {"message": {"role":"assistant","content":"..."}, ...}
        return (data.get("message") or {}).get("content", "").strip()
