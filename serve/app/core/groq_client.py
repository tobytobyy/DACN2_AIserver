from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.settings import settings
from fastapi import HTTPException
from openai import OpenAI


def groq_client() -> OpenAI:
    """
    Create a Groq client using OpenAI-compatible SDK.
    """
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def call_groq_chat(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_retries: int = 2,
    timeout_s: int = 20,
    backoff_s: float = 0.8,
) -> Tuple[str, Dict[str, Any]]:
    """
    Canonical Groq chat caller with:
      - retry + linear backoff
      - request timeout
      - meta (latency_ms, attempts, model, usage best-effort)

    Returns:
      (answer, meta)

    Notes:
      - `messages` should be OpenAI-style: [{"role":"system|user|assistant", "content":"..."}]
      - Raises HTTPException(502) on final failure.
    """
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    use_model = model or getattr(settings, "groq_model", None)
    if not use_model:
        raise HTTPException(status_code=500, detail="groq_model is not configured")

    if max_retries < 0:
        raise HTTPException(status_code=400, detail="max_retries must be >= 0")
    if timeout_s <= 0:
        raise HTTPException(status_code=400, detail="timeout_s must be > 0")
    if backoff_s < 0:
        raise HTTPException(status_code=400, detail="backoff_s must be >= 0")

    client = groq_client()
    last_err: Optional[Exception] = None
    t0 = time.time()

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=temperature,
                timeout=timeout_s,
            )

            # Robustly extract content
            answer = ""
            try:
                answer = resp.choices[0].message.content or ""
            except Exception:
                answer = ""

            meta: Dict[str, Any] = {
                "model": use_model,
                "attempts": attempt + 1,
                "latency_ms": int((time.time() - t0) * 1000),
            }

            # Best-effort usage extraction (SDK objects can vary)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                meta["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

            return answer, meta

        except Exception as e:
            last_err = e

            # retry if we still have attempts left
            if attempt < max_retries:
                # linear backoff: backoff_s * (attempt+1)
                sleep_s = backoff_s * (attempt + 1)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            # out of retries → surface as 502
            raise HTTPException(
                status_code=502,
                detail=f"Groq call failed: {str(last_err)}",
            )


def call_groq_chat_raw(
    *,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    timeout_s: int = 20,
) -> Any:
    """
    Optional helper if you ever need the raw response object (no retry).
    Keeping this separate prevents retry duplication elsewhere.
    """
    use_model = model or getattr(settings, "groq_model", None)
    if not use_model:
        raise HTTPException(status_code=500, detail="groq_model is not configured")

    client = groq_client()
    return client.chat.completions.create(
        model=use_model,
        messages=messages,
        temperature=temperature,
        timeout=timeout_s,
    )
