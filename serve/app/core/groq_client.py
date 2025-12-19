import time

from app.core.settings import settings
from fastapi import HTTPException
from openai import OpenAI


def groq_client() -> OpenAI:
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _call_groq(client, model, messages, temperature=0.3, max_retries=2):
    last_err = None
    for i in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=20,  # giây
            )
        except Exception as e:
            last_err = e
            if i < max_retries:
                time.sleep(0.8 * (i + 1))
            else:
                raise HTTPException(
                    status_code=502, detail=f"Groq call failed: {str(last_err)}"
                )
