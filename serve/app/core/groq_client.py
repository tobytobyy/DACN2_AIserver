from app.core.settings import settings
from openai import OpenAI


def groq_client() -> OpenAI:
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
