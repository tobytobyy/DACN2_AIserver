from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"
    chat_db_path: str = "chat.db"


settings = Settings()
