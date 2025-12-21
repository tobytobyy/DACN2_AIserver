from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# parents[2] => serve/
BASE_DIR = Path(__file__).resolve().parents[2]
artifacts_dir: str = str(BASE_DIR.parent / "artifacts")  # DACN2_AIserver/artifacts


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
    )

    groq_api_key: str
    groq_model: str = "llama-3.1-70b-versatile"
    artifacts_dir: str = artifacts_dir


settings = Settings()
