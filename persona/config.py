"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration — values can be overridden via env vars or .env."""

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"  # "anthropic" | "openai"
    llm_model: str = "claude-sonnet-4-20250514"

    db_path: str = str(_PROJECT_ROOT / "persona_market.db")
    platform_commission: float = 0.20  # 20 %
    default_pricing: float = 5.0

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
