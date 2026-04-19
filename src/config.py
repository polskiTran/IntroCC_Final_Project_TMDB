"""Centralized configuration for the TMDB project.

All tunables live here so the rest of the code never reads env vars directly.
Values can be overridden via the `.env` file or process environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tmdb_api_key: str = Field(default="", description="TMDB v3 API key")
    tmdb_bearer_token: str | None = Field(
        default=None, description="TMDB v4 bearer token (preferred if set)"
    )
    tmdb_base_url: str = "https://api.themoviedb.org/3"

    requests_per_second: int = 40
    concurrency: int = 20

    sample_counts: int = 10000
    min_vote_count: int = 10
    start_year: int = 1980
    min_budget_usd: int = 100_000

    bronze_dir: Path = _PROJECT_ROOT / "data" / "bronze"
    silver_dir: Path = _PROJECT_ROOT / "data" / "silver"
    gold_dir: Path = _PROJECT_ROOT / "data" / "gold"
    ml_dir: Path = _PROJECT_ROOT / "data" / "ml"
    model_card_path: Path = _PROJECT_ROOT / "model_card.md"

    @property
    def discover_dir(self) -> Path:
        return self.bronze_dir / "discover"

    @property
    def movies_bronze_dir(self) -> Path:
        return self.bronze_dir / "movies"


@lru_cache
def get_settings() -> Settings:
    return Settings()
