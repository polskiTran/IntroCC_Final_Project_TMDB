"""Centralized configuration for the TMDB project.

All tunables live here so the rest of the code never reads env vars directly.
Values can be overridden via the `.env` file or process environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
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
    discover_page_concurrency: int = 15
    save_discover_pages: bool = False

    sample_counts: int = 10000
    min_vote_count: int = 10
    start_year: int = 1980
    min_budget_usd: int = 100_000

    bronze_dir: Path = _PROJECT_ROOT / "data" / "bronze"
    silver_dir: Path = _PROJECT_ROOT / "data" / "silver"
    gold_dir: Path = _PROJECT_ROOT / "data" / "gold"
    ml_dir: Path = _PROJECT_ROOT / "models"
    model_card_path: Path = _PROJECT_ROOT / "models" / "model_card.md"

    data_backend: Literal["local", "s3"] = Field(
        default="local",
        description=(
            "Where Streamlit/ML read medallion data: local `data/` paths or `s3://`. "
            "Ingestion always writes to local `bronze_dir` / `silver_dir` / `gold_dir`; "
            "use `python -m src.ingest upload-s3` to sync those folders to the bucket."
        ),
    )
    aws_region: str = "us-east-2"
    s3_bucket: str = Field(
        default="",
        description="S3 bucket for reading (when data_backend=s3) and for upload-s3.",
    )
    s3_prefix: str = Field(
        default="",
        description="Optional key prefix inside the bucket (no leading/trailing slashes).",
    )
    s3_upload_concurrency: int = Field(
        default=24,
        ge=1,
        description="Parallel S3 uploads for the upload-s3 ingest stage.",
    )

    @model_validator(mode="after")
    def _require_s3_bucket_when_remote(self) -> Settings:
        if self.data_backend == "s3" and not str(self.s3_bucket).strip():
            msg = "s3_bucket is required when data_backend is 's3'"
            raise ValueError(msg)
        return self

    @property
    def discover_dir(self) -> Path:
        return self.bronze_dir / "discover"

    @property
    def movies_bronze_dir(self) -> Path:
        return self.bronze_dir / "movies"


@lru_cache
def get_settings() -> Settings:
    return Settings()
