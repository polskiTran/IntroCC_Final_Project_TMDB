"""Tests for batch S3 upload of local medallion directories (moto)."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.config import Settings
from src.ingest.upload_s3 import upload_medallion_to_s3
from src.io.store import s3_object_key


@pytest.fixture
def upload_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_backend="local",
        s3_bucket="upload-test-bucket",
        s3_prefix="proj",
        aws_region="us-east-1",
        tmdb_api_key="x",
        s3_upload_concurrency=4,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
    )


@mock_aws
def test_upload_medallion_writes_expected_keys(upload_settings: Settings) -> None:
    (upload_settings.bronze_dir / "movies" / "id_prefix=000").mkdir(parents=True)
    (upload_settings.bronze_dir / "movies" / "id_prefix=000" / "1.json.gz").write_bytes(
        b"x"
    )
    upload_settings.silver_dir.mkdir(parents=True, exist_ok=True)
    (upload_settings.silver_dir / "movies.parquet").write_bytes(b"parq")
    upload_settings.gold_dir.mkdir(parents=True, exist_ok=True)
    (upload_settings.gold_dir / "gold_movies.parquet").write_bytes(b"gold")

    boto3.client("s3", region_name="us-east-1").create_bucket(
        Bucket="upload-test-bucket"
    )

    out = upload_medallion_to_s3(upload_settings)
    assert out["files"] == 3
    assert out["bytes"] > 0

    client = boto3.client("s3", region_name="us-east-1")
    for rel in (
        ("bronze", "movies", "id_prefix=000", "1.json.gz"),
        ("silver", "movies.parquet"),
        ("gold", "gold_movies.parquet"),
    ):
        key = s3_object_key(upload_settings, *rel)
        head = client.head_object(Bucket="upload-test-bucket", Key=key)
        assert int(head["ContentLength"]) > 0
