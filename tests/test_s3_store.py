"""Smoke tests for S3 helpers using moto (no real AWS)."""

from __future__ import annotations

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.config import Settings
from src.io.store import s3_object_exists, s3_prefix_metrics, write_text_s3


@pytest.fixture
def s3_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_backend="s3",
        s3_bucket="test-bucket",
        s3_prefix="proj",
        aws_region="us-east-1",
        tmdb_api_key="x",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
    )


@mock_aws
def test_s3_write_text_exists_and_metrics(s3_settings: Settings) -> None:
    client = boto3.client("s3", region_name="us-east-1")
    client.create_bucket(Bucket="test-bucket")

    write_text_s3(
        s3_settings,
        "bronze",
        "manifests",
        "run_date=2026-01-01",
        "run_manifest.json",
        text='{"ok": true}',
    )
    assert s3_object_exists(
        s3_settings,
        "bronze",
        "manifests",
        "run_date=2026-01-01",
        "run_manifest.json",
    )

    files, mb, last = s3_prefix_metrics(
        s3_settings,
        "bronze",
        "manifests",
        "run_date=2026-01-01",
        "run_manifest.json",
    )
    assert files == 1
    assert mb > 0
    assert last is not None
