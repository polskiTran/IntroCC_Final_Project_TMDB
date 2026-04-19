"""Medallion layer paths: local ``Path`` vs ``s3://`` URIs and Polars storage options."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import Settings

GOLD_FILENAME = "gold_movies.parquet"
SILVER_PARQUET_NAMES = ("movies.parquet", "cast.parquet", "crew.parquet")


def normalized_s3_prefix(settings: Settings) -> str:
    return settings.s3_prefix.strip().strip("/")


def s3_object_key(settings: Settings, *parts: str) -> str:
    """S3 object key (no ``s3://``, no bucket)."""
    prefix = normalized_s3_prefix(settings)
    segments = [prefix] if prefix else []
    segments.extend(parts)
    return "/".join(segments)


def s3_uri(settings: Settings, *parts: str) -> str:
    key = s3_object_key(settings, *parts)
    return f"s3://{settings.s3_bucket}/{key}"


def _s3_client(settings: Settings) -> Any:
    import boto3

    return boto3.client("s3", region_name=settings.aws_region)


def polars_storage_options(settings: Settings) -> dict[str, str] | None:
    if settings.data_backend != "s3":
        return None
    opts: dict[str, str] = {"region": settings.aws_region}
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    st = os.environ.get("AWS_SESSION_TOKEN")
    if ak:
        opts["aws_access_key_id"] = ak
    if sk:
        opts["aws_secret_access_key"] = sk
    if st:
        opts["aws_session_token"] = st
    return opts


def silver_parquet_ref(
    settings: Settings, name: str
) -> tuple[Path | str, dict[str, str] | None]:
    if settings.data_backend != "s3":
        return settings.silver_dir / name, None
    return s3_uri(settings, "silver", name), polars_storage_options(settings)


def gold_parquet_ref(
    settings: Settings,
) -> tuple[Path | str, dict[str, str] | None]:
    if settings.data_backend != "s3":
        return settings.gold_dir / GOLD_FILENAME, None
    return s3_uri(settings, "gold", GOLD_FILENAME), polars_storage_options(settings)


def s3_object_exists(settings: Settings, *parts: str) -> bool:
    client = _s3_client(settings)
    key = s3_object_key(settings, *parts)
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=settings.s3_bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    return True


def write_gzipped_json_s3(
    settings: Settings,
    *parts: str,
    payload: dict[str, Any],
) -> None:
    import gzip
    import io
    import json

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    key = s3_object_key(settings, *parts)
    _s3_client(settings).put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=buf.getvalue(),
    )


def write_text_s3(settings: Settings, *parts: str, text: str) -> None:
    key = s3_object_key(settings, *parts)
    _s3_client(settings).put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=text.encode("utf-8"),
    )


def _coerce_last_modified(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw
    if isinstance(raw, float):
        return datetime.fromtimestamp(raw, tz=UTC)
    return None


def s3_prefix_metrics(
    settings: Settings, *prefix_parts: str
) -> tuple[int, float, datetime | None]:
    """Object count under prefix, total MB, latest LastModified."""
    from botocore.exceptions import ClientError

    key = s3_object_key(settings, *prefix_parts)
    client = _s3_client(settings)
    bucket = settings.s3_bucket

    try:
        head = client.head_object(Bucket=bucket, Key=key)
        return (
            1,
            int(head["ContentLength"]) / 1_000_000,
            _coerce_last_modified(head["LastModified"]),
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in ("404", "403", "NoSuchKey", "NotFound"):
            raise

    list_prefix = key if key.endswith("/") else f"{key}/"
    paginator = client.get_paginator("list_objects_v2")
    files = 0
    total_bytes = 0
    latest_ts: datetime | None = None
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            obj_key = obj["Key"]
            if obj_key.endswith("/"):
                continue
            files += 1
            total_bytes += int(obj["Size"])
            ts = _coerce_last_modified(obj["LastModified"])
            if ts is not None and (latest_ts is None or ts > latest_ts):
                latest_ts = ts

    return files, total_bytes / 1_000_000, latest_ts
