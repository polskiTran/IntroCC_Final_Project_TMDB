"""Async TMDB HTTP client with rate limiting and retry."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from types import TracebackType
from typing import Any, Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings


class TMDBError(RuntimeError):
    """Raised when TMDB responds with a non-retryable error."""


class _RateLimiter:
    """Simple sliding-window limiter: at most `rate` calls per second."""

    def __init__(self, rate: int) -> None:
        self._rate = rate
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._rate <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            while self._window and now - self._window[0] >= 1.0:
                self._window.popleft()
            if len(self._window) >= self._rate:
                sleep_for = 1.0 - (now - self._window[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while self._window and now - self._window[0] >= 1.0:
                    self._window.popleft()
            self._window.append(time.monotonic())


class TMDBClient:
    """Minimal async TMDB client covering the endpoints we need."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._limiter = _RateLimiter(settings.requests_per_second)
        self._semaphore = asyncio.Semaphore(settings.concurrency)

        headers: dict[str, str] = {"Accept": "application/json"}
        params: dict[str, str] = {}
        if settings.tmdb_bearer_token:
            headers["Authorization"] = f"Bearer {settings.tmdb_bearer_token}"
        elif settings.tmdb_api_key:
            params["api_key"] = settings.tmdb_api_key
        else:
            raise TMDBError(
                "No TMDB credentials: set TMDB_BEARER_TOKEN or TMDB_API_KEY."
            )

        self._client = httpx.AsyncClient(
            base_url=settings.tmdb_base_url,
            headers=headers,
            params=params,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def _get(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type((httpx.HTTPError, _Retryable)),
            reraise=True,
        ):
            with attempt:
                await self._limiter.acquire()
                async with self._semaphore:
                    resp = await self._client.get(path, params=params)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    await asyncio.sleep(retry_after)
                    raise _Retryable(f"429 rate limited, retry after {retry_after}s")
                if 500 <= resp.status_code < 600:
                    raise _Retryable(f"{resp.status_code} server error")
                if resp.status_code >= 400:
                    raise TMDBError(
                        f"TMDB {resp.status_code} on {path}: {resp.text[:200]}"
                    )
                return resp.json()
        raise TMDBError(f"Unreachable: retry loop exited without result for {path}")

    async def discover_movies(
        self, page: int, *, filters: dict[str, Any]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, **filters}
        return await self._get("/discover/movie", params=params)

    async def movie_details(
        self, movie_id: int, *, append: str = "credits"
    ) -> dict[str, Any]:
        params = {"append_to_response": append} if append else None
        return await self._get(f"/movie/{movie_id}", params=params)


class _Retryable(Exception):
    """Internal marker to trigger tenacity retry without being a TMDBError."""
