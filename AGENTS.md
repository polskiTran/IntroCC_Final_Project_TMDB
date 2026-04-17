# AGENTS.md

This file is the authoritative guide for any coding agent (Claude, Cursor, Codex, Copilot, etc.) working in this repo. Humans should follow it too.

## Golden rules

1. **Package manager is `uv` — never `pip`.**
   - Add deps: `uv add <pkg>`
   - Add dev deps: `uv add --dev <pkg>`
   - Install: `uv sync` (uses `uv.lock`)
   - Run anything: `uv run <cmd>`
   - Never edit `uv.lock` by hand.
2. **Python ≥ 3.12** only. Use modern typing (`list[str]`, `X | None`, `match`).
3. **Everything runs in Docker.** Any command in the README must also work as `docker compose run …`.
4. **Idempotency.** Any data-pipeline stage must be safe to re-run without corrupting downstream state.
5. **Secrets via `.env`** (never commit).

## Lint / type / test

- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Types: `uv run ty check `
- Tests: `uv run pytest`
- New code needs tests. Data-pipeline code needs at least a smoke test with synthetic data.

## Data contracts

- **Bronze**: raw TMDB JSON, gzipped
- **Silver**: typed Parquet, one file per entity
- **Gold**: modeling-ready Parquet 

## Scope constraints (locked)

- Movies only (no TV).
- English only (`original_language == 'en'`).
- Non-adult only (TMDB `adult == False`).
- Release date between 1980 and today.
- `budget >= $100_000` and `revenue > 0` for Gold modeling tables.
- Release window granularity = **month** (not day).

## Commit style

Conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`.

## Don't

- Don't `pip install`.
- Don't touch `uv.lock` manually.
- Don't add TV / adult / non-English media.
- Don't persist TMDB API keys in code or commit `.env`.
- Don't ship commented-out code or print-debugging into main.
- Don't hit TMDB faster than ~40 req/s.
