"""
Pytest fixtures for nmem-sym-sensor.

Mirrors nmem/nmem-sym: unit tests run anywhere; integration tests need
PostgreSQL + pgvector and are marked ``integration`` (deselect with
``-m "not integration"``).

Point ``NMEM_SENSOR_TEST_DSN`` at a THROWAWAY database — the ``clean_pool``
fixture DROPs and recreates the ``public`` schema on every use, so it must never
touch a live agent DB (e.g. michelle_ai). If no test DB is reachable the
integration tests skip rather than fail. The default DSN targets the disposable
container described in ``tests/README.md``.
"""
from __future__ import annotations

import os
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

# Path to the shipped migrations, resolved without importing the (heavy) package.
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "src" / "nmem_sym_sensor" / "migrations"

DEFAULT_TEST_DSN = "postgresql://sensor:sensor@127.0.0.1:5435/sensor_test"
TEST_DSN = os.environ.get("NMEM_SENSOR_TEST_DSN", DEFAULT_TEST_DSN)


@pytest_asyncio.fixture
async def clean_pool():
    """An asyncpg pool on a PRISTINE schema.

    Drops + recreates ``public`` and installs the ``vector`` extension, so each
    test starts from an empty database. DESTRUCTIVE — only ever run against a
    throwaway DB. Skips the test when no test Postgres is reachable.
    """
    try:
        pool = await asyncpg.create_pool(TEST_DSN, min_size=1, max_size=4, timeout=5)
    except Exception as e:  # noqa: BLE001 — any connect failure means "no test DB"
        pytest.skip(f"no test Postgres at {TEST_DSN}: {e}")

    try:
        async with pool.acquire() as conn:
            await conn.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        yield pool
    finally:
        await pool.close()
