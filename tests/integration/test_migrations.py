"""Migrations apply on a clean DB, are idempotent, and bootstrap reconciles a
pre-migration (hand-applied) schema.

These lock the appliance provisioning contract: ``SensorGraph.connect()`` runs
``bootstrap_existing`` then ``MigrationRunner.run()`` — a fresh provision applies
all migrations, and an agent whose ``sensory_*`` tables predate the migration
system reconciles without a rebuild.
"""
from __future__ import annotations

import pytest
from nmem.migrate import MigrationRunner

from nmem_sym_sensor.migrate_bootstrap import bootstrap_existing
from tests.conftest import MIGRATIONS_DIR

pytestmark = pytest.mark.integration

_ALL_MIGRATIONS = sorted(MIGRATIONS_DIR.glob("*.sql"))


def _runner(pool) -> MigrationRunner:
    return MigrationRunner(project="nmem-sym-sensor", migrations_dir=MIGRATIONS_DIR, pool=pool)


async def _tables(pool) -> set[str]:
    rows = await pool.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    return {r["table_name"] for r in rows}


async def test_migrations_apply_on_clean_db(clean_pool):
    runner = _runner(clean_pool)
    assert await bootstrap_existing(runner) == 0  # nothing to reconcile on a clean DB

    applied = await runner.run()
    assert len(applied) == len(_ALL_MIGRATIONS)

    tables = await _tables(clean_pool)
    for expected in (
        "sensory_nodes", "scene_snapshots", "screen_pursuit_links",
        "speech_syllables", "syllable_sequences",  # 006 (folded from lazy init)
    ):
        assert expected in tables, f"{expected} missing after migrations"

    # 004 SUPERSEDES 003: chronoception's coarse 8x8-colour scene identity was
    # non-discriminative for UI screenshots (cosine 0.997 between different screens),
    # so 003's scene_pursuit_links is dropped in favour of the dhash-keyed
    # screen_pursuit_links. The old table must be gone after the full chain.
    assert "scene_pursuit_links" not in tables, "003's table should be superseded by 004"

    # 005 added the read-back A/B column to screen_pursuit_links.
    col = await clean_pool.fetchval(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'screen_pursuit_links' AND column_name = 'readback'"
    )
    assert col == "readback"


async def test_run_is_idempotent(clean_pool):
    first = _runner(clean_pool)
    await bootstrap_existing(first)
    await first.run()

    second = _runner(clean_pool)
    assert await second.run() == []  # nothing pending on a second pass
    status = await second.status()
    assert status["pending"] == 0
    assert status["applied"] == len(_ALL_MIGRATIONS)


async def test_bootstrap_reconciles_hand_applied_initial_schema(clean_pool):
    # Simulate a DB where 001 was hand-applied (schema.sql) before migration
    # tracking existed: the sensory schema is present but schema_migrations is empty.
    sql_001 = (MIGRATIONS_DIR / "001_initial_schema.sql").read_text(encoding="utf-8")
    async with clean_pool.acquire() as conn:
        await conn.execute(sql_001)

    runner = _runner(clean_pool)
    assert await bootstrap_existing(runner) == 1  # 001 detected + marked applied

    applied = await runner.run()
    # 001 is skipped (bootstrapped); only the genuinely-new migrations run.
    assert len(applied) == len(_ALL_MIGRATIONS) - 1
    assert all("001_" not in name for name in applied)
    assert (await runner.status())["pending"] == 0
