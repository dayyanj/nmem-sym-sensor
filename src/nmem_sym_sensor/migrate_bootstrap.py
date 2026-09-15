"""Bootstrap detection for existing nmem-sym-sensor installations.

When the migration system is introduced to a DB that already has the sensory schema
(e.g. an agent whose `sensory_*` tables were hand-applied from schema.sql before migrations
existed), this detects the effectively-already-applied migrations and marks them, so the
runner only applies genuinely-new ones — no rebuild, no duplicate DDL.

Mirrors nmem-sym's migrate_bootstrap. Called once on first run (schema_migrations empty for
project 'nmem-sym-sensor' but the sensory tables clearly exist).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nmem.migrate import MigrationRunner

log = logging.getLogger(__name__)


async def bootstrap_existing(runner: MigrationRunner) -> int:
    """Detect an existing sensory schema and mark migrations applied. Returns count marked."""
    applied = await runner.applied()
    if applied:
        return 0  # already bootstrapped / under migration control

    all_files = runner.discover()
    if not all_files:
        return 0

    pool = runner._exec._pool  # type: ignore[attr-defined]
    marked = 0

    # 001: the initial sensory schema — detect via sensory_nodes.
    has_nodes = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'sensory_nodes')"
    )
    if has_nodes:
        migration = _find(all_files, 1)
        if migration:
            content = migration.read_text(encoding="utf-8")
            await runner.mark_applied(1, migration.name, content)
            marked += 1
            log.info("Bootstrap: marked 001 (initial sensory schema) as applied")

    if marked:
        log.info("Bootstrap: marked %d existing sensory migration(s) as applied", marked)
    return marked


def _find(files: list[tuple[int, Path]], version: int) -> Path | None:
    for v, p in files:
        if v == version:
            return p
    return None
