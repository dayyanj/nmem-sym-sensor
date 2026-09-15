"""Unit checks on the migration files themselves — no database required.

Pure-filesystem, so this runs in the ``-m "not integration"`` lane and guards
the property the runner relies on: forward-only, contiguous, uniquely-versioned
migrations.
"""
from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "src" / "nmem_sym_sensor" / "migrations"
_VERSION_RE = re.compile(r"^(\d+)_.+\.sql$")


def _versions() -> list[int]:
    return sorted(
        int(m.group(1))
        for p in MIGRATIONS_DIR.glob("*.sql")
        if (m := _VERSION_RE.match(p.name))
    )


def test_migrations_dir_exists_and_nonempty():
    assert MIGRATIONS_DIR.is_dir()
    assert _versions(), "no *.sql migrations found"


def test_migration_versions_are_contiguous_from_one():
    versions = _versions()
    assert versions[0] == 1
    assert versions == list(range(1, len(versions) + 1)), (
        f"migration versions must be contiguous 1..N, got {versions}"
    )


def test_migration_versions_are_unique():
    names = [p.name for p in MIGRATIONS_DIR.glob("*.sql") if _VERSION_RE.match(p.name)]
    versions = [int(_VERSION_RE.match(n).group(1)) for n in names]
    assert len(versions) == len(set(versions)), f"duplicate migration versions in {names}"
