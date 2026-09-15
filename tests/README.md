# nmem-sym-sensor tests

Two lanes (mirrors nmem / nmem-sym):

- **unit** — no database. `pytest -m "not integration"`
- **integration** — PostgreSQL + pgvector, marked `integration`.

## Integration DB (throwaway only)

The `clean_pool` fixture **drops and recreates the `public` schema** on every
use, so it must only ever point at a disposable database — never a live agent DB
(e.g. `michelle_ai`).

Spin up a throwaway container:

```bash
docker run -d --name sensor-test-pg \
  -e POSTGRES_USER=sensor -e POSTGRES_PASSWORD=sensor -e POSTGRES_DB=sensor_test \
  -p 127.0.0.1:5435:5432 pgvector/pgvector:pg16
```

Point tests at it (this is also the built-in default):

```bash
export NMEM_SENSOR_TEST_DSN=postgresql://sensor:sensor@127.0.0.1:5435/sensor_test
pytest                    # all tests
pytest -m "not integration"   # unit only (no DB needed)
```

If no test Postgres is reachable, the integration tests **skip** (they don't
fail), so the unit lane is always runnable.

## What's covered (baseline)

- `test_migration_files.py` — migrations are contiguous, uniquely versioned (unit).
- `test_selection_resilience.py` — a failing selection sub-mechanism doesn't abort
  the cycle (locks commit `aa148f4`; unit, sub-mechanisms monkeypatched).
- `integration/test_migrations.py` — migrations apply on a clean DB, are
  idempotent, and `bootstrap_existing` reconciles a hand-applied initial schema.
- `integration/test_graph_embedder_partition.py` — neurons only match within one
  `embedder_id` vector space (tag=tag, NULL=NULL).
