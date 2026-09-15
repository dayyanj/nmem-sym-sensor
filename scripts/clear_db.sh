#!/usr/bin/env bash
# Truncate all sensory_memory tables for a clean learning restart.
# WARNING: This deletes ALL learned data.
#
# Usage:
#   ./scripts/clear_db.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

: "${NMEM_SENSOR_DB_DSN:?set NMEM_SENSOR_DB_DSN to the sensory DB DSN (no hardcoded credential)}"

echo "This will DELETE all data in the sensory DB ($NMEM_SENSOR_DB_DSN)."
read -p "Are you sure? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

python3 -c "
import asyncio, asyncpg, os

async def main():
    conn = await asyncpg.connect(os.environ['NMEM_SENSOR_DB_DSN'])
    tables = await conn.fetch(\"SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename\")
    for t in tables:
        name = t['tablename']
        await conn.execute(f'TRUNCATE TABLE \"{name}\" CASCADE')
    print(f'Truncated {len(tables)} tables.')
    await conn.close()

asyncio.run(main())
"
