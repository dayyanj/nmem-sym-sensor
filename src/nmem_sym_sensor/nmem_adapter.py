"""
nmem write adapter for sensory STT labels.

Provides the nmem_write_fn callback that text_bridge.py expects,
writing speech-extracted labels to nmem's journal table.

Zero import coupling with nmem — uses direct SQL against the nmem DB.
All configuration via environment variables:
  NMEM_SENSOR_NMEM_DB_DSN: Connection string for the nmem database
  NMEM_SENSOR_NMEM_AGENT_ID: Agent ID for journal entries (default: "sensory")
  NMEM_SENSOR_NMEM_ENABLED: Set to "1" to enable (default: "0")

Usage:
    adapter = NmemWriteAdapter(db_dsn="postgresql://...")
    await adapter.connect()

    callback = create_text_callback(
        nmem_write_fn=adapter.write,
        sensor_graph=sg,
    )
    # Now STT labels flow into nmem journal entries

    await adapter.close()
"""
import json
import logging
import os

import asyncpg

log = logging.getLogger(__name__)

# Configuration
NMEM_DB_DSN = os.environ.get("NMEM_SENSOR_NMEM_DB_DSN")
NMEM_AGENT_ID = os.environ.get("NMEM_SENSOR_NMEM_AGENT_ID", "sensory")
NMEM_ENABLED = os.environ.get("NMEM_SENSOR_NMEM_ENABLED", "0") == "1"


class NmemWriteAdapter:
    """Writes sensory STT labels to nmem's journal table.

    Implements the nmem_write_fn interface expected by create_text_callback:
        async def write(content: str, metadata: dict) -> None
    """

    def __init__(
        self,
        db_dsn: str | None = None,
        agent_id: str | None = None,
    ):
        self.db_dsn = db_dsn or NMEM_DB_DSN
        self.agent_id = agent_id or NMEM_AGENT_ID
        self._pool: asyncpg.Pool | None = None

    async def connect(self):
        """Connect to the nmem database. Respects NMEM_ENABLED flag."""
        if not NMEM_ENABLED:
            log.info("nmem adapter: disabled (set NMEM_SENSOR_NMEM_ENABLED=1 to enable)")
            return
        if not self.db_dsn:
            log.warning("nmem adapter: no DB DSN configured, writes will be no-ops")
            return

        try:
            self._pool = await asyncpg.create_pool(self.db_dsn, min_size=1, max_size=2)
            log.info("nmem adapter: connected to %s",
                     self.db_dsn.split("@")[1] if "@" in self.db_dsn else "nmem DB")
        except Exception as e:
            log.warning("nmem adapter: connection failed: %s", e)
            self._pool = None

    async def close(self):
        """Cleanup."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def write(self, content: str, metadata: dict) -> None:
        """Write a speech-extracted label to nmem's journal.

        This implements the nmem_write_fn interface.

        Args:
            content: The extracted label/concept (e.g., "red ball").
            metadata: Source metadata from text_bridge:
                - source: "stt"
                - original_text: full sentence
                - timestamp: float seconds
                - event_type: "naming" | "mention" | "description"
                - confidence: 0-1
        """
        if not self._pool:
            return

        event_type = metadata.get("event_type", "mention")
        confidence = metadata.get("confidence", 0.5)
        original_text = metadata.get("original_text", "")

        # Map confidence to importance (1-10 scale for nmem)
        importance = max(1, min(10, int(confidence * 10)))

        # Naming events are more important than mentions
        if event_type == "naming":
            importance = min(10, importance + 2)

        title = f"Sensory: {content}"
        journal_content = (
            f"Heard speech label: \"{content}\"\n"
            f"Original text: \"{original_text}\"\n"
            f"Event type: {event_type}\n"
            f"Confidence: {confidence:.2f}"
        )

        tags = json.dumps(["sensory", "stt", event_type])

        try:
            await self._pool.execute(
                """
                INSERT INTO nmem_journal_entries
                    (agent_id, title, content, importance, entry_type, record_type, tags, grounding)
                VALUES ($1, $2, $3, $4, $5, 'observation', $6, 'inferred')
                """,
                self.agent_id,
                title,
                journal_content,
                importance,
                "observation",  # entry_type
                tags,
            )
            log.debug("nmem journal: wrote '%s' (importance=%d)", content, importance)
        except Exception as e:
            log.warning("nmem journal write failed for '%s': %s", content, e)
