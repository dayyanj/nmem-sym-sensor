-- nmem-sym-sensor migration 003: Scene ↔ pursuit outcome links
--
-- Visual-memory Phase 2 read-back needs to know, when a whole-screen "scene" recurs, whether that
-- scene previously appeared in a FAILED pursuit — so the agent can be warned "you have been on this
-- screen before and it did not work; try differently." A recognized scene_snapshot carries no
-- outcome by itself, so this table links each scene the agent lands on to the pursuit (goal +
-- verdict) it occurred in. Read-back queries prior 'f' (failed) links for the scenes a new pursuit
-- revisits; the count of revisited-failed scenes is also the Phase-3 behavioral-advantage metric.

CREATE TABLE IF NOT EXISTS scene_pursuit_links (
    id          BIGSERIAL PRIMARY KEY,
    scene_id    BIGINT NOT NULL REFERENCES scene_snapshots(id) ON DELETE CASCADE,
    agent_id    TEXT NOT NULL,
    goal_id     BIGINT,                              -- the pursuit's goal (nmem-sym goal id); NULL if unknown
    verdict     TEXT NOT NULL,                       -- 'v' = verified/success, 'f' = failed/unverified
    objective   TEXT,                                -- short objective text, for a human-legible warning
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scene_pursuit_links_scene ON scene_pursuit_links (scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_pursuit_links_agent_verdict ON scene_pursuit_links (agent_id, verdict);
