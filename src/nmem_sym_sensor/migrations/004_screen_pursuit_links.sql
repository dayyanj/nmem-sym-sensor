-- nmem-sym-sensor migration 004: screen (perceptual-hash) ↔ pursuit outcome links
--
-- Supersedes 003's scene_pursuit_links. Measurement showed chronoception's scene identity — a
-- coarse 8x8 COLOR embedding — is non-discriminative for UI/desktop SCREENSHOTS: two genuinely
-- different screens score cosine 0.997 (even at 32x32), because screenshots are dominated by large
-- uniform (white/grey) regions, so every screen collapses into one scene and the read-back "warns"
-- indistinguishably. The perceptual dhash we already compute per keyframe is a gradient/structure
-- hash built for exactly this: different UI screens measure Hamming 18–40 apart, same-screen ≤4.
-- So screen identity for the visual-memory read-back is keyed on the dhash, with Hamming matching.
--
-- phash is stored as TEXT (16-hex, the sandbox's dhash format) to avoid unsigned-64 → signed BIGINT
-- overflow; matching is a bounded Python-side Hamming scan over recent same-agent links.

DROP TABLE IF EXISTS scene_pursuit_links;

CREATE TABLE IF NOT EXISTS screen_pursuit_links (
    id          BIGSERIAL PRIMARY KEY,
    phash       TEXT NOT NULL,                       -- perceptual dhash of the screen (16-hex, 64-bit)
    agent_id    TEXT NOT NULL,
    goal_id     BIGINT,                              -- the pursuit's goal (nmem-sym goal id); NULL if unknown
    verdict     TEXT NOT NULL,                       -- 'v' = verified/success, 'f' = failed/unverified
    objective   TEXT,                                -- short objective text, for a human-legible warning
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Read-back scans recent same-agent links (optionally filtered to failures) and Hamming-compares.
CREATE INDEX IF NOT EXISTS idx_screen_pursuit_links_agent ON screen_pursuit_links (agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_screen_pursuit_links_agent_verdict ON screen_pursuit_links (agent_id, verdict);
