-- nmem-sym-sensor migration 005: record the read-back arm on each screen link
--
-- For a CAUSAL A/B of visual memory's behavioral advantage we keep ingest + link-recording ON in
-- both arms (so the same screens accumulate the same measurement data) and toggle ONLY the read-back
-- intervention (the working-memory warning that steers the next proposal), via
-- NMEM_VISUAL_READBACK_ENABLED. Stamping each link with the arm active when it was created lets the
-- metric attribute a revisit's re-fail / recovery to whether the agent was actually warned then —
-- no reliance on remembering switch timestamps.

ALTER TABLE screen_pursuit_links
    ADD COLUMN IF NOT EXISTS readback BOOLEAN NOT NULL DEFAULT TRUE;
