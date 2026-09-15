"""Regression: one selection sub-mechanism failing must not abort the cycle.

Locks in commit aa148f4 — ``run_selection_pressure`` guards each of the six
competitive mechanisms independently so the audio/sound co-occurrence path (which
has no data on a visual-only deployment and used to raise ``column "unit_a_id"
does not exist``) can't take down the visual-relevant cluster/label competition
that runs after it.

No database required — the sub-mechanisms are monkeypatched.
"""
from __future__ import annotations

from nmem_sym_sensor import selection

_MECHANISMS = (
    "run_word_competition",
    "prune_word_spread",
    "run_sound_competition",
    "prune_sound_spread",
    "compete_cluster_memberships",
    "compete_labels",
)
_STAT_KEYS = {
    "word_competition",
    "word_spread",
    "sound_competition",
    "sound_spread",
    "cluster_competition",
    "label_competition",
}


async def test_failing_sound_step_does_not_abort_cycle(monkeypatch):
    async def ok(pool):
        return {"ran": True}

    async def boom(pool):
        raise RuntimeError("audio path exploded")

    for name in _MECHANISMS:
        monkeypatch.setattr(selection, name, ok)
    # The sound-spread step blows up (as it did before aa148f4 on visual-only DBs).
    monkeypatch.setattr(selection, "prune_sound_spread", boom)

    stats = await selection.run_selection_pressure(pool=None)

    # Every mechanism is represented — none were skipped by the failure.
    assert set(stats) == _STAT_KEYS
    # The failing step is captured, not propagated.
    assert stats["sound_spread"] == {"error": "audio path exploded"}
    # Steps BEFORE and AFTER the failure both still ran.
    assert stats["word_competition"] == {"ran": True}
    assert stats["cluster_competition"] == {"ran": True}
    assert stats["label_competition"] == {"ran": True}


async def test_all_steps_ok_returns_all_stats(monkeypatch):
    async def ok(pool):
        return {"ran": True}

    for name in _MECHANISMS:
        monkeypatch.setattr(selection, name, ok)

    stats = await selection.run_selection_pressure(pool=None)
    assert set(stats) == _STAT_KEYS
    assert all(v == {"ran": True} for v in stats.values())
