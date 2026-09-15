"""Typed-config parity + behaviour (H1).

Guards the SensorConfig refactor: the same legacy env-var names still resolve,
defaults are unchanged, bad values fail fast, and the module-level compat shim
mirrors the singleton. No database required.
"""
from __future__ import annotations

import os

import pytest

from nmem_sym_sensor import config
from nmem_sym_sensor.config import SensorConfig

# Env vars this suite manipulates; cleared so process env can't skew defaults.
_OWNED = [k for k in os.environ if k.startswith("NMEM_SENSOR_")] + ["NMEM_SYM_DB_DSN"]


@pytest.fixture
def clean_env(monkeypatch):
    for k in _OWNED:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_defaults_unchanged(clean_env):
    s = SensorConfig()
    # A representative slice across sections — values must match the pre-refactor constants.
    assert s.db_dsn is None
    assert s.sym_db_dsn is None
    assert s.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.visual_dim == 512
    assert s.audio_dim == 128
    assert s.encoder_device == "cuda"
    assert s.acuity_enabled is False
    assert s.familiarity is True
    assert s.foveal_enabled is False
    assert s.neuron_vis_thresh == 0.95
    assert s.neuron_aud_thresh == 0.90
    assert s.visual_min_area == 0.005
    assert s.intelligence is True
    assert s.language is True
    assert s.sound_speaker_norm is True
    assert s.oculomotor is True
    assert s.saccade_max_fix == 8
    assert s.tabula2_enabled is False
    assert s.tabula2_checkpoint.endswith("distentangler_512_512.pt")


def test_legacy_env_names_resolve(clean_env):
    clean_env.setenv("NMEM_SENSOR_VISUAL_DIM", "256")
    clean_env.setenv("NMEM_SENSOR_NEURON_VIS_THRESH", "0.5")
    clean_env.setenv("NMEM_SENSOR_FOVEAL_ENABLED", "1")
    clean_env.setenv("NMEM_SENSOR_ACUITY_ENABLED", "0")
    clean_env.setenv("NMEM_SENSOR_CREATE_MIN_OBS", "99")
    clean_env.setenv("NMEM_SENSOR_DB_DSN", "postgresql://sensor-db")
    clean_env.setenv("NMEM_SYM_DB_DSN", "postgresql://sym-db")  # cross-prefix alias

    s = SensorConfig()
    assert s.visual_dim == 256
    assert s.neuron_vis_thresh == 0.5
    assert s.foveal_enabled is True
    assert s.acuity_enabled is False
    assert s.create_min_obs == 99
    assert s.db_dsn == "postgresql://sensor-db"
    assert s.sym_db_dsn == "postgresql://sym-db"


def test_constructor_override_by_field_name(clean_env):
    # Every field — including sym_db_dsn (env alias NMEM_SYM_DB_DSN) — must accept
    # programmatic construction by Python name, for H5's generic provisioning seam.
    s = SensorConfig(db_dsn="postgresql://sensor-db", sym_db_dsn="postgresql://sym-db", visual_dim=256)
    assert s.db_dsn == "postgresql://sensor-db"
    assert s.sym_db_dsn == "postgresql://sym-db"
    assert s.visual_dim == 256


def test_zero_one_flags_match_legacy(clean_env):
    # Legacy flags were `== "1"`; "0"/"1" must still map to False/True.
    clean_env.setenv("NMEM_SENSOR_FAMILIARITY", "0")
    clean_env.setenv("NMEM_SENSOR_INTELLIGENCE", "1")
    s = SensorConfig()
    assert s.familiarity is False
    assert s.intelligence is True


def test_bad_value_fails_fast(clean_env):
    clean_env.setenv("NMEM_SENSOR_VISUAL_DIM", "not-an-int")
    with pytest.raises(Exception):  # pydantic ValidationError — clear, not a silent default
        SensorConfig()


def test_shim_mirrors_settings():
    # The module-level compat shim reads off the singleton.
    assert config.VISUAL_FEATURE_DIM == config.settings.visual_dim
    assert config.AUDIO_FEATURE_DIM == config.settings.audio_dim
    assert config.NEURON_VISUAL_THRESHOLD == config.settings.neuron_vis_thresh
    assert config.SYM_DB_DSN == config.settings.sym_db_dsn
    assert config.TABULA2_CHECKPOINT == config.settings.tabula2_checkpoint
    assert config.SACCADE_EMBEDDING_DIM == config.settings.saccade_max_fix * 3


def test_hardcoded_constants_present():
    # Non-env engine constants must remain module-level and unchanged.
    assert config.EMBED_DIMENSIONS == 384
    assert config.VISUAL_BACKEND == "edge"
    assert config.SENSORY_BASE_THRESHOLD == 0.15
    assert config.GROUNDING_MIN_CONFIDENCE == 0.6
    assert config.RECHALLENGE_MIN_MATURITY == 30
    assert config.TABULA2_VOICE_DIM == 512
    assert "shape" in config.VISUAL_NODE_TYPES
    assert "grounded_in" in config.SENSORY_EDGE_TYPES
