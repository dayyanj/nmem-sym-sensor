"""
Central configuration for nmem-sym-sensor.

All env-driven configuration lives on a single ``pydantic-settings`` model,
``SensorConfig``, exposed as the module-level ``settings`` singleton (mirrors
nmem / nmem-sym). Read new code as ``config.settings.<field>``.

Env vars keep their historical names (mostly ``NMEM_SENSOR_`` + an abbreviated
suffix; ``NMEM_SYM_DB_DSN`` is the one cross-prefix alias). Every legacy
module-level constant (``config.DB_DSN`` …) is preserved below as a thin compat
shim reading off ``settings`` — no call site had to change. Field names mirror
the env suffix; the shim maps them back to the legacy constant name.

Hardcoded engine constants (activation defaults, iconic/short-term decay, split
parameters, rechallenge tuning, node/edge-type sets) stay module-level — they are
not env-tunable by design.
"""
from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _tabula2_default(filename: str) -> str:
    """Default checkpoint path, relative to the package (repo_root/checkpoints/tabula2)."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "tabula2", filename)


class SensorConfig(BaseSettings):
    """All env-tunable nmem-sym-sensor configuration.

    Env var = ``NMEM_SENSOR_`` + the field name upper-cased, except
    ``sym_db_dsn`` which reads ``NMEM_SYM_DB_DSN`` (the symbol graph shares
    nmem-sym's canonical name).
    """

    model_config = SettingsConfigDict(
        env_prefix="NMEM_SENSOR_",
        extra="ignore",
        case_sensitive=False,
        # Let every field be set by its Python name too, not only its env alias —
        # otherwise sym_db_dsn (validation_alias NMEM_SYM_DB_DSN) can't be passed
        # programmatically, unlike the prefix-named fields. Matters for H5's
        # generic provisioning seam constructing SensorConfig with explicit values.
        populate_by_name=True,
    )

    # ── Database ─────────────────────────────────────────────
    db_dsn: str | None = Field(default=None, description="Postgres DSN for the sensory schema; must be set by caller or env.")
    sym_db_dsn: str | None = Field(default=None, validation_alias="NMEM_SYM_DB_DSN", description="nmem-sym Postgres DSN for symbolic predictions + drive events.")

    # ── Embedding ────────────────────────────────────────────
    embed_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Text embedding model for cross-modal label grounding (same as nmem-sym).")
    visual_dim: int = Field(default=512, description="Visual feature embedding dimensions (encoder output).")
    audio_dim: int = Field(default=128, description="Audio feature embedding dimensions (encoder output).")

    # ── Learned encoders (ONNX / JEPA) ───────────────────────
    visual_onnx: str | None = Field(default=None, description="Path to visual_encoder.onnx; None → hand-crafted embedding.")
    audio_onnx: str | None = Field(default=None, description="Path to audio_encoder.onnx; None → hand-crafted embedding.")
    jepa_onnx: str | None = Field(default=None, description="Path to jepa_v2.onnx (preferred dual-stream encoder).")
    jepa_checkpoint: str | None = Field(default=None, description="Path to jepa_v2_best.pt (PyTorch fallback).")
    visual_input_size: int = Field(default=128, description="Input resolution for the learned visual encoder.")
    encoder_device: str = Field(default="cuda", description="ONNX inference device: cpu, cuda, or auto.")

    # ── Adaptive acuity ──────────────────────────────────────
    acuity_enabled: bool = Field(default=False, description="Two-pass visual processing: structure extraction → adaptive tile encoding.")
    acuity_grid_size: int = Field(default=64)
    acuity_edge_low: int = Field(default=30)
    acuity_edge_high: int = Field(default=100)

    # ── Foveal attention ─────────────────────────────────────
    foveal_enabled: bool = Field(default=False, description="Biologically-inspired saliency-focused attention.")
    familiarity: bool = Field(default=True, description="Attention controller + scene-awareness (cluster cache).")
    foveal_motion_weight: float = Field(default=0.8)
    foveal_edge_weight: float = Field(default=0.2)
    foveal_ema_alpha: float = Field(default=0.8)
    foveal_switch_margin: float = Field(default=0.05)
    foveal_min_dwell: int = Field(default=1)
    foveal_kalman_blend: float = Field(default=0.4)
    foveal_radius_min: int = Field(default=96)
    foveal_radius_max: int = Field(default=256)
    foveal_periphery_scale: float = Field(default=0.25)
    foveal_face_prior: bool = Field(default=False)
    foveal_face_weight: float = Field(default=0.3)
    foveal_flow_downscale: float = Field(default=0.5)

    # ── Streaming / sensor bus ───────────────────────────────
    camera_fps: float = Field(default=2.0)
    audio_window_s: float = Field(default=0.5)
    audio_hop_s: float = Field(default=0.25)
    consolidation_s: float = Field(default=30.0)
    grounding_s: float = Field(default=120.0)
    stt_buffer_s: float = Field(default=5.0)
    camera_queue: int = Field(default=64)
    audio_queue: int = Field(default=128)

    # ── Neuron matching thresholds ───────────────────────────
    neuron_vis_thresh: float = Field(default=0.95, description="Cosine similarity to share a visual neuron.")
    neuron_aud_thresh: float = Field(default=0.90, description="Cosine similarity to share an audio neuron.")

    # ── Visual encoder ───────────────────────────────────────
    visual_min_area: float = Field(default=0.005, description="Min segment area as fraction of image (noise filter).")
    visual_max_segments: int = Field(default=64, description="Max segments per frame (cap graph flooding).")
    visual_color_bins: int = Field(default=8, description="Color quantization bins per channel.")

    # ── Audio encoder ────────────────────────────────────────
    audio_sample_rate: int = Field(default=16000)
    audio_n_mels: int = Field(default=64)
    audio_hop_length: int = Field(default=512)
    audio_onset_threshold: float = Field(default=0.3)
    audio_max_events: int = Field(default=32)

    # ── nmem-sym bridge / grounding ──────────────────────────
    create_symbols: bool = Field(default=False, description="Create new symbol nodes in nmem-sym when grounding finds no match.")
    create_min_obs: int = Field(default=20, description="Min cluster maturity before creating a symbol node.")
    create_min_coherence: float = Field(default=0.7)

    # ── Intelligence loops ───────────────────────────────────
    intelligence: bool = Field(default=True, description="Enable the predict→observe→compare→update cycle.")
    surprise_high: float = Field(default=0.7)
    surprise_low: float = Field(default=0.2)
    dreamstate_every: int = Field(default=5, description="Run dreamstate after every N videos (0 = disabled).")
    dreamstate_duration: float = Field(default=30.0)

    # ── Language acquisition ─────────────────────────────────
    language: bool = Field(default=True, description="Sound-level language learning.")
    sound_threshold: float = Field(default=0.45)
    sound_warmup_threshold: float = Field(default=0.30, description="More aggressive merge threshold during warmup.")
    sound_warmup_until: int = Field(default=50, description="Total units below which warmup merging applies.")
    sound_speaker_norm: bool = Field(default=True, description="Subtract running global centroid to remove speaker bias.")

    # ── Concept linking ──────────────────────────────────────
    concept_threshold: float = Field(default=0.3)
    concept_min_bind: int = Field(default=3)
    self_play_ltp: int = Field(default=2)
    self_play_ltd: int = Field(default=1)

    # ── Speech hierarchy (syllable chunking) ─────────────────
    syllable_min_count: int = Field(default=5)
    syllable_min_conf: float = Field(default=0.5)
    syllable_min_ms: float = Field(default=50)
    syllable_max_ms: float = Field(default=500)
    syllable_max_len: int = Field(default=4)
    syllable_quality: float = Field(default=0.6)

    # ── TABULA2 auditory cortex ──────────────────────────────
    tabula2_enabled: bool = Field(default=False, description="Voice/noise separation, learned VAD, syllable-consistent embeddings.")
    tabula2_checkpoint: str = Field(default_factory=lambda: _tabula2_default("distentangler_512_512.pt"))
    tabula2_decoder: str = Field(default_factory=lambda: _tabula2_default("streaming_parity_latest.pt"))
    tabula2_vad_threshold: float = Field(default=0.5)
    tabula2_min_syllable: int = Field(default=10)

    # ── Oculomotor (saccade trajectory memory) ───────────────
    oculomotor: bool = Field(default=True, description="Store eye-movement patterns as motor memory.")
    saccade_threshold: float = Field(default=0.70)
    saccade_max_fix: int = Field(default=8)

    # ── Debug tracing (see debug.py, which keeps its own runtime enable/disable) ──
    debug: bool = Field(default=False, description="Structured subsystem tracing.")
    debug_filter: str = Field(default="", description="Comma-separated subsystems to trace (empty = all).")


# Singleton — reads the environment ONCE at import (preserves the historical
# module-constant semantics: a later os.environ change does not re-derive these).
settings = SensorConfig()


# ══════════════════════════════════════════════════════════════════════════════
# Compat shim — legacy module-level names, sourced from `settings`.
# New code should prefer `config.settings.<field>`; these are kept so the ~78
# existing `config.X` call sites across the repo (and michelle) keep working.
# ══════════════════════════════════════════════════════════════════════════════

# ── Database ──
DB_DSN = settings.db_dsn
SYM_DB_DSN = settings.sym_db_dsn

# ── Embedding ──
EMBED_MODEL = settings.embed_model
EMBED_DIMENSIONS = 384  # hardcoded — matches all-MiniLM-L6-v2
VISUAL_FEATURE_DIM = settings.visual_dim
AUDIO_FEATURE_DIM = settings.audio_dim

# ── Learned encoders ──
VISUAL_ENCODER_ONNX = settings.visual_onnx
AUDIO_ENCODER_ONNX = settings.audio_onnx
JEPA_V2_ONNX = settings.jepa_onnx
JEPA_V2_CHECKPOINT = settings.jepa_checkpoint
VISUAL_ENCODER_INPUT_SIZE = settings.visual_input_size
ENCODER_DEVICE = settings.encoder_device

# ── Adaptive acuity ──
ADAPTIVE_ACUITY_ENABLED = settings.acuity_enabled
ACUITY_GRID_SIZE = settings.acuity_grid_size
ACUITY_EDGE_LOW = settings.acuity_edge_low
ACUITY_EDGE_HIGH = settings.acuity_edge_high

# ── Foveal attention ──
FOVEAL_ATTENTION_ENABLED = settings.foveal_enabled
FAMILIARITY_ENABLED = settings.familiarity
FOVEAL_MOTION_WEIGHT = settings.foveal_motion_weight
FOVEAL_EDGE_WEIGHT = settings.foveal_edge_weight
FOVEAL_EMA_ALPHA = settings.foveal_ema_alpha
FOVEAL_SWITCH_MARGIN = settings.foveal_switch_margin
FOVEAL_MIN_DWELL = settings.foveal_min_dwell
FOVEAL_KALMAN_BLEND = settings.foveal_kalman_blend
FOVEAL_RADIUS_MIN = settings.foveal_radius_min
FOVEAL_RADIUS_MAX = settings.foveal_radius_max
FOVEAL_PERIPHERY_SCALE = settings.foveal_periphery_scale
FOVEAL_FACE_PRIOR = settings.foveal_face_prior
FOVEAL_FACE_WEIGHT = settings.foveal_face_weight
FOVEAL_FLOW_DOWNSCALE = settings.foveal_flow_downscale

# ── Streaming / sensor bus ──
STREAM_CAMERA_FPS = settings.camera_fps
STREAM_AUDIO_WINDOW_S = settings.audio_window_s
STREAM_AUDIO_HOP_S = settings.audio_hop_s
STREAM_CONSOLIDATION_S = settings.consolidation_s
STREAM_GROUNDING_S = settings.grounding_s
STREAM_STT_BUFFER_S = settings.stt_buffer_s
STREAM_CAMERA_QUEUE = settings.camera_queue
STREAM_AUDIO_QUEUE = settings.audio_queue

# ── Neuron matching thresholds ──
NEURON_VISUAL_THRESHOLD = settings.neuron_vis_thresh
NEURON_AUDIO_THRESHOLD = settings.neuron_aud_thresh

# ── Visual encoder ──
VISUAL_BACKEND = "edge"  # hardcoded — Canny edge detection + contour analysis (CPU only)
VISUAL_MIN_SEGMENT_AREA = settings.visual_min_area
VISUAL_MAX_SEGMENTS = settings.visual_max_segments
VISUAL_COLOR_BINS = settings.visual_color_bins

# ── Audio encoder ──
AUDIO_SAMPLE_RATE = settings.audio_sample_rate
AUDIO_N_MELS = settings.audio_n_mels
AUDIO_HOP_LENGTH = settings.audio_hop_length
AUDIO_ONSET_THRESHOLD = settings.audio_onset_threshold
AUDIO_MAX_EVENTS = settings.audio_max_events

# ── Sensory graph (hardcoded) ──
SENSORY_BASE_THRESHOLD = 0.15
SENSORY_MAX_THRESHOLD = 0.7
SENSORY_MAX_HOPS = 2
SENSORY_MAX_ACTIVATED = 30
SENSORY_TIMEOUT_MS = 50
SENSORY_DECAY_FACTOR = 0.5

# ── Iconic memory (hardcoded) ──
ICONIC_DECAY_SECONDS = 5.0
ICONIC_PROMOTION_THRESHOLD = 2
ICONIC_MAX_BUFFER_SIZE = 256

# ── Short-term sensory memory (hardcoded) ──
SHORT_TERM_DECAY_HOURS = 1.0
SHORT_TERM_PROMOTION_THRESHOLD = 10

# ── Consolidation (hardcoded) ──
VISUAL_MERGE_THRESHOLD = 0.85
AUDIO_MERGE_THRESHOLD = 0.80
CROSS_MODAL_BIND_THRESHOLD = 0.7
CONCEPT_FORMATION_MIN_MEMBERS = 5
CONCEPT_FORMATION_MIN_GROUNDEDNESS = 10

# ── Cluster maturation & splitting (hardcoded) ──
SPLIT_COHERENCE_THRESHOLD = 0.5
SPLIT_MIN_MEMBERS = 8
SPLIT_MIN_OBSERVATIONS = 50
SPLIT_TIGHTER_FACTOR = 1.15
SPLIT_MIN_SUBCLUSTER_SIZE = 4
SPLIT_MAX_GENERATION = 3

# ── Cross-modal binding (hardcoded) ──
BINDING_WINDOW_MS = 2000
BINDING_MIN_COOCCURRENCES = 3

# ── nmem-sym bridge / grounding ──
GROUNDING_MIN_CONFIDENCE = 0.6  # hardcoded
GROUNDING_LABEL_SIMILARITY = 0.75  # hardcoded
GROUNDING_CREATE_SYMBOLS = settings.create_symbols
GROUNDING_CREATE_MIN_OBSERVATIONS = settings.create_min_obs
GROUNDING_CREATE_MIN_COHERENCE = settings.create_min_coherence

# ── Intelligence loops ──
INTELLIGENCE_LOOPS_ENABLED = settings.intelligence
SURPRISE_HIGH = settings.surprise_high
SURPRISE_LOW = settings.surprise_low
DREAMSTATE_EVERY_VIDEOS = settings.dreamstate_every
DREAMSTATE_MAX_DURATION_S = settings.dreamstate_duration

# ── Language acquisition ──
LANGUAGE_LEARNING_ENABLED = settings.language
SOUND_SIMILARITY_THRESHOLD = settings.sound_threshold
SOUND_WARMUP_THRESHOLD = settings.sound_warmup_threshold
SOUND_WARMUP_UNTIL = settings.sound_warmup_until
SOUND_SPEAKER_NORMALIZE = settings.sound_speaker_norm

# ── Concept linking ──
CONCEPT_LINK_THRESHOLD = settings.concept_threshold
CONCEPT_LINK_MIN_BINDINGS = settings.concept_min_bind
SELF_PLAY_LTP_BONUS = settings.self_play_ltp
SELF_PLAY_LTD_PENALTY = settings.self_play_ltd

# ── Speech hierarchy ──
SYLLABLE_MIN_CHAIN_COUNT = settings.syllable_min_count
SYLLABLE_MIN_CONFIDENCE = settings.syllable_min_conf
SYLLABLE_MIN_DURATION_MS = settings.syllable_min_ms
SYLLABLE_MAX_DURATION_MS = settings.syllable_max_ms
SYLLABLE_MAX_LENGTH = settings.syllable_max_len
SYLLABLE_PRACTICE_QUALITY_THRESHOLD = settings.syllable_quality

# ── Rechallenge (hardcoded) ──
RECHALLENGE_OBSERVATION_INTERVAL = 25
RECHALLENGE_DECAY_RATE = 0.02
RECHALLENGE_DEMOTION_THRESHOLD = 0.4
RECHALLENGE_DRIFT_THRESHOLD = 0.3
RECHALLENGE_REVOKE_AFTER_FAILURES = 2
RECHALLENGE_MIN_MATURITY = 30

# ── TABULA2 auditory cortex ──
TABULA2_ENABLED = settings.tabula2_enabled
TABULA2_CHECKPOINT = settings.tabula2_checkpoint
TABULA2_DECODER_CHECKPOINT = settings.tabula2_decoder
TABULA2_VAD_THRESHOLD = settings.tabula2_vad_threshold
TABULA2_MIN_SYLLABLE_FRAMES = settings.tabula2_min_syllable
TABULA2_VOICE_DIM = 512  # hardcoded
TABULA2_NOISE_DIM = 512  # hardcoded

# ── Oculomotor ──
OCULOMOTOR_ENABLED = settings.oculomotor
SACCADE_SIMILARITY_THRESHOLD = settings.saccade_threshold
SACCADE_MAX_FIXATIONS = settings.saccade_max_fix
SACCADE_EMBEDDING_DIM = SACCADE_MAX_FIXATIONS * 3  # derived

# ── Debug tracing ──
# debug.py keeps its own runtime toggle (enable()/disable()) initialised from the
# same env vars; these expose the documented `config.DEBUG_ENABLED` single source.
DEBUG_ENABLED = settings.debug
DEBUG_FILTER = settings.debug_filter

# ── Sensory node types (hardcoded) ──
VISUAL_NODE_TYPES: frozenset[str] = frozenset({
    "shape",        # geometric primitive (circle, rectangle, irregular contour)
    "color",        # quantized color region
    "texture",      # surface pattern descriptor
    "spatial_rel",  # spatial relationship between segments (above, inside, adjacent)
    "visual_cluster",  # consolidated group of co-occurring visual primitives
    "visual_object",   # promoted cluster with stable identity
})

AUDIO_NODE_TYPES: frozenset[str] = frozenset({
    "frequency",    # spectral band / pitch cluster
    "rhythm",       # temporal pattern (onset spacing)
    "timbre",       # spectral envelope shape
    "audio_event",  # discrete sound onset/offset
    "audio_cluster",   # consolidated group of co-occurring audio primitives
    "audio_object",    # promoted cluster with stable identity (e.g., "clinking sound")
})

# ── Sensory edge types (hardcoded) ──
SENSORY_EDGE_TYPES: frozenset[str] = frozenset({
    # Spatial (visual)
    "adjacent_to", "contains", "contained_by",
    "above", "below", "left_of", "right_of",
    "overlaps",
    # Compositional
    "part_of", "co_occurs_with", "member_of",
    # Temporal (audio + video)
    "simultaneous", "follows", "precedes",
    # Cross-modal
    "bound_to",         # visual ↔ audio temporal binding
    # Cluster/concept
    "instance_of",      # primitive → cluster
    "prototype_of",     # cluster → best-exemplar primitive
    "grounded_in",      # nmem-sym concept → sensory cluster
})
