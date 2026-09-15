"""
nmem-sym-sensor — Sensory Memory: Visual and Auditory Cognition for AI Agents

Layer 0 companion to nmem-sym. Decomposes visual and audio inputs into
unlabeled primitives, clusters them through repeated observation, and
grounds the resulting concepts to the symbolic graph.

No classification — the system learns what things are through experience.
"""
__version__ = "0.2.0"

from nmem_sym_sensor.api import SensorGraph
from nmem_sym_sensor.audio import AudioAnalysis, AudioPrimitive, TemporalRelation
from nmem_sym_sensor.describe import SensoryDescription
from nmem_sym_sensor.visual import FrameAnalysis, SpatialRelation, VisualPrimitive

__all__ = [
    "SensorGraph",
    "SensoryDescription",
    "FrameAnalysis", "VisualPrimitive", "SpatialRelation",
    "AudioAnalysis", "AudioPrimitive", "TemporalRelation",
]
