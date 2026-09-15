"""
TABULA2 auditory cortex — biological audio processing.

Provides voice/noise disentanglement, VAD, and syllable-level
embeddings for the sensory memory system.
"""
from nmem_sym_sensor.tabula2.wrapper import TABULA2Disentangler, TABULA2Output

__all__ = ["TABULA2Disentangler", "TABULA2Output"]
