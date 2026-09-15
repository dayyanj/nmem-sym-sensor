"""
Video ingestion pipeline for sensory learning.

Decomposes video into temporally-aligned visual frames, audio windows,
and speech-to-text transcripts. Feeds all three into the sensory graph
with proper temporal binding so that co-occurring observations
(seeing a cup + hearing "cup") ground automatically.
"""
