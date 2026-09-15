"""
Syllable burst segmentation from VAD-gated voice embeddings.

Takes TABULA2's per-frame voice embeddings and VAD mask, extracts
contiguous speech spans as syllable bursts. Each burst gets a mean
embedding suitable for sound unit clustering.

Does NOT cluster or identify — just segments the audio stream into
discrete speech events. Clustering happens downstream via
SoundLanguage.observe_sound() and visual co-occurrence.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SyllableBurst:
    """A contiguous span of speech — a syllable-level audio event."""

    embedding: np.ndarray  # (512,) L2-normalized mean of voice frames
    start_frame: int  # first frame index in the burst
    end_frame: int  # last frame index (exclusive)
    duration_frames: int  # end - start
    energy: float  # mean VAD confidence across the burst
    # Sub-feature centroids for decoder reconstruction
    pitch_embedding: np.ndarray | None = None  # (96,)
    harmonic_embedding: np.ndarray | None = None  # (96,)
    formant_embedding: np.ndarray | None = None  # (80,)
    vad_embedding: np.ndarray | None = None  # (48,)
    spectral_embedding: np.ndarray | None = None  # (192,)
    # Temporal frame sequences for speech production (decoder needs these)
    voice_frames: np.ndarray | None = None  # (T, 512) raw voice frames
    sub_feature_frames: dict | None = None  # {key: (T, D)} raw sub-feature frames


def segment_syllables(
    voice_embeddings: np.ndarray,
    vad_mask: np.ndarray,
    vad_features: np.ndarray,
    min_frames: int = 3,
    merge_gap_frames: int = 2,
    tail_pad_frames: int = 10,
    sub_features: dict[str, np.ndarray] | None = None,
) -> list[SyllableBurst]:
    """Segment VAD-active spans into syllable bursts.

    Contiguous runs of VAD-active frames form syllable candidates.
    Very short gaps between bursts are merged (co-articulation).
    Each burst is extended by tail_pad_frames to capture unvoiced
    trailing consonants (k, s, t) that the VAD misses.
    Each burst gets a mean embedding over its frames.

    Args:
        voice_embeddings: (T, 512) per-frame voice features.
        vad_mask: (T,) bool, True where speech detected.
        vad_features: (T, 48) raw VAD features for energy.
        min_frames: Minimum burst length (discard shorter as noise).
        merge_gap_frames: Merge bursts separated by gaps this short.
        sub_features: Optional dict of (T, D) arrays for pitch, harmonic,
            formant, spectral sub-features. Keys should match TABULA2 output
            names (voice_pitch_features, etc.)

    Returns:
        List of SyllableBurst objects in temporal order.
    """
    if len(vad_mask) == 0:
        return []

    # Find contiguous True spans (run-length encoding)
    spans = _find_true_spans(vad_mask)

    # Merge spans separated by very short gaps (co-articulation)
    if merge_gap_frames > 0:
        spans = _merge_close_spans(spans, merge_gap_frames)

    # Filter short spans
    spans = [(s, e) for s, e in spans if (e - s) >= min_frames]

    # Extend each span by tail_pad_frames to capture unvoiced trailing
    # consonants (k, s, t, p) that fall below the VAD threshold.
    # Clamp to array bounds and don't overlap with the next span.
    T = len(voice_embeddings)
    padded = []
    for i, (s, e) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else T
        new_end = min(e + tail_pad_frames, T, next_start)
        padded.append((s, new_end))
    spans = padded

    # Build bursts
    bursts = []
    for start, end in spans:
        # Mean-pool voice embeddings in this span
        segment = voice_embeddings[start:end]
        mean_emb = segment.mean(axis=0)

        # L2-normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        # VAD energy for this span
        vad_energy = np.abs(vad_features[start:end]).mean()

        # Extract sub-feature means for this burst
        sub_embs = {}
        if sub_features:
            for key, name in [
                ("voice_pitch_features", "pitch_embedding"),
                ("voice_harmonic_features", "harmonic_embedding"),
                ("voice_formant_features", "formant_embedding"),
                ("voice_vad_features", "vad_embedding"),
                ("voice_spectral_features", "spectral_embedding"),
            ]:
                if key in sub_features:
                    feat = sub_features[key][start:end]
                    m = feat.mean(axis=0)
                    n = np.linalg.norm(m)
                    sub_embs[name] = m / n if n > 0 else m

        # Keep raw frame sequences for speech production
        sub_feat_frames = {}
        if sub_features:
            for key in ("voice_pitch_features", "voice_harmonic_features",
                        "voice_formant_features", "voice_vad_features",
                        "voice_spectral_features"):
                if key in sub_features:
                    sub_feat_frames[key] = sub_features[key][start:end].copy()

        bursts.append(SyllableBurst(
            embedding=mean_emb,
            start_frame=start,
            end_frame=end,
            duration_frames=end - start,
            energy=float(vad_energy),
            voice_frames=segment.copy(),
            sub_feature_frames=sub_feat_frames or None,
            **sub_embs,
        ))

    return bursts


def _find_true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True spans in a boolean array.

    Returns list of (start, end) tuples where end is exclusive.
    """
    spans = []
    in_span = False
    start = 0

    for i in range(len(mask)):
        if mask[i] and not in_span:
            in_span = True
            start = i
        elif not mask[i] and in_span:
            in_span = False
            spans.append((start, i))

    if in_span:
        spans.append((start, len(mask)))

    return spans


def _merge_close_spans(
    spans: list[tuple[int, int]],
    max_gap: int,
) -> list[tuple[int, int]]:
    """Merge spans separated by gaps <= max_gap frames."""
    if not spans:
        return []

    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= max_gap:
            # Merge with previous
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged
