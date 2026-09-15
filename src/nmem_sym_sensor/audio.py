"""
Audio encoder: waveform → sensory primitives.

Decomposes audio into spectral/temporal features without classification.
Produces frequency clusters, rhythm patterns, timbre descriptors, and
discrete onset events. The graph learns to associate these with labels
through repeated co-occurrence with text or visual inputs.

Requires only numpy by default. Optional torchaudio backend for higher
quality mel spectrograms and onset detection.
"""
import logging
from dataclasses import dataclass

import numpy as np

from nmem_sym_sensor import config

log = logging.getLogger(__name__)


@dataclass
class AudioPrimitive:
    """A single audio feature extracted from a waveform segment."""
    label: str                          # auto-generated: "mid-freq-burst", "slow-rhythm"
    node_type: str                      # frequency | rhythm | timbre | audio_event
    features: dict                      # modality-specific descriptors
    embedding: list[float] | None = None  # audio feature vector
    time_start: float = 0.0            # seconds from start
    time_end: float = 0.0              # seconds from start
    energy: float = 0.0                # relative energy level


@dataclass
class TemporalRelation:
    """A temporal relationship between two audio events."""
    source_label: str
    target_label: str
    relation: str                       # simultaneous | follows | precedes
    gap_ms: float = 0.0                # time gap in milliseconds


@dataclass
class AudioAnalysis:
    """Complete analysis of an audio window."""
    window_id: str
    primitives: list[AudioPrimitive]
    relations: list[TemporalRelation]
    duration_s: float
    sample_rate: int
    formant_embedding: list[float] | None = None
    has_onset: bool = False       # whether an auditory onset was detected
    onset_energy: float = 0.0     # peak onset energy (0-1)


# ── Spectral analysis (numpy-only) ──────────────────────

_mel_filterbank_cache: dict[tuple, np.ndarray] = {}


def _get_mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int,
) -> np.ndarray:
    """Build a mel filterbank matrix (n_mels, n_fft//2+1).

    Computed once and cached. Triangular filters spaced on the mel scale.
    """
    key = (n_mels, n_fft, sample_rate)
    if key in _mel_filterbank_cache:
        return _mel_filterbank_cache[key]

    n_freqs = n_fft // 2 + 1
    # Mel scale conversion
    low_mel = 0.0
    high_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        # Rising slope
        for j in range(left, center):
            if j < n_freqs and center > left:
                filterbank[i, j] = (j - left) / (center - left)
        # Falling slope
        for j in range(center, right):
            if j < n_freqs and right > center:
                filterbank[i, j] = (right - j) / (right - center)

    _mel_filterbank_cache[key] = filterbank
    return filterbank


def _compute_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int | None = None,
    n_mels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a power spectrogram from a waveform.

    Returns (spectrogram, frequencies, times).
    Uses numpy STFT — no torchaudio dependency required.
    """
    hop = hop_length or config.AUDIO_HOP_LENGTH

    # Mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0) if waveform.shape[0] > 1 else waveform[0]

    # Normalize
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    # STFT via windowed FFT
    n_frames = 1 + (len(waveform) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 1)), np.zeros(n_fft // 2 + 1), np.zeros(1)

    window = np.hanning(n_fft)
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        frame = waveform[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        stft[:, i] = spectrum ** 2  # power spectrum

    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = np.arange(n_frames) * hop / sample_rate

    return stft, frequencies, times


def _stft_to_mel(
    stft: np.ndarray, sample_rate: int, n_mels: int = 64, n_fft: int = 1024,
) -> np.ndarray:
    """Convert a power STFT to a log-mel spectrogram.

    Args:
        stft: (n_freqs, T) power spectrogram from _compute_spectrogram.
        sample_rate: Sample rate used for the STFT.
        n_mels: Number of mel bands (must match encoder training, default 64).
        n_fft: FFT size used (must match _compute_spectrogram).

    Returns:
        (n_mels, T) log-mel spectrogram, ready for the learned encoder.
    """
    fb = _get_mel_filterbank(n_mels, n_fft, sample_rate)  # (n_mels, n_freqs)
    mel = fb @ stft                                        # (n_mels, T)
    return np.log1p(mel).astype(np.float32)


def _detect_onsets(
    spectrogram: np.ndarray,
    times: np.ndarray,
    threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Detect onset times from spectral flux.

    Returns list of (time_seconds, energy) pairs.
    """
    threshold = threshold or config.AUDIO_ONSET_THRESHOLD

    # Spectral flux: sum of positive frequency bin changes
    flux = np.zeros(spectrogram.shape[1])
    for i in range(1, spectrogram.shape[1]):
        diff = spectrogram[:, i] - spectrogram[:, i - 1]
        flux[i] = np.sum(np.maximum(diff, 0))

    # Normalize
    max_flux = np.max(flux)
    if max_flux > 0:
        flux = flux / max_flux

    # Peak picking
    onsets = []
    for i in range(1, len(flux) - 1):
        if flux[i] > threshold and flux[i] > flux[i - 1] and flux[i] >= flux[i + 1]:
            onsets.append((float(times[i]), float(flux[i])))

    return onsets[:config.AUDIO_MAX_EVENTS]


def _extract_frequency_bands(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
) -> list[dict]:
    """Extract dominant frequency bands from the spectrogram.

    Groups the spectrum into perceptual bands and identifies
    which bands carry the most energy.
    """
    # Define frequency bands (roughly perceptual)
    bands = [
        ("sub_bass", 20, 60),
        ("bass", 60, 250),
        ("low_mid", 250, 500),
        ("mid", 500, 2000),
        ("upper_mid", 2000, 4000),
        ("presence", 4000, 6000),
        ("brilliance", 6000, 20000),
    ]

    band_energies = []
    total_energy = np.sum(spectrogram)
    if total_energy == 0:
        return []

    for name, low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        energy = np.sum(spectrogram[mask, :])
        relative = energy / total_energy
        if relative > 0.05:  # at least 5% of total energy
            band_energies.append({
                "band": name,
                "low_hz": low,
                "high_hz": high,
                "relative_energy": round(float(relative), 4),
                "peak_hz": round(float(frequencies[mask][np.argmax(
                    np.mean(spectrogram[mask, :], axis=1)
                )]), 1),
            })

    return sorted(band_energies, key=lambda b: b["relative_energy"], reverse=True)


def _extract_rhythm(onsets: list[tuple[float, float]]) -> dict | None:
    """Extract rhythm descriptor from onset times.

    Computes inter-onset intervals and classifies the tempo.
    """
    if len(onsets) < 3:
        return None

    times = [o[0] for o in onsets]
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    mean_ioi = np.mean(intervals)
    std_ioi = np.std(intervals)
    regularity = 1.0 - min(std_ioi / max(mean_ioi, 0.001), 1.0)

    # Tempo classification
    if mean_ioi > 1.0:
        tempo_class = "very_slow"
    elif mean_ioi > 0.5:
        tempo_class = "slow"
    elif mean_ioi > 0.25:
        tempo_class = "moderate"
    elif mean_ioi > 0.125:
        tempo_class = "fast"
    else:
        tempo_class = "very_fast"

    return {
        "mean_ioi_ms": round(mean_ioi * 1000, 1),
        "std_ioi_ms": round(std_ioi * 1000, 1),
        "regularity": round(regularity, 3),
        "tempo_class": tempo_class,
        "onset_count": len(onsets),
    }


def _compute_timbre(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
) -> dict:
    """Compute timbre descriptor from spectral envelope.

    Captures the "brightness" and "warmth" of the sound without
    identifying what instrument or source it is.
    """
    # Mean spectrum across time
    mean_spectrum = np.mean(spectrogram, axis=1)
    total = np.sum(mean_spectrum)
    if total == 0:
        return {"spectral_centroid": 0.0, "spectral_spread": 0.0, "brightness": "neutral"}

    # Spectral centroid (perceived pitch center)
    centroid = float(np.sum(frequencies * mean_spectrum) / total)

    # Spectral spread (timbral width)
    spread = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * mean_spectrum) / total))

    # Spectral rolloff (frequency below which 85% of energy lies)
    cumsum = np.cumsum(mean_spectrum)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * total)
    rolloff = float(frequencies[min(rolloff_idx, len(frequencies) - 1)])

    # Brightness classification
    if centroid > 4000:
        brightness = "bright"
    elif centroid > 2000:
        brightness = "neutral"
    elif centroid > 500:
        brightness = "warm"
    else:
        brightness = "dark"

    return {
        "spectral_centroid": round(centroid, 1),
        "spectral_spread": round(spread, 1),
        "spectral_rolloff": round(rolloff, 1),
        "brightness": brightness,
    }


# ── Learned encoder (singleton) ──────────────────────────

_learned_audio_encoder = None


def _get_learned_encoder():
    """Lazy-load the ONNX audio encoder if configured."""
    global _learned_audio_encoder
    if _learned_audio_encoder is not None:
        return _learned_audio_encoder

    if not config.AUDIO_ENCODER_ONNX:
        return None

    try:
        from training.export import ONNXEncoder
        _learned_audio_encoder = ONNXEncoder(config.AUDIO_ENCODER_ONNX)
        log.info("Loaded learned audio encoder: %s", config.AUDIO_ENCODER_ONNX)
        return _learned_audio_encoder
    except Exception as e:
        log.warning("Failed to load learned audio encoder, falling back to hand-crafted: %s", e)
        return None


# ── Feature vector construction ──────────────────────────

def _build_audio_embedding(
    band_energies: list[dict],
    timbre: dict,
    rhythm: dict | None,
    spectrogram: np.ndarray | None = None,
) -> list[float]:
    """Build an audio feature vector.

    If a learned ONNX encoder is configured and a spectrogram is provided,
    uses it directly. Otherwise falls back to hand-crafted embedding.

    Args:
        band_energies: Frequency band energy dicts.
        timbre: Timbre descriptor dict.
        rhythm: Rhythm descriptor dict (or None).
        spectrogram: Raw mel spectrogram (n_mels, T). Used by learned encoder.

    Returns:
        List of floats (embed_dim).
    """
    # Try learned encoder first
    if spectrogram is not None:
        encoder = _get_learned_encoder()
        if encoder is not None:
            try:
                embedding = encoder.encode(spectrogram.astype(np.float32))
                return embedding.tolist()
            except Exception as e:
                log.warning("Learned audio encoder failed, using hand-crafted: %s", e)

    # Hand-crafted fallback
    dim = config.AUDIO_FEATURE_DIM
    vec = np.zeros(dim, dtype=np.float32)

    # Band energies (dims 0-13, two per band)
    band_map = {
        "sub_bass": 0, "bass": 2, "low_mid": 4, "mid": 6,
        "upper_mid": 8, "presence": 10, "brilliance": 12,
    }
    for band in band_energies:
        idx = band_map.get(band["band"])
        if idx is not None:
            vec[idx] = band["relative_energy"]
            vec[idx + 1] = band["peak_hz"] / 20000.0  # normalized

    # Timbre (dims 16-23)
    vec[16] = timbre.get("spectral_centroid", 0) / 10000.0
    vec[17] = timbre.get("spectral_spread", 0) / 5000.0
    vec[18] = timbre.get("spectral_rolloff", 0) / 20000.0
    brightness_map = {"dark": 0.0, "warm": 0.33, "neutral": 0.66, "bright": 1.0}
    vec[19] = brightness_map.get(timbre.get("brightness", "neutral"), 0.5)

    # Rhythm (dims 24-31)
    if rhythm:
        vec[24] = min(rhythm.get("mean_ioi_ms", 0) / 2000.0, 1.0)
        vec[25] = rhythm.get("regularity", 0)
        tempo_map = {"very_slow": 0.0, "slow": 0.25, "moderate": 0.5, "fast": 0.75, "very_fast": 1.0}
        vec[26] = tempo_map.get(rhythm.get("tempo_class", "moderate"), 0.5)
        vec[27] = min(rhythm.get("onset_count", 0) / 32.0, 1.0)

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


def _build_formant_embedding(
    mel_spectrogram: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
) -> list[float]:
    """Build a formant-focused embedding from mel spectrogram.

    Focuses on the 300-3400Hz formant region where vowels and consonants
    are distinguished. Captures spectral shape, temporal dynamics, and
    rate-of-change features that cluster sounds by phonemic content
    rather than speaker timbre.

    Applied to ALL audio — the system learns through co-occurrence
    which sounds correspond to which meanings, without an explicit
    speech/non-speech detector.
    """
    dim = config.AUDIO_FEATURE_DIM  # 128
    vec = np.zeros(dim, dtype=np.float32)

    if mel_spectrogram is None or mel_spectrogram.size == 0:
        return vec.tolist()

    # Formant-range mel bands (~300-3400Hz → bands 10-45 for 64-mel @ 16kHz)
    formant_low = max(0, int(n_mels * 0.15))
    formant_high = min(n_mels, int(n_mels * 0.70))
    n_formant = formant_high - formant_low
    formant_bands = mel_spectrogram[formant_low:formant_high, :]

    # Spectral mean per band (phoneme fingerprint — vowels have distinct formant peaks)
    spectral_mean = formant_bands.mean(axis=1)
    sm_max = spectral_mean.max()
    if sm_max > 0:
        spectral_mean = spectral_mean / sm_max

    # Delta features (rate of change — captures consonant transitions like /t/, /k/)
    if formant_bands.shape[1] > 1:
        delta = np.diff(formant_bands, axis=1)
        delta_mean = np.abs(delta).mean(axis=1)
    else:
        delta_mean = np.zeros(n_formant, dtype=np.float32)
    dm_max = delta_mean.max()
    if dm_max > 0:
        delta_mean = delta_mean / dm_max

    # Temporal envelope (energy contour — captures word rhythm/cadence)
    envelope = formant_bands.sum(axis=0)
    env_max = envelope.max()
    if env_max > 0:
        envelope = envelope / env_max

    # Pack into embedding vector
    # dims 0..n_formant-1: spectral mean
    end1 = min(n_formant, dim)
    vec[:end1] = spectral_mean[:end1]

    # dims n_formant..2*n_formant-1: delta mean
    start2 = n_formant
    end2 = min(2 * n_formant, dim)
    vec[start2:end2] = delta_mean[:end2 - start2]

    # dims 2*n_formant..end: envelope statistics
    offset = 2 * n_formant
    if offset < dim:
        vec[offset] = float(envelope.mean()) if len(envelope) > 0 else 0.0
    if offset + 1 < dim:
        vec[offset + 1] = float(envelope.std()) if len(envelope) > 0 else 0.0
    if offset + 2 < dim:
        vec[offset + 2] = float(envelope.max()) if len(envelope) > 0 else 0.0
    if offset + 3 < dim and len(envelope) > 1:
        # Zero-crossing rate of envelope (rhythm proxy)
        zcr = np.sum(np.diff(np.sign(envelope - envelope.mean())) != 0) / len(envelope)
        vec[offset + 3] = float(zcr)
    if offset + 4 < dim and len(envelope) > 2:
        # Energy onset sharpness (how abruptly speech starts)
        vec[offset + 4] = float(np.max(np.diff(envelope)))

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.tolist()


# ── Temporal relation extraction ─────────────────────────

def _extract_temporal_relations(
    primitives: list[AudioPrimitive],
) -> list[TemporalRelation]:
    """Extract temporal relationships between audio events."""
    events = [p for p in primitives if p.node_type == "audio_event"]
    events = sorted(events, key=lambda e: e.time_start)
    relations = []

    for i, a in enumerate(events):
        for b in events[i + 1:]:
            gap = (b.time_start - a.time_end) * 1000  # ms

            if gap < 0:
                # Overlapping
                relations.append(TemporalRelation(
                    a.label, b.label, "simultaneous", gap_ms=0,
                ))
            elif gap < config.BINDING_WINDOW_MS:
                relations.append(TemporalRelation(
                    a.label, b.label, "follows", gap_ms=round(gap, 1),
                ))
                relations.append(TemporalRelation(
                    b.label, a.label, "precedes", gap_ms=round(gap, 1),
                ))

    return relations


# ── Public interface ─────────────────────────────────────

def analyze_audio(
    waveform: np.ndarray,
    sample_rate: int | None = None,
    window_id: str = "window_0",
) -> AudioAnalysis:
    """Analyze an audio waveform and extract audio primitives.

    Args:
        waveform: 1D or 2D numpy array (channels x samples or just samples).
        sample_rate: Sample rate in Hz. Defaults to config.AUDIO_SAMPLE_RATE.
        window_id: Identifier for this audio window.

    Returns:
        AudioAnalysis with primitives and temporal relations.
    """
    sr = sample_rate or config.AUDIO_SAMPLE_RATE

    # Ensure 1D
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0) if waveform.shape[0] <= waveform.shape[1] else waveform[:, 0]

    duration = len(waveform) / sr
    primitives: list[AudioPrimitive] = []

    # Compute spectrogram
    spec, freqs, times = _compute_spectrogram(waveform, sr)

    # Extract frequency bands
    bands = _extract_frequency_bands(spec, freqs)
    for band in bands:
        label = f"{band['band']}-{int(band['peak_hz'])}hz"
        primitives.append(AudioPrimitive(
            label=label,
            node_type="frequency",
            features=band,
            time_start=0.0,
            time_end=duration,
            energy=band["relative_energy"],
        ))

    # Extract timbre
    timbre = _compute_timbre(spec, freqs)
    primitives.append(AudioPrimitive(
        label=f"{timbre['brightness']}-timbre",
        node_type="timbre",
        features=timbre,
        time_start=0.0,
        time_end=duration,
    ))

    # Detect onsets → audio events
    onsets = _detect_onsets(spec, times)
    for i, (onset_time, energy) in enumerate(onsets):
        # Estimate event duration (until next onset or end)
        end_time = onsets[i + 1][0] if i + 1 < len(onsets) else duration

        # Find dominant band at onset time
        if len(times) > 0:
            frame_idx = min(int(onset_time * sr / config.AUDIO_HOP_LENGTH), spec.shape[1] - 1)
            frame_spec = spec[:, frame_idx]
            peak_freq_idx = np.argmax(frame_spec)
            peak_freq = float(freqs[peak_freq_idx])
        else:
            peak_freq = 0.0

        # Classify frequency range for label
        if peak_freq < 250:
            freq_class = "low"
        elif peak_freq < 2000:
            freq_class = "mid"
        else:
            freq_class = "high"

        label = f"{freq_class}-onset-{round(onset_time, 2)}s"
        primitives.append(AudioPrimitive(
            label=label,
            node_type="audio_event",
            features={
                "onset_time": round(onset_time, 4),
                "peak_frequency": round(peak_freq, 1),
                "energy": round(energy, 4),
                "freq_class": freq_class,
            },
            time_start=onset_time,
            time_end=end_time,
            energy=energy,
        ))

    # Extract rhythm
    rhythm = _extract_rhythm(onsets)
    if rhythm:
        primitives.append(AudioPrimitive(
            label=f"{rhythm['tempo_class']}-rhythm",
            node_type="rhythm",
            features=rhythm,
            time_start=0.0,
            time_end=duration,
        ))

    # Build embeddings for all primitives
    # Convert STFT → mel spectrogram for the learned encoder (64 mel bands, not 513 FFT bins)
    # Pad/trim to 32 frames — the ONNX model has a fixed time dimension
    mel_spec = None
    if spec.size > 0:
        mel = _stft_to_mel(spec, sr)  # (64, T)
        target_frames = 32
        if mel.shape[1] < target_frames:
            mel_spec = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])))
        elif mel.shape[1] > target_frames:
            mel_spec = mel[:, :target_frames]
        else:
            mel_spec = mel
    embedding = _build_audio_embedding(bands, timbre, rhythm, spectrogram=mel_spec)
    for p in primitives:
        p.embedding = embedding

    # Formant embedding for SoundLanguage (phonemic clustering)
    formant_emb = _build_formant_embedding(mel_spec, sr) if mel_spec is not None else None

    # Temporal relations
    relations = _extract_temporal_relations(primitives)

    return AudioAnalysis(
        window_id=window_id,
        primitives=primitives,
        relations=relations,
        duration_s=duration,
        sample_rate=sr,
        formant_embedding=formant_emb,
        has_onset=len(onsets) > 0,
        onset_energy=max((e for _, e in onsets), default=0.0),
    )
