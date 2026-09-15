# Audio Processing

Extracts frequency, timbre, rhythm, and onset primitives from audio waveforms. The system's "ears."

## Modules

| Module | Purpose |
|--------|---------|
| `audio.py` | Waveform analysis: STFT, mel spectrogram, frequency band extraction, onset detection, timbre classification. Produces AudioPrimitive objects with 128-dim MoE embeddings |

## Data Flow

```
Raw audio waveform (16kHz)
    |
    v
[audio.py] ── STFT (1024 FFT, 512 hop) ──> power spectrogram
    |
    v
mel filterbank (64 bands) ──> mel spectrogram
    |
    ├── _detect_onsets() ──> audio_event primitives (low/mid/high onset)
    ├── _extract_frequency_bands() ──> frequency primitives (7 perceptual bands)
    ├── _compute_timbre() ──> timbre primitives (warm/bright/dark)
    └── _extract_rhythm() ──> rhythm primitives (tempo, regularity)
    |
    v
MoE Audio Encoder ── 4 expert paths:
    [0] Silence ── minimal processing
    [1] Environmental ── broad spectral
    [2] Music ── harmonic-aware
    [3] Speech ── VoiceDisentangler + TemporalAttention
    |
    v
128-dim embedding per audio window
    |
    v
[graph.py] ── buffer_observation() ──> iconic buffer
```

## Links to Other Components

- **graph.py** (Sensory Graph): audio primitives enter the iconic buffer
- **binding.py** (Cross-Modal Binding): audio nodes bind to simultaneous visual nodes within 2s window
- **language.py** (Sound Language): audio embeddings feed into sound unit clustering for emergent language acquisition
- **vocal_tract.py** (Speech Production): the encoder is used for self-monitoring (re-encode produced sounds to compare against targets)
- **vocoder.py** (Vocoder): Griffin-Lim converts mel spectrograms back to waveforms for speech production
- **text_bridge.py** (Text Bridge): STT runs on the same audio to extract words, which are temporally aligned with audio primitives

## MoE Audio Encoder Architecture

```
Input: mel spectrogram (64 bands x 32 frames)
    |
    v
Router ── classifies into 4 categories
    |
    ├── [Silence] ── minimal conv layers
    ├── [Environmental] ── standard conv path
    ├── [Music] ── harmonic-aware convolutions
    └── [Speech] ── VoiceDisentangler (separates voice from noise)
                    + TemporalAttention (attends to speech frames)
    |
    v
128-dim embedding (L2 normalised)
```

## ONNX Models

- `models/audio_moe_v1.onnx` (145KB + 5.4MB data) — 4-expert MoE, 128-dim output
- `models/audio_decoder_v1.onnx` (3KB + 2.2MB data) — decoder for speech production (128-dim -> 64x32 mel)

## Key Config

```
AUDIO_SAMPLE_RATE = 16000
AUDIO_N_MELS = 64
AUDIO_HOP_LENGTH = 512
AUDIO_FEATURE_DIM = 128
ENCODER_DEVICE = "cpu" | "cuda" | "auto"
```
