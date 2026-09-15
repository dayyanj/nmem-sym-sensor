# Speech Production

The system's "mouth" — generates waveforms from learned sound representations. Includes self-monitoring (hearing its own output) and a babbling-to-refinement practice loop.

## Modules

| Module | Purpose |
|--------|---------|
| `vocal_tract.py` | Speech production: sound unit embedding -> decoder -> mel spectrogram -> vocoder -> waveform. Self-monitoring re-encodes output to compare against target. Practice loop refines motor programs through self-play |
| `vocoder.py` | Griffin-Lim phase reconstruction: mel spectrogram -> waveform. Pure numpy/scipy, no external dependencies |
| `imagery.py` | Mental imagery: when producing speech, cross-activates visual representations ("I said elephant, I see elephant in my mind's eye") to validate meaning matches intent |

## Production Pipeline

```
Sound unit centroid (128-dim)
  or motor program embedding
       |
       v
[Audio Decoder ONNX] ── 128-dim -> (64, 32) mel spectrogram
       |
       v
[vocoder.py Griffin-Lim] ── mel -> waveform (32 iterations)
       |
       v
Raw waveform (16kHz)
       |
       ├──> Output (save as WAV / play)
       |
       └──> Self-monitoring:
            [Audio Encoder ONNX] ── waveform -> mel -> 128-dim
                  |
                  v
            cosine(re_encoded, target_centroid) = quality score
```

## Motor Programs (Babbling -> Refinement)

```
Initial: use sound unit centroid as production embedding
  quality = -0.08 (random noise level)
       |
       v
Practice loop (self-play):
  1. Perturb embedding with small noise
  2. Decode -> re-encode -> compare to target
  3. Keep perturbation if quality improved
  4. Noise scale shrinks as quality improves (convergence)
       |
       v
After N iterations:
  quality = 0.5+ (recognisable approximation)
       |
       v
Myelinated (quality >= 0.8):
  Motor program is reliable, no further practice needed
```

## Links to Other Components

- **language.py** (Language Acquisition): sound units are the targets for speech production
- **audio.py** (Audio Processing): encoder used for self-monitoring (re-encode produced sounds)
- **imagery.py** (Mental Imagery): produced sounds activate visual representations for meaning validation
- **sensory_bridge.py** (Bridge): communication.success/failure events feed drive system
- **api.py** (Public API): `speak()`, `practice_speaking()` exposed on SensorGraph

## nmem-sym Integration

Communication drive (in nmem-sym/drives.py):
- Builds pressure when system recognises something it can name
- Fires intent -> motor planning -> VocalTract.produce()
- Success -> LTP on motor program + drive satisfaction
- Failure -> LTD on motor program + competence pressure
- Drive is **modular**: only enabled via `NMEM_SYM_COMMUNICATION_DRIVE=1`, doesn't affect nmem-sym when sensor not present

## Database Tables

| Table | Purpose |
|-------|---------|
| `motor_programs` | Refined production embeddings per sound unit, with quality score and myelination status |

## ONNX Models

- `models/audio_decoder_v1.onnx` (3KB + 2.2MB data) — 128-dim -> 64x32 mel spectrogram
- `models/audio_moe_v1.onnx` (145KB + 5.4MB data) — used for self-monitoring (re-encoding)

## Key Config

Speech production is enabled automatically when decoder + encoder ONNX models are present. No explicit config flag — the vocal tract initialises if the models exist.
