"""
Wrapper around TABULA2's DisentanglerModel for inference.

Loads the checkpoint once, provides a clean interface for the
nmem-sym-sensor pipeline. Runs on CPU (17x realtime for 0.5s chunks).
"""
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class TABULA2Output:
    """Disentangled audio output from TABULA2."""

    voice_embeddings: np.ndarray  # (T, 512) per-frame voice features
    noise_embeddings: np.ndarray  # (T, 512) per-frame noise features
    vad_features: np.ndarray  # (T, 48) per-frame VAD confidence
    vad_mask: np.ndarray  # (T,) bool — True where speech detected
    frame_rate: float  # frames per second (typically 200)
    # Sub-features for decoder reconstruction and detailed learning
    sub_features: dict[str, np.ndarray] | None = None  # all voice_* sub-feature arrays (T, D)


class TABULA2Disentangler:
    """TABULA2 auditory cortex for voice/noise separation.

    The disentangler separates audio into two streams:
    - Voice: pitch, harmonics, formants, spectral — speech content
    - Noise: broadband, transients, environmental — non-speech

    Each stream produces 512-dim per-frame embeddings at 200fps.
    The VAD (voice activity detection) identifies speech frames.

    Usage:
        t2 = TABULA2Disentangler("checkpoints/tabula2/disentangler_512_512.pt")
        output = t2.process(waveform_16khz)
        # output.voice_embeddings: (T, 512) — speech content per frame
        # output.vad_mask: (T,) — True where speech detected
    """

    def __init__(
        self,
        checkpoint_path: str,
        decoder_path: str | None = None,
        vad_threshold: float = 0.5,
    ):
        self.vad_threshold = vad_threshold
        self._decoder_path = decoder_path
        self._decoder = None  # lazy-loaded on first decode call

        # Add tabula2 model directory to path for imports
        tabula2_dir = os.path.dirname(os.path.abspath(__file__))
        if tabula2_dir not in sys.path:
            sys.path.insert(0, tabula2_dir)

        from nmem_sym_sensor.tabula2.models.disentangler.disentangler_model import (
            DisentanglerModel,
        )

        self._model = DisentanglerModel(
            use_batch_training=True,
            voice_embedding_dim=512,
            noise_embedding_dim=512,
            enable_reconstruction=True,
            dimension_preset="large",
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt["model_state"], strict=False)
        self._model.eval()
        self._model.enable_reconstruction = False

        # Extract frame rate from model config
        self._frame_rate = 200.0  # 16000 Hz / 80 hop = 200 fps

        log.info(
            "TABULA2 disentangler loaded (step=%s, val_loss=%.4f)",
            ckpt.get("global_step", "?"),
            ckpt.get("best_val_loss", 0),
        )

    @torch.no_grad()
    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> TABULA2Output:
        """Process audio and return disentangled voice/noise streams.

        Args:
            waveform: 1D float32 mono audio (values in [-1, 1]).
            sample_rate: Must be 16000.

        Returns:
            TABULA2Output with per-frame embeddings and VAD mask.
        """
        if sample_rate != 16000:
            raise ValueError(f"TABULA2 requires 16kHz audio, got {sample_rate}")

        # Model expects (batch, channels, samples)
        if isinstance(waveform, np.ndarray):
            tensor = torch.tensor(waveform, dtype=torch.float32)
        else:
            tensor = waveform.float()

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)  # (1, C, T)

        out = self._model(tensor)

        # Extract and transpose: model outputs (B, D, T) → we want (T, D)
        voice = out["voice_embeddings"][0].numpy().T  # (T, 512)
        noise = out["noise_embeddings"][0].numpy().T  # (T, 512)
        vad = out["voice_vad_features"][0].numpy().T  # (T, 48)

        # Collect all voice sub-features for downstream use
        sub_features = {}
        for key in ("voice_pitch_features", "voice_harmonic_features",
                     "voice_formant_features", "voice_vad_features",
                     "voice_spectral_features"):
            if key in out and isinstance(out[key], torch.Tensor):
                sub_features[key] = out[key][0].numpy().T  # (T, D)

        # Derive VAD mask from VAD features (absolute energy threshold)
        # Speech windows have mean VAD energy ~1.8, silence ~0.13.
        # Using absolute threshold avoids the percentile bug where silence
        # windows always produce "speech" frames.
        vad_energy = np.abs(vad).mean(axis=1)  # (T,)
        vad_mask = vad_energy > self.vad_threshold

        return TABULA2Output(
            voice_embeddings=voice,
            noise_embeddings=noise,
            vad_features=vad,
            vad_mask=vad_mask,
            frame_rate=self._frame_rate,
            sub_features=sub_features,
        )

    @torch.no_grad()
    def decode_voice(
        self,
        voice_embedding: np.ndarray,
        n_frames: int = 50,
        sub_features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Reconstruct waveform from voice embedding.

        Args:
            voice_embedding: (512,) centroid or (T, 512) frame sequence.
            n_frames: Number of temporal frames to synthesize (if input is 1D).
            sub_features: Optional dict of sub-feature centroids from sound_units:
                pitch (96), harmonic (96), formant (80), vad (48), spectral (192).
                Keys: 'pitch', 'harmonic', 'formant', 'vad', 'spectral'.

        Returns:
            float32 mono waveform at 16kHz, or None if decoder not loaded.
        """
        if self._decoder is None:
            if self._decoder_path and os.path.exists(self._decoder_path):
                self._load_decoder()
            if self._decoder is None:
                return None

        emb = torch.tensor(voice_embedding, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).expand(n_frames, -1)  # (T, 512)
        emb = emb.unsqueeze(0)  # (1, T, 512)

        # Map short keys to decoder conditioner keys
        feature_key_map = {
            "pitch": "voice_pitch_features",
            "harmonic": "voice_harmonic_features",
            "formant": "voice_formant_features",
            "vad": "voice_vad_features",
            "spectral": "voice_spectral_features",
        }

        try:
            # Actual temporal length (from embedding, not n_frames param)
            t_len = emb.shape[1]

            voice_input = {"voice_embeddings": emb}

            if sub_features:
                # Use stored sub-features — expand centroids or keep sequences
                for short_key, decoder_key in feature_key_map.items():
                    if short_key in sub_features and sub_features[short_key] is not None:
                        feat = torch.tensor(sub_features[short_key], dtype=torch.float32)
                        if feat.dim() == 1:
                            feat = feat.unsqueeze(0).expand(t_len, -1)  # (T, D)
                        feat = feat.unsqueeze(0)  # (1, T, D)
                        voice_input[decoder_key] = feat

            # Fill any missing sub-features with zeros
            dim_defaults = {
                "voice_pitch_features": 96,
                "voice_harmonic_features": 96,
                "voice_formant_features": 80,
                "voice_vad_features": 48,
                "voice_spectral_features": 192,
            }
            for key, dim in dim_defaults.items():
                if key not in voice_input:
                    voice_input[key] = torch.zeros(1, t_len, dim)

            outputs = self._decoder({"voice": voice_input})
            waveform = outputs.get("voice_waveform")
            if waveform is not None:
                return waveform[0].numpy()  # (samples,)
        except Exception as e:
            log.debug("Decoder failed: %s", e)

        return None

    @torch.no_grad()
    def decode_noise(self, noise_embedding: np.ndarray, n_frames: int = 50) -> np.ndarray | None:
        """Reconstruct environmental sound from noise embedding.

        The noise stream captures non-speech: water, birds, music, doors.
        """
        if self._decoder is None:
            if self._decoder_path and os.path.exists(self._decoder_path):
                self._load_decoder()
            if self._decoder is None:
                return None

        emb = torch.tensor(noise_embedding, dtype=torch.float32)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0).expand(n_frames, -1)
        emb = emb.unsqueeze(0)

        try:
            silence = torch.zeros(1, 1, n_frames * 80)
            template = self._model(silence)

            noise_input = {}
            for k, v in template.items():
                if k.startswith("noise_") and isinstance(v, torch.Tensor):
                    if k == "noise_embeddings":
                        noise_input[k] = emb
                    else:
                        feat = v[0, :, :n_frames].unsqueeze(0).permute(0, 2, 1)
                        noise_input[k] = feat

            outputs = self._decoder({"noise": noise_input})
            waveform = outputs.get("noise_waveform")
            if waveform is not None:
                return waveform[0].numpy()
        except Exception as e:
            log.debug("Noise decoder failed: %s", e)

        return None

    def _load_decoder(self):
        """Load the decoder model for audio reconstruction."""
        if not self._decoder_path or not os.path.exists(self._decoder_path):
            return

        try:
            from nmem_sym_sensor.tabula2.models.decoder.enhanced_decoder import (
                EnhancedMultiDecoderSystem,
            )

            self._decoder = EnhancedMultiDecoderSystem(
                signal_types=["voice", "noise"],
                d_hid=512,
                n_layers=6,
                n_basis=160,
                frame_len=400,
                hop=80,
            )

            ckpt = torch.load(
                self._decoder_path, map_location="cpu", weights_only=False,
            )
            sd = ckpt.get("model_state_dict", ckpt)

            # Pre-initialize conditioner layers from checkpoint dimensions
            # This is critical — conditioners use dynamic init and would
            # silently discard trained weights without this step.
            self._decoder.initialize_conditioners_from_checkpoint(sd)

            # Now load all weights — both voice and noise decoders
            missing, unexpected = self._decoder.load_state_dict(sd, strict=False)
            self._decoder.eval()

            if missing:
                log.warning("Decoder: %d missing keys", len(missing))

            log.info("TABULA2 decoder loaded (voice+noise, %d params)",
                     sum(p.numel() for p in self._decoder.parameters()))
        except Exception as e:
            log.warning("Failed to load TABULA2 decoder: %s", e)
            self._decoder = None
