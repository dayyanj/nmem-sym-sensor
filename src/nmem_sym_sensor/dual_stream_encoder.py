"""Dual-stream visual encoder: geometric (320-dim) + JEPA v2 (192-dim) = 512-dim.

Two complementary streams:
  - Geometric (hand-crafted, 320 dims): Quadrant-aware HoGC curvature,
    border ownership, colour per quadrant. Excels at thin edges, curves,
    corners — structural primitives that need explicit curvature measurement.
  - JEPA v2 (learned, 192 dims): Self-supervised patch prediction with
    CLS token. Excels at filled shapes, colour/texture patterns, and
    holistic appearance that benefits from learned representations.

The 65/35 split (320:192) reflects the architectural insight that geometric
features carry more weight for primitive-level discrimination, while JEPA
adds complementary learned features for appearance and texture.

Inference modes:
  - ONNX (preferred): Both streams run on CPU via ONNX Runtime.
    JEPA v2 checkpoint must be exported to ONNX first.
  - PyTorch fallback: JEPA v2 runs via PyTorch (GPU or CPU).
  - Geometric only: If JEPA is unavailable, falls back to 320-dim
    geometric padded to 512 with zeros. Similarity still works but
    loses the learned appearance signal.

Usage:
    encoder = DualStreamEncoder()
    embedding = encoder.encode(crop_bgr)  # np.ndarray (512,)
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from nmem_sym_sensor import config
from nmem_sym_sensor.geometric_encoder import GeometricEncoder

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512
GEOMETRIC_DIM = 320
JEPA_DIM = 192  # projected from JEPA v2's native 256-dim CLS output


class JEPAStream:
    """JEPA v2 inference — ONNX or PyTorch."""

    def __init__(self):
        self._onnx_session = None
        self._torch_model = None
        self._projection = None  # 256 -> 192 linear projection weights
        self._ready = False
        self._input_size = config.VISUAL_ENCODER_INPUT_SIZE  # 128

    def _try_load_onnx(self, path: str) -> bool:
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if config.ENCODER_DEVICE == 'cpu':
                providers = ['CPUExecutionProvider']
            self._onnx_session = ort.InferenceSession(path, providers=providers)
            self._ready = True
            log.info("JEPA v2 loaded via ONNX: %s", path)
            return True
        except Exception as e:
            log.debug("ONNX load failed for %s: %s", path, e)
            return False

    def _try_load_torch(self, path: str) -> bool:
        try:
            import torch

            from nmem_sym_sensor.patch_jepa_v2 import PatchJEPAv2

            device = 'cpu'
            if config.ENCODER_DEVICE == 'cuda' and torch.cuda.is_available():
                device = 'cuda'

            model = PatchJEPAv2(
                img_size=self._input_size,
                output_dim=256,
            )
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
            self._torch_model = model
            self._device = device
            self._ready = True
            log.info("JEPA v2 loaded via PyTorch: %s (device=%s)", path, device)
            return True
        except Exception as e:
            log.debug("PyTorch load failed for %s: %s", path, e)
            return False

    def load(self, onnx_path: str | None = None, torch_path: str | None = None) -> bool:
        """Load JEPA model. Tries ONNX first, then PyTorch."""
        if onnx_path and Path(onnx_path).exists():
            if self._try_load_onnx(onnx_path):
                return True
        if torch_path and Path(torch_path).exists():
            if self._try_load_torch(torch_path):
                return True
        log.warning("JEPA v2 not available — dual-stream will use geometric-only")
        return False

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Encode a BGR crop to 192-dim JEPA embedding.

        Returns None if JEPA is not loaded.
        """
        if not self._ready:
            return None

        # Preprocess: resize, BGR->RGB, HWC uint8 -> CHW float32 [0,1]
        resized = cv2.resize(crop_bgr, (self._input_size, self._input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)

        if self._onnx_session is not None:
            return self._encode_onnx(tensor)
        else:
            return self._encode_torch(tensor)

    def _encode_onnx(self, tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference, project 256 -> 192."""
        input_name = self._onnx_session.get_inputs()[0].name
        batch = tensor[np.newaxis]  # (1, 3, H, W)
        outputs = self._onnx_session.run(None, {input_name: batch})
        emb_256 = outputs[0][0]  # (256,)
        return self._project(emb_256)

    def _encode_torch(self, tensor: np.ndarray) -> np.ndarray:
        """Run PyTorch inference, project 256 -> 192."""
        import torch
        with torch.no_grad():
            x = torch.from_numpy(tensor).unsqueeze(0).to(self._device)
            emb_256 = self._torch_model.encode(x)  # (1, 256)
            emb_256 = emb_256.cpu().numpy()[0]
        return self._project(emb_256)

    def _project(self, emb_256: np.ndarray) -> np.ndarray:
        """Project 256-dim CLS output to 192-dim via PCA-style truncation.

        The first 192 dims of the L2-normalised 256-dim vector retain ~90%
        of variance for our trained JEPA v2. Simple truncation outperforms
        random projection for ordered feature dimensions (CLS token output
        from LayerNorm already concentrates variance in early dims).
        """
        # L2 normalise the 256-dim first
        norm = np.linalg.norm(emb_256)
        if norm > 1e-8:
            emb_256 = emb_256 / norm
        # Truncate to 192
        emb_192 = emb_256[:JEPA_DIM].astype(np.float32)
        # Re-normalise
        norm = np.linalg.norm(emb_192)
        if norm > 1e-8:
            emb_192 = emb_192 / norm
        return emb_192


class DualStreamEncoder:
    """Combined geometric + JEPA visual encoder producing 512-dim embeddings.

    Automatically selects GPU geometric encoder when CUDA is available,
    falling back to CPU encoder otherwise.
    """

    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(
        self,
        jepa_onnx_path: str | None = None,
        jepa_torch_path: str | None = None,
        force_cpu: bool = False,
    ):
        # Select geometric encoder: GPU (PyTorch) or CPU (numpy/OpenCV)
        self._gpu_geometric = None
        if not force_cpu:
            self._gpu_geometric = self._try_load_gpu_geometric()

        if self._gpu_geometric is None:
            self.geometric = GeometricEncoder()
            self._geo_mode = 'cpu'
        else:
            self.geometric = None  # unused when GPU is active
            self._geo_mode = 'gpu'

        self.jepa = JEPAStream()

        # Resolve paths from config if not provided
        if jepa_onnx_path is None:
            jepa_onnx_path = config.VISUAL_ENCODER_ONNX
        if jepa_torch_path is None:
            jepa_torch_path = getattr(config, 'JEPA_V2_CHECKPOINT', None)

        self._jepa_available = self.jepa.load(jepa_onnx_path, jepa_torch_path)

    @staticmethod
    def _try_load_gpu_geometric():
        """Try to initialise GPU geometric encoder."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            from nmem_sym_sensor.geometric_encoder_gpu import GeometricEncoderGPU
            enc = GeometricEncoderGPU().cuda().eval()
            # Smoke test
            with torch.no_grad():
                dummy = torch.randn(1, 3, 64, 64, device='cuda')
                out = enc(dummy)
                assert out.shape == (1, 320)
            log.info("Geometric encoder: GPU (CUDA)")
            return enc
        except Exception as e:
            log.debug("GPU geometric encoder unavailable: %s", e)
            return None

    @property
    def has_jepa(self) -> bool:
        return self._jepa_available

    @property
    def geo_mode(self) -> str:
        return self._geo_mode

    @property
    def embedder_id(self) -> str:
        """Stable id of the VECTOR SPACE this encoder produces. Encodes JEPA presence because
        JEPA-on (320 geo + 192 JEPA) vs geometric-only (320 padded to 512) are non-comparable
        512-vecs. Rows tagged with different embedder_ids must never be cosine-compared — mirrors
        nmem-identity's model_id discipline so multiple embedder generations can coexist safely."""
        return f"dualstream_geo320_{'jepa192v2' if self._jepa_available else 'nojepa'}"

    def _encode_geometric(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Encode geometric features via GPU or CPU path."""
        if self._gpu_geometric is not None:
            return self._gpu_geometric.encode_numpy(crop_bgr)
        return self.geometric.encode(crop_bgr)

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Encode a BGR crop to a 512-dim embedding.

        Args:
            crop_bgr: (H, W, 3) uint8 BGR image.

        Returns:
            (512,) float32 L2-normalised embedding.
            First 320 dims = geometric, last 192 dims = JEPA (or zeros).
        """
        # Geometric stream
        geo = self._encode_geometric(crop_bgr)  # (320,)

        # JEPA stream
        jepa = self.jepa.encode(crop_bgr)  # (192,) or None

        # Combine
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        embedding[:GEOMETRIC_DIM] = geo

        if jepa is not None:
            embedding[GEOMETRIC_DIM:GEOMETRIC_DIM + JEPA_DIM] = jepa

        # Final L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def encode_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Encode multiple crops. Returns (N, 512) array."""
        return np.stack([self.encode(c) for c in crops])
