import torch.nn as nn


class SpeechNoiseDisentangler(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=256, emb_dim=128):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Separate projection heads for voice and noise
        self.voice_proj = nn.Linear(hidden_dim, emb_dim)
        self.noise_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, frames):
        """
        frames: (T, input_dim) — raw audio frames
        Returns:
            voice_emb: (T, emb_dim)
            noise_emb: (T, emb_dim)
        """
        shared = self.shared(frames)
        voice_emb = self.voice_proj(shared)
        noise_emb = self.noise_proj(shared)
        return voice_emb, noise_emb
