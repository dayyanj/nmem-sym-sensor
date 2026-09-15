# TABULA2/modules/instinctive_head.py

import torch.nn as nn


class InstinctiveHead(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=128, output_dim=3):
        """
        output_dim: 3 for [threat, friendly, neutral/other]
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # multi-label
        )

    def forward(self, frames):
        """
        frames: (T, frame_len) — raw waveform chunks
        Returns: (T, 3) — instinct scores per frame
        """
        return self.model(frames)
