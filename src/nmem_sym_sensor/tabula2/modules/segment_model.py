import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class SegmentPoolingHead(nn.Module):
    def __init__(self, input_dim, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

    def forward(self, frame_embeddings, boundary_mask):
        B, T, D = frame_embeddings.shape
        pooled_segments = []
        segment_lengths = []

        for b in range(B):
            emb = frame_embeddings[b]
            mask = boundary_mask[b]
            boundaries = (mask == 1).nonzero(as_tuple=False).squeeze(-1).tolist()

            if not boundaries or boundaries[-1] != T - 1:
                boundaries.append(T - 1)

            segs = []
            start = 0
            for end in boundaries:
                seg = emb[start:end + 1]
                if self.use_attention:
                    attn_weights = self.attn(seg).squeeze(-1)
                    attn_weights = torch.softmax(attn_weights, dim=0).unsqueeze(1)
                    pooled = torch.sum(attn_weights * seg, dim=0)
                else:
                    pooled = seg.mean(dim=0)
                segs.append(pooled)
                start = end + 1

            pooled_segments.append(torch.stack(segs))
            segment_lengths.append(len(segs))

        padded = pad_sequence(pooled_segments, batch_first=True)
        segment_mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=padded.device)
        for b, sl in enumerate(segment_lengths):
            segment_mask[b, :sl] = 1

        return padded, segment_mask


class SegmentModel(nn.Module):
    def __init__(self, seg_head: nn.Module, input_dim: int, use_attention=False, threshold: float = 0.5):
        super().__init__()
        self.seg_head = seg_head.eval()  # Freeze segmentation head
        for param in self.seg_head.parameters():
            param.requires_grad = False

        self.pool_head = SegmentPoolingHead(input_dim, use_attention)
        self.threshold = threshold

    def forward(self, frame_embeddings):  # (B, T, D)
        with torch.no_grad():
            logits = self.seg_head(frame_embeddings)  # (B, T)
        boundary_mask = (logits > self.threshold).long()  # (B, T)

        segments, segment_mask = self.pool_head(frame_embeddings, boundary_mask)
        return segments, segment_mask
