import logging
import math
from pathlib import Path
from typing import Optional, cast

import torch
from torch import nn

from src.models.config import Tier2SignFormerConfig

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe_tensor: torch.Tensor = cast(torch.Tensor, getattr(self, "pe"))
        return x + pe_tensor[:, : x.size(1), :]


def build_76_keypoint_adjacency() -> torch.Tensor:
    """Build normalized anatomical adjacency matrix for 76-keypoint Holistic topology.

    Nodes 0..20: Left Hand
    Nodes 21..41: Right Hand
    Nodes 42..52: Upper Body Pose (11 nodes)
    Nodes 53..75: Face Non-Manual Markers (23 nodes)
    """
    num_nodes = 76
    adj = torch.eye(num_nodes)

    # Hand connections (wrist to MCPs, fingers)
    def connect_hand(offset: int) -> None:
        # Wrist to finger roots
        for root in [1, 5, 9, 13, 17]:
            adj[offset, offset + root] = 1.0
            adj[offset + root, offset] = 1.0
        # Finger bones
        for finger in range(5):
            base = offset + 1 + finger * 4
            for j in range(3):
                adj[base + j, base + j + 1] = 1.0
                adj[base + j + 1, base + j] = 1.0

    connect_hand(0)  # Left hand
    connect_hand(21)  # Right hand

    # Pose upper-body connections (42..52)
    # 42=nose, 43=l_shoulder, 44=r_shoulder, 45=l_elbow, 46=r_elbow, 47=l_wrist, 48=r_wrist
    pose_edges = [
        (42, 43),
        (42, 44),
        (43, 44),  # Head / shoulders
        (43, 45),
        (45, 47),  # Left arm
        (44, 46),
        (46, 48),  # Right arm
        (47, 0),  # Connect left wrist to left hand
        (48, 21),  # Connect right wrist to right hand
        (43, 49),  # Left shoulder to left hip
        (44, 50),  # Right shoulder to right hip
        (49, 50),  # Left hip to right hip
        (42, 51),  # Nose to left eye inner
        (42, 52),  # Nose to right eye inner
    ]
    for u, v in pose_edges:
        if u < num_nodes and v < num_nodes:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

    # Face connections (53..75) - Anatomically correct sub-graphs
    # Left Eyebrow (53-57)
    for i in range(53, 57):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    adj[42, 53] = 1.0
    adj[53, 42] = 1.0

    # Right Eyebrow (58-62)
    for i in range(58, 62):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    adj[42, 58] = 1.0
    adj[58, 42] = 1.0

    # Left Eye (63-64)
    adj[63, 64] = 1.0
    adj[64, 63] = 1.0
    adj[42, 63] = 1.0
    adj[63, 42] = 1.0

    # Right Eye (65-66)
    adj[65, 66] = 1.0
    adj[66, 65] = 1.0
    adj[42, 65] = 1.0
    adj[65, 42] = 1.0

    # Mouth contour (67-75)
    for i in range(67, 75):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    # Degree normalization: D^(-1/2) * A * D^(-1/2)
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

    return cast(torch.Tensor, norm_adj)


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution over keypoint skeleton."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 76):
        super().__init__()
        self.num_nodes = num_nodes
        self.fc = nn.Linear(in_channels, out_channels)
        self.register_buffer("adj", build_76_keypoint_adjacency())
        self.bn = nn.BatchNorm1d(out_channels * num_nodes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_nodes, in_channels)
        b, t, v, c = x.shape
        x_flat = self.fc(x)  # (b, t, v, out_c)

        adj_buf: torch.Tensor = cast(torch.Tensor, getattr(self, "adj"))
        # Graph aggregation: (v, v) @ (b, t, v, out_c) -> (b, t, v, out_c)
        out = torch.einsum("vw,btwc->btvc", adj_buf, x_flat)
        out = out.reshape(b * t, v * out.shape[-1])
        out = self.bn(out)
        out = self.relu(out)
        out = out.reshape(b, t, v, -1)
        return cast(torch.Tensor, out)


class TemporalGraphConv(nn.Module):
    """Temporal convolution across sequential graph frames."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_nodes, in_channels) -> permute to (batch, in_channels, seq_len, num_nodes)
        x = x.permute(0, 3, 1, 2)
        out = self.relu(self.bn(self.conv(x)))
        # permute back to (batch, seq_len, num_nodes, out_channels)
        out = out.permute(0, 2, 3, 1)
        return cast(torch.Tensor, out)


class STGCNBlock(nn.Module):
    """Fused Spatio-Temporal Graph Convolution Block with residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 76, temporal_kernel: int = 5):
        super().__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels, num_nodes)
        self.tgcn = TemporalGraphConv(out_channels, out_channels, kernel_size=temporal_kernel)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.sgcn(x)
        out = self.tgcn(out)
        return cast(torch.Tensor, out + res)


class EuclideanSelfAttention(nn.Module):
    """Euclidean-distance Self-Attention Layer for motion burst stabilization."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)  # (b, h, t, head_dim)
        k = self.k_proj(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)  # (b, h, t, head_dim)
        v = self.v_proj(x).view(b, t, self.nhead, self.head_dim).transpose(1, 2)  # (b, h, t, head_dim)

        # Squared Euclidean distance: ||q_i - k_j||^2 = ||q_i||^2 + ||k_j||^2 - 2 * q_i . k_j
        q_norm_sq = torch.sum(q**2, dim=-1, keepdim=True)  # (b, h, t, 1)
        k_norm_sq = torch.sum(k**2, dim=-1, keepdim=True)  # (b, h, t, 1)
        dot_product = torch.matmul(q, k.transpose(-2, -1))  # (b, h, t, t)
        dist_sq = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot_product
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # Attention logits: - dist_sq / scale
        attn_logits = -dist_sq / self.scale

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # (b, h, t, head_dim)
        context = context.transpose(1, 2).contiguous().view(b, t, d)
        return cast(torch.Tensor, self.out_proj(context))


class SignFormerGCN(nn.Module):
    """Tier-2 Research Track: Continuous ISL Translation Model.

    Combines ST-GCN spatial-temporal landmark feature extraction with a
    Transformer Sequence Encoder-Decoder with Euclidean attention.
    """

    def __init__(self, config: Optional[Tier2SignFormerConfig] = None):
        super().__init__()
        self.config = config or Tier2SignFormerConfig()

        # ST-GCN Front-End
        self.stgcn1 = STGCNBlock(self.config.in_channels, self.config.graph_hidden_dim // 2, self.config.num_nodes)
        self.stgcn2 = STGCNBlock(self.config.graph_hidden_dim // 2, self.config.graph_hidden_dim, self.config.num_nodes)

        # Spatial pooling across 76 skeleton nodes -> graph feature vector
        self.proj_to_transformer = nn.Linear(self.config.graph_hidden_dim, self.config.transformer_d_model)

        # Euclidean Self-Attention Encoder Layer
        self.encoder_attn = EuclideanSelfAttention(
            self.config.transformer_d_model,
            nhead=self.config.nhead,
            dropout=self.config.dropout,
        )
        self.encoder_norm1 = nn.LayerNorm(self.config.transformer_d_model)
        self.encoder_ffn = nn.Sequential(
            nn.Linear(self.config.transformer_d_model, self.config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dim_feedforward, self.config.transformer_d_model),
        )
        self.encoder_norm2 = nn.LayerNorm(self.config.transformer_d_model)

        # Autoregressive Decoder additions
        self.tgt_embedding = nn.Embedding(self.config.vocab_size, self.config.transformer_d_model)
        self.pos_encoder = PositionalEncoding(self.config.transformer_d_model, max_len=self.config.max_target_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.transformer_d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_decoder_layers)

        # Sequence translation head (gloss-free text generation logits)
        self.translation_head = nn.Linear(self.config.transformer_d_model, self.config.vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 76, 2) or (batch, seq_len, 152)
        if x.dim() == 3:
            b, t, c = x.shape
            x = x.view(b, t, self.config.num_nodes, self.config.in_channels)

        feat = self.stgcn1(x)
        feat = self.stgcn2(feat)  # (batch, seq_len, 76, hidden_dim)

        # Mean pool over joints
        feat = feat.mean(dim=2)  # (batch, seq_len, hidden_dim)
        feat = self.proj_to_transformer(feat)  # (batch, seq_len, d_model)

        # Euclidean attention encoder
        attn_out = self.encoder_attn(feat)
        feat = self.encoder_norm1(feat + attn_out)
        ffn_out = self.encoder_ffn(feat)
        memory = self.encoder_norm2(feat + ffn_out)
        return cast(torch.Tensor, memory)

    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for training with teacher forcing.

        src: (batch, seq_len, num_nodes, in_channels) or flat
        tgt: (batch, tgt_len) target token IDs. If None, acts as a sequence classifier.
        """
        memory = self.encode(src)

        if tgt is None:
            return cast(torch.Tensor, self.translation_head(memory))

        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)

        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

        logits = self.translation_head(out)
        return cast(torch.Tensor, logits)

    def generate(self, src: torch.Tensor, max_len: int, start_token_id: int, end_token_id: int) -> torch.Tensor:
        """Autoregressive greedy decoding.

        src: (batch, seq_len, num_nodes, in_channels) or flat
        """
        device = src.device
        batch_size = src.size(0)

        memory = self.encode(src)

        tgt = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_emb = self.tgt_embedding(tgt)
            tgt_emb = self.pos_encoder(tgt_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)

            logits = self.translation_head(out[:, -1, :])
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)

            if (tgt == end_token_id).any(dim=1).all():
                break

        return tgt

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.__dict__,
                "model_type": "signformer_gcn",
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SignFormerGCN":
        checkpoint = torch.load(path, map_location=device)
        config = Tier2SignFormerConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
