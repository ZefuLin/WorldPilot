"""
Trainable fusion layers for integrating Cosmos world-model predictions into ABot.

- CosmosImageFuser:   flattens Cosmos VAE latent frames to one token per camera
                      (2 tokens total), adds temporal embeddings, then cross-attends
                      into VLM hidden states.
- CosmosActionProjector: encodes Cosmos advisor action chunk into a single token
                         that is prepended to the DiT action-head input sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CosmosTemporalEmbedding(nn.Module):
    """Learnable embedding that distinguishes 'current' from 'future step k'."""

    def __init__(self, d_model: int, max_future_steps: int = 16):
        super().__init__()
        self.type_embed = nn.Embedding(2, d_model)           # 0=current, 1=future
        self.step_embed = nn.Embedding(max_future_steps + 1, d_model)  # step index 0..K
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.step_embed.weight, std=0.02)

    def forward(self, features: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:  (B, N, D)
            step_ids:  (B, N) long — 0 = current, 1..K = future step index
        Returns:
            (B, N, D) with temporal info added.
        """
        type_ids = (step_ids > 0).long()
        return features + self.type_embed(type_ids) + self.step_embed(step_ids)


class CosmosImageFuser(nn.Module):
    """
    Flatten Cosmos VAE latent frames -> token sequence, add temporal embedding,
    then fuse into VLM hidden states via cross-attention (Q = VLM, KV = cosmos).
    """

    def __init__(
        self,
        hidden_size: int,
        cosmos_latent_channels: int = 16,
        cosmos_latent_spatial: int = 28,
        num_cameras: int = 2,
        nhead: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        latent_dim = cosmos_latent_channels * cosmos_latent_spatial * cosmos_latent_spatial
        self.projector = nn.Linear(latent_dim, hidden_size)
        self.temporal_embed = CosmosTemporalEmbedding(hidden_size)
        self.num_cameras = num_cameras

        from WorldPilot.model.modules.vggt_tools import CrossAttention
        self.cross_attn = CrossAttention(
            d_model=hidden_size,
            d_hidden=hidden_size,
            nhead=nhead,
            dropout=dropout,
            kv_dim=hidden_size,
        )

    def forward(
        self,
        vl_hidden: torch.Tensor,
        future_image_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vl_hidden:             (B, L, H) — VLM last hidden states.
            future_image_latents:  (B, N_cam, C', H', W') — Cosmos latent frames.
        Returns:
            (B, L, H) — VLM hidden fused with future-image information.
        """
        B, N_cam, C, Hp, Wp = future_image_latents.shape
        flat = future_image_latents.reshape(B, N_cam, C * Hp * Wp)
        tokens = self.projector(flat.to(vl_hidden.dtype))

        step_ids = torch.ones(B, N_cam, dtype=torch.long, device=vl_hidden.device)
        tokens = self.temporal_embed(tokens, step_ids)

        return self.cross_attn(vl_hidden, tokens)


class CosmosActionProjector(nn.Module):
    """
    Encode the Cosmos advisor action chunk into token(s) that will be
    prepended to the DiT action-head input sequence.
    """

    def __init__(
        self,
        action_dim: int = 7,
        cosmos_chunk_size: int = 16,
        abot_action_horizon: int = 10,
        output_dim: int = 768,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.abot_action_horizon = abot_action_horizon
        self.action_horizon = min(cosmos_chunk_size, abot_action_horizon)
        input_dim = action_dim * self.action_horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cosmos_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cosmos_actions: (B, cosmos_chunk_size, action_dim), normalized [-1,1]
        Returns:
            (B, 1, output_dim) — single advisor token for DiT input.
        """
        truncated = cosmos_actions[:, : self.abot_action_horizon, :]
        flat = truncated.reshape(truncated.shape[0], -1)
        return self.mlp(flat).unsqueeze(1)
