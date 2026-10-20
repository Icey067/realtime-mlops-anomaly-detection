"""Autoencoder with a 30 -> 16 -> 8 -> 16 -> 30 topology."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Reconstructs 30-dimensional transactions through an 8-dimensional bottleneck."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE between input and reconstruction."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse
