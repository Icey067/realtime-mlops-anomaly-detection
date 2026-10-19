"""PyTorch Autoencoder for unsupervised anomaly detection on financial transactions.

Architecture:
  - Input: 30 features (Time, V1..V28, Amount)
  - Encoder: 30 -> 16 -> 8 (Latent bottleneck)
  - Decoder: 8 -> 16 -> 30 (Reconstruction)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Deep autoencoder with LeakyReLU activations and latent bottleneck representation."""

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

        # Encoder: 30 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Decoder: 8 -> 16 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input vector to latent bottleneck representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct original feature space from latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete forward pass: input -> bottleneck -> reconstruction."""
        z = self.encode(x)
        return self.decode(z)

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample Mean Squared Error (MSE) reconstruction loss."""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse
