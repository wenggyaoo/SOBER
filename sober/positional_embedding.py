import math

import torch
import torch.nn as nn


class SinusoidalSpatialEmbedding(nn.Module):
    """Sinusoidal encoding of relative Cartesian coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 to evenly split between x/y sin/cos, but got {d_model}")
        self.d_model = d_model

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Map (batch, cells, 2) coordinates to (batch, cells, d_model)."""
        batch_size, seq_len, _ = coordinates.shape
        device = coordinates.device

        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        half_d_model = self.d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, 2, device=device) *
                             -(math.log(10000.0) / half_d_model))

        x_coords = coordinates[:, :, 0].unsqueeze(-1)
        y_coords = coordinates[:, :, 1].unsqueeze(-1)

        pos_embed[:, :, 0::4] = torch.sin(x_coords * div_term)
        pos_embed[:, :, 1::4] = torch.cos(x_coords * div_term)
        pos_embed[:, :, 2::4] = torch.sin(y_coords * div_term)
        pos_embed[:, :, 3::4] = torch.cos(y_coords * div_term)

        return pos_embed


class SinusoidalRotationalEmbedding(nn.Module):
    """Sinusoidal radius/angle encoding. Angle channels retain orientation."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for radius/angle sin/cos pairs, but got {d_model}")
        self.d_model = d_model

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Encode Cartesian coordinates relative to the center cell at (0, 0)."""
        batch_size, seq_len, _ = coordinates.shape
        device = coordinates.device

        x = coordinates[:, :, 0]
        y = coordinates[:, :, 1]
        radius = torch.sqrt(x**2 + y**2).unsqueeze(-1)
        angle = torch.atan2(y, x).unsqueeze(-1)

        pos_embed = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        half_d_model = self.d_model // 2
        div_term = torch.exp(torch.arange(0, half_d_model, 2, device=device) *
                             -(math.log(10000.0) / half_d_model))

        pos_embed[:, :, 0:half_d_model:2] = torch.sin(radius * div_term)
        pos_embed[:, :, 1:half_d_model:2] = torch.cos(radius * div_term)

        pos_embed[:, :, half_d_model::2] = torch.sin(angle * div_term)
        pos_embed[:, :, half_d_model + 1::2] = torch.cos(angle * div_term)

        return pos_embed
