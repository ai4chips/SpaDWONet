from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(sizes):
    layers = []
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        linear = nn.Linear(left, right)
        
        nn.init.xavier_normal_(linear.weight)
        layers.append(linear)
        if index < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class DeepOHeat2D(nn.Module):
    def __init__(self, size, branch_layers, trunk_layers, fourier_frequencies,
                 fourier_std=1.0, fourier_frequency_scale="2*pi", activation="silu"):
        super().__init__()
        if activation != "silu" or fourier_frequency_scale != "2*pi":
            raise ValueError("This adaptation preserves the author's SiLU / 2*pi Fourier encoding")
        if size < 2 or branch_layers[0] != size * size:
            raise ValueError("The branch must accept the complete square power map")
        if (fourier_frequencies < 1 or trunk_layers[0] != 2 * fourier_frequencies
                or branch_layers[-1] != trunk_layers[-1]):
            raise ValueError("Incompatible Fourier or branch/trunk feature dimensions")
        self.size = size
        self.branch = _mlp(branch_layers)
        self.trunk = _mlp(trunk_layers)
        self.register_buffer("fourier_features",
                             torch.randn(2, fourier_frequencies) * fourier_std * (2 * math.pi))
        axis = torch.linspace(0.0, 1.0, size)
        self.register_buffer("coordinates",
                             torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2))
        self.output_bias = nn.Parameter(torch.empty(1).uniform_())

    def forward(self, power):
        if power.ndim != 4 or power.shape[1:] != (1, self.size, self.size):
            raise ValueError(f"Expected Bx1x{self.size}x{self.size}, received {tuple(power.shape)}")
        phases = self.coordinates @ self.fourier_features
        encoded = torch.cat((phases.sin(), phases.cos()), dim=-1)
        basis = self.trunk(encoded)
        coefficients = self.branch(power.flatten(1))
        values = coefficients @ basis.T + self.output_bias
        return values.reshape(-1, 1, self.size, self.size)

    def parameter_allocation(self):
        branch = sum(p.numel() for p in self.branch.parameters() if p.requires_grad)
        trunk = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        bias = self.output_bias.numel()
        return {"branch": branch, "joint_trunk": trunk, "output_bias": bias,
                "total": branch + trunk + bias}
