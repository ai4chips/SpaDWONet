
import jax, jax.numpy as jnp, equinox as eqx
from typing import Tuple
from jax_wavelets import jax_dwt1d, jax_idwt1d


class WaveConv1d(eqx.Module):
    weight0: jnp.ndarray
    weight_coeffs: Tuple[jnp.ndarray, ...]
    num_param_levels: int = eqx.field(static=True)

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    modes: int = eqx.field(static=True)
    coeff_shapes: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, in_channels, out_channels, level, dummy, key, num_param_levels=1):
        self.in_channels, self.out_channels, self.level = in_channels, out_channels, level
        self.num_param_levels = num_param_levels

        dummy_approx, dummy_detail_coeffs = jax_dwt1d(dummy, level)
        self.modes = dummy_approx.shape[-1]
        self.coeff_shapes = tuple(coeff.shape[-1] for coeff in dummy_detail_coeffs)

        scale = 1.0 / (in_channels * out_channels)
        k0, *keys_coeffs = jax.random.split(key, len(self.coeff_shapes) + 1)

        self.weight0 = scale * jax.random.uniform(k0, (in_channels, out_channels, self.modes))

        
        w_list = []
        for i in range(self.level - num_param_levels, self.level):
            w_list.append(scale * jax.random.uniform(
                keys_coeffs[i], (in_channels, out_channels, self.coeff_shapes[i])))
        self.weight_coeffs = tuple(w_list)

    def mul1d(self, input, weights):
        return jnp.einsum("ix,iox->ox", input, weights)

    def __call__(self, x):
        x_ft, x_coeff = jax_dwt1d(x, self.level)
        out_ft = self.mul1d(x_ft, self.weight0)

        new_x_coeff = []
        for i, coeff in enumerate(x_coeff):
            param_idx = i - (self.level - self.num_param_levels)
            if param_idx >= 0:
                transformed_coeff = self.mul1d(coeff, self.weight_coeffs[param_idx])
            else:
                transformed_coeff = jnp.zeros_like(coeff)
            new_x_coeff.append(transformed_coeff)

        return jax_idwt1d(out_ft, new_x_coeff)


class Conv1dResidual(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(self, width, key):
        super().__init__()
        self.conv = eqx.nn.Conv1d(width, width, 1, key=key)

    def __call__(self, x):
        return self.conv(x)
