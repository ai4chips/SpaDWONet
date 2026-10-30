import jax, jax.numpy as jnp, equinox as eqx
from typing import Tuple
import os, sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from jax_wavelets import jax_dwt1d, jax_idwt1d


def _dwt2d_single(x: jnp.ndarray):
    C, H, W = x.shape
    xr = x.reshape(-1, W)
    ar, dr = jax_dwt1d(xr, 1)
    dr0 = dr[0]
    A = ar.reshape(C, H, W // 2)
    D = dr0.reshape(C, H, W // 2)

    ac = A.transpose(0, 2, 1).reshape(-1, H)
    ll_a, lh_d = jax_dwt1d(ac, 1)
    LL = ll_a.reshape(C, W // 2, H // 2).transpose(0, 2, 1)
    LH = lh_d[0].reshape(C, W // 2, H // 2).transpose(0, 2, 1)

    dc = D.transpose(0, 2, 1).reshape(-1, H)
    hl_a, hh_d = jax_dwt1d(dc, 1)
    HL = hl_a.reshape(C, W // 2, H // 2).transpose(0, 2, 1)
    HH = hh_d[0].reshape(C, W // 2, H // 2).transpose(0, 2, 1)

    return LL, (LH, HL, HH)


def _idwt2d_single(LL: jnp.ndarray, details: Tuple):
    LH, HL, HH = details
    C, Hh, Wh = LL.shape

    ll_c = LL.transpose(0, 2, 1).reshape(-1, Hh)
    lh_c = LH.transpose(0, 2, 1).reshape(-1, Hh)
    row_approx = jax_idwt1d(ll_c, [lh_c]).reshape(C, Wh, Hh*2).transpose(0, 2, 1)

    hl_c = HL.transpose(0, 2, 1).reshape(-1, Hh)
    hh_c = HH.transpose(0, 2, 1).reshape(-1, Hh)
    row_detail = jax_idwt1d(hl_c, [hh_c]).reshape(C, Wh, Hh*2).transpose(0, 2, 1)

    ra = row_approx.reshape(-1, Wh)
    rd = row_detail.reshape(-1, Wh)
    return jax_idwt1d(ra, [rd]).reshape(C, Hh*2, Wh*2)


class WaveConv2d(eqx.Module):
    w_ll: Tuple[jnp.ndarray, ...]
    w_lh: Tuple[jnp.ndarray, ...]
    w_hl: Tuple[jnp.ndarray, ...]
    w_hh: Tuple[jnp.ndarray, ...]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    num_param_levels: int = eqx.field(static=True)

    def __init__(self, in_channels, out_channels, level, dummy, key, num_param_levels=4):
        self.in_channels, self.out_channels, self.level = in_channels, out_channels, level
        self.num_param_levels = num_param_levels
        scale = 1.0 / (in_channels * out_channels)
        k = num_param_levels
        first = level - k

        w_ll, w_lh, w_hl, w_hh = [], [], [], []
        current = dummy
        for lvl in range(level):
            LL, _ = _dwt2d_single(current)
            h, w = LL.shape[-2], LL.shape[-1]
            k0, k1, k2, k3, key = jax.random.split(key, 5)
            w_ll.append(scale * jax.random.uniform(k0, (in_channels, out_channels, h, w)))
            if lvl >= first:
                w_lh.append(scale * jax.random.uniform(k1, (in_channels, out_channels, h, w)))
                w_hl.append(scale * jax.random.uniform(k2, (in_channels, out_channels, h, w)))
                w_hh.append(scale * jax.random.uniform(k3, (in_channels, out_channels, h, w)))
            current = LL

        self.w_ll = tuple(w_ll); self.w_lh = tuple(w_lh)
        self.w_hl = tuple(w_hl); self.w_hh = tuple(w_hh)

    def mul2d(self, x, w):
        return jnp.einsum('ihw,iohw->ohw', x, w)

    def __call__(self, x):
        current = x
        details_stack = []
        for _ in range(self.level):
            LL, details = _dwt2d_single(current)
            details_stack.append(details)
            current = LL

        current = self.mul2d(current, self.w_ll[-1])
        first = self.level - self.num_param_levels
        for lvl in range(self.level - 1, -1, -1):
            LH, HL, HH = details_stack[lvl]
            if lvl >= first:
                p = lvl - first
                LH = self.mul2d(LH, self.w_lh[p])
                HL = self.mul2d(HL, self.w_hl[p])
                HH = self.mul2d(HH, self.w_hh[p])
            else:
                LH = jnp.zeros_like(LH)
                HL = jnp.zeros_like(HL)
                HH = jnp.zeros_like(HH)
            current = _idwt2d_single(current, (LH, HL, HH))
        return current


class Conv2dResidual(eqx.Module):
    conv: eqx.nn.Conv2d
    def __init__(self, width, key):
        super().__init__()
        self.conv = eqx.nn.Conv2d(width, width, 1, key=key)
    def __call__(self, x):
        return self.conv(x)
