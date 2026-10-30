import jax, jax.numpy as jnp, equinox as eqx
from typing import Tuple


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class Dense(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray
    def __init__(self, in_f, out_f, key):
        self.w = jax.random.normal(key, (out_f, in_f)) * (1.0 / (in_f ** 0.5))
        self.b = jnp.zeros(out_f)
    def __call__(self, x):
        return x @ self.w.T + self.b


class ChannelMLP(eqx.Module):
    fc0: eqx.nn.Conv2d
    fc1: eqx.nn.Conv2d
    def __init__(self, in_ch, hidden_ch, out_ch, key):
        k0, k1 = jax.random.split(key)
        self.fc0 = eqx.nn.Conv2d(in_ch, hidden_ch, 1, key=k0)
        self.fc1 = eqx.nn.Conv2d(hidden_ch, out_ch, 1, key=k1)
    def __call__(self, x):
        return self.fc1(jax.nn.gelu(self.fc0(x)))


class SpectralConv2d(eqx.Module):
    weight: jnp.ndarray  
    bias: jnp.ndarray    
    in_ch: int = eqx.field(static=True)
    out_ch: int = eqx.field(static=True)
    modes1: int = eqx.field(static=True)
    modes2: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, modes1, modes2, key):
        self.in_ch, self.out_ch = in_ch, out_ch
        
        k1_stored = modes1
        k2_stored = modes2 // 2 + 1
        self.modes1, self.modes2 = modes1, k2_stored
        scale = (2.0 / (in_ch + out_ch)) ** 0.5
        kr, ki, kb = jax.random.split(key, 3)
        wr = scale * jax.random.normal(kr, (in_ch, out_ch, k1_stored, k2_stored))
        wi = scale * jax.random.normal(ki, (in_ch, out_ch, k1_stored, k2_stored))
        self.weight = wr + 1j * wi
        self.bias = scale * jax.random.normal(kb, (out_ch, 1, 1))

    def __call__(self, x):
        
        H, W = x.shape[-2], x.shape[-1]
        
        x_ft = jnp.fft.rfft2(x)        
        x_ft = jnp.fft.fftshift(x_ft, axes=(-2,))  

        h_center = H // 2
        k1 = self.modes1
        k1_neg = k1 // 2
        k1_pos = k1 - k1_neg
        h_slice = slice(h_center - k1_neg, h_center + k1_pos)
        w_slice = slice(None, self.modes2)

        out_ft = jnp.zeros((self.out_ch, H, W // 2 + 1), dtype=jnp.complex64)
        out_ft = out_ft.at[:, h_slice, w_slice].set(
            jnp.einsum('ihw,iohw->ohw', x_ft[:, h_slice, w_slice], self.weight))

        
        out_ft = jnp.fft.ifftshift(out_ft, axes=(-2,))
        return jnp.fft.irfft2(out_ft, s=(H, W)).real + self.bias


class FNOBlock(eqx.Module):
    conv: SpectralConv2d
    skip_conv: eqx.nn.Conv2d  
    mlp: ChannelMLP
    skip_mlp: eqx.nn.Conv2d   
    ch: int = eqx.field(static=True)

    def __init__(self, ch, modes1, modes2, mlp_expansion, key):
        kc, ksc, km, ksm = jax.random.split(key, 4)
        self.ch = ch
        self.conv = SpectralConv2d(ch, ch, modes1, modes2, kc)
        self.skip_conv = eqx.nn.Conv2d(ch, ch, 1, key=ksc)
        mlp_hidden = int(round(ch * mlp_expansion))
        self.mlp = ChannelMLP(ch, mlp_hidden, ch, km)
        self.skip_mlp = eqx.nn.Conv2d(ch, ch, 1, key=ksm)

    def __call__(self, x):
        
        x1 = self.conv(x) + self.skip_conv(x)
        x1 = jax.nn.gelu(x1)
        
        x2 = self.mlp(x1) + self.skip_mlp(x1)
        return jax.nn.gelu(x2)


class FNO2d(eqx.Module):
    lifting: ChannelMLP
    blocks: Tuple[FNOBlock, ...]
    projection: ChannelMLP
    hidden: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)

    def __init__(self, in_ch=1, out_ch=1, hidden=64, modes=12, n_blocks=4,
                 lifting_ratio=2, proj_ratio=2, mlp_expansion=0.5, key=None):
        self.hidden, self.n_blocks = hidden, n_blocks
        if key is None:
            key = jax.random.PRNGKey(0)
        kl, kp, *kb = jax.random.split(key, 2 + n_blocks + 1)

        lift_hidden = int(hidden * lifting_ratio)
        proj_hidden = int(hidden * proj_ratio)

        self.lifting = ChannelMLP(in_ch, lift_hidden, hidden, kl)
        blocks = [FNOBlock(hidden, modes, modes, mlp_expansion, kb[i])
                  for i in range(n_blocks)]
        self.blocks = tuple(blocks)
        self.projection = ChannelMLP(hidden, proj_hidden, out_ch, kp)

    def __call__(self, x):
        
        vf = jax.vmap
        v = vf(self.lifting)(x)
        for blk in self.blocks:
            v = vf(blk)(v)
        return vf(self.projection)(v)
