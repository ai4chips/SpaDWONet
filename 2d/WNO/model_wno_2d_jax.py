import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_jax_dir = os.path.join(_this_dir, '..')
sys.path.insert(0, _jax_dir)

import jax, jax.numpy as jnp, equinox as eqx
from waveconv2d_jax import WaveConv2d, Conv2dResidual


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


class WNO2d(eqx.Module):
    fc0: Dense
    conv: tuple  
    w_res: tuple  
    fc1: Dense
    fc2: Dense
    width: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    s: int = eqx.field(static=True)
    num_param_levels: int = eqx.field(static=True)
    num_wavelet_layers: int = eqx.field(static=True)

    def __init__(self, width=64, level=4, s=64,
                 num_param_levels=1, num_wavelet_layers=6, key=None):
        self.width, self.level, self.s = width, level, s
        self.num_param_levels = num_param_levels
        self.num_wavelet_layers = num_wavelet_layers
        N = num_wavelet_layers

        if key is None:
            key = jax.random.PRNGKey(0)
        k_fc0, k_fc1, k_fc2, key = jax.random.split(key, 4)

        self.fc0 = Dense(1, width, k_fc0)

        dummy = jnp.ones((width, s, s))
        conv_list, res_list = [], []
        for i in range(N):
            kc, kr, key = jax.random.split(key, 3)
            conv_list.append(WaveConv2d(width, width, level, dummy, kc, num_param_levels=num_param_levels))
            res_list.append(Conv2dResidual(width, kr))
        self.conv = tuple(conv_list)
        self.w_res = tuple(res_list)

        self.fc1 = Dense(width, 512, k_fc1)
        self.fc2 = Dense(512, 1, k_fc2)

    def __call__(self, x):
        B, N_layers = x.shape[0], self.num_wavelet_layers
        v = jax.nn.gelu(self.fc0(x))               
        v = v.transpose(0, 3, 1, 2)                 

        vf = jax.vmap
        for i in range(N_layers):
            x1 = vf(self.conv[i])(v); x2 = vf(self.w_res[i])(v)
            v = x1 + x2 if i == N_layers - 1 else jax.nn.gelu(x1 + x2)

        v = v.transpose(0, 2, 3, 1)                 
        v = jnp.tanh(self.fc1(v))
        return self.fc2(v)                           
