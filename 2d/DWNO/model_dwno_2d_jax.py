import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_jax_dir = os.path.join(_this_dir, '..')
sys.path.insert(0, _jax_dir)

import jax, jax.numpy as jnp, equinox as eqx
from waveconv1d_jax import WaveConv1d, Conv1dResidual


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class Dense(eqx.Module):
    w: jnp.ndarray
    b: jnp.ndarray
    def __init__(self, in_features, out_features, key):
        self.w = jax.random.normal(key, (out_features, in_features)) * (1.0 / (in_features ** 0.5))
        self.b = jnp.zeros(out_features)
    def __call__(self, x):
        return x @ self.w.T + self.b


class DWNO2d(eqx.Module):
    fc0: Dense
    weight_x: jnp.ndarray  
    weight_y: jnp.ndarray  
    conv_x: tuple
    w_x: tuple
    conv_y: tuple
    w_y: tuple
    fc1: Dense
    fc2: Dense
    width: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    s: int = eqx.field(static=True)
    num_param_levels: int = eqx.field(static=True)
    num_wavelet_layers: int = eqx.field(static=True)

    def __init__(self, width=64, level=4, s=64, P=25,
                 num_param_levels=4, num_wavelet_layers=6, key=None):
        self.width, self.level, self.s = width, level, s
        self.P = P
        self.num_param_levels = num_param_levels
        self.num_wavelet_layers = num_wavelet_layers
        N = num_wavelet_layers

        if key is None:
            key = jax.random.PRNGKey(0)
        k_fc0, k_wx, k_wy, k_fc1, k_fc2, *keys_c = jax.random.split(key, 5 + 4*N)

        self.fc0 = Dense(1, width, k_fc0)
        self.weight_x = jax.random.normal(k_wx, (P, s))
        self.weight_y = jax.random.normal(k_wy, (P, s))

        fx = jnp.ones((width, s))
        fy = jnp.ones((width, s))

        cx_list, wx_list = [], []
        for i in range(N):
            ki = keys_c[i]
            cx_list.append(WaveConv1d(width, width, level, fx, ki, num_param_levels=num_param_levels))
            wx_list.append(Conv1dResidual(width, keys_c[N + i]))
        self.conv_x = tuple(cx_list)
        self.w_x = tuple(wx_list)

        cy_list, wy_list = [], []
        for i in range(N):
            ki = keys_c[2*N + i]
            cy_list.append(WaveConv1d(width, width, level, fy, ki, num_param_levels=num_param_levels))
            wy_list.append(Conv1dResidual(width, keys_c[3*N + i]))
        self.conv_y = tuple(cy_list)
        self.w_y = tuple(wy_list)

        self.fc1 = Dense(width, 512, k_fc1)
        self.fc2 = Dense(512, 1, k_fc2)

    def __call__(self, x):
        B, N_layers = x.shape[0], self.num_wavelet_layers

        
        v = jax.nn.gelu(self.fc0(x))
        v0 = v.transpose(0, 3, 1, 2)           

        
        Vx = jnp.einsum('bchw,ph->bpcw', v0, self.weight_x)  
        Vy = jnp.einsum('bchw,pw->bpch', v0, self.weight_y)  

        v = jax.vmap  

        
        Vx_r = Vx.reshape(-1, self.width, self.s)            
        for i in range(N_layers):
            x1 = v(self.conv_x[i])(Vx_r); x2 = v(self.w_x[i])(Vx_r)
            Vx_r = x1 + x2 if i == N_layers - 1 else jax.nn.gelu(x1 + x2)
        Vx = Vx_r.reshape(B, self.P, self.width, self.s)

        
        Vy_r = Vy.reshape(-1, self.width, self.s)
        for i in range(N_layers):
            y1 = v(self.conv_y[i])(Vy_r); y2 = v(self.w_y[i])(Vy_r)
            Vy_r = y1 + y2 if i == N_layers - 1 else jax.nn.gelu(y1 + y2)
        Vy = Vy_r.reshape(B, self.P, self.width, self.s)

        
        vT = jnp.einsum('bpcw,bpch->bchw', Vx, Vy)          
        vT = vT.transpose(0, 2, 3, 1)                        

        
        vT = jnp.tanh(self.fc1(vT))
        return self.fc2(vT)                                   
