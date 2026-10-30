
import jax, jax.numpy as jnp, equinox as eqx
from typing import Sequence, Optional
import os, sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_jax_root = os.path.join(_this_dir, '..')
sys.path.insert(0, _jax_root)
from waveconv1d_jax import WaveConv1d, Conv1dResidual


class MLP(eqx.Module):
    layers: tuple
    def __init__(self, layer_sizes, *, key):
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layers = tuple(eqx.nn.Linear(layer_sizes[i], layer_sizes[i+1], key=keys[i], use_bias=True)
                           for i in range(len(layer_sizes) - 1))
    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x


class DWNOSeparableDeepONet2d(eqx.Module):

    fc0: eqx.nn.Linear
    weight_x: jnp.ndarray; weight_y: jnp.ndarray

    conv_x0:WaveConv1d;conv_x1:WaveConv1d;conv_x2:WaveConv1d;conv_x3:WaveConv1d;conv_x4:WaveConv1d;conv_x5:WaveConv1d
    wx0:Conv1dResidual;wx1:Conv1dResidual;wx2:Conv1dResidual;wx3:Conv1dResidual;wx4:Conv1dResidual;wx5:Conv1dResidual
    conv_y0:WaveConv1d;conv_y1:WaveConv1d;conv_y2:WaveConv1d;conv_y3:WaveConv1d;conv_y4:WaveConv1d;conv_y5:WaveConv1d
    wy0:Conv1dResidual;wy1:Conv1dResidual;wy2:Conv1dResidual;wy3:Conv1dResidual;wy4:Conv1dResidual;wy5:Conv1dResidual

    branch_head: eqx.nn.Linear
    trunk_x: MLP; trunk_y: MLP
    output_bias: jnp.ndarray
    coords_x: jnp.ndarray; coords_y: jnp.ndarray

    nx:int=eqx.field(static=True);ny:int=eqx.field(static=True)
    p:int=eqx.field(static=True);width:int=eqx.field(static=True);level:int=eqx.field(static=True);P:int=eqx.field(static=True)
    num_param_levels:int=eqx.field(static=True)

    def __init__(self, trunk_layers, nx=64, ny=64, width=64, level=4, P=25, num_param_levels=4, *, key):
        super().__init__()
        self.nx,self.ny,self.width,self.level,self.P=nx,ny,width,level,P
        self.num_param_levels = num_param_levels
        self.p = trunk_layers[-1]

        s = max(nx, ny)
        k_fc0,k_bh,k_tx,k_ty,k_conv = jax.random.split(key, 5)
        self.fc0 = eqx.nn.Linear(1, width, key=k_fc0, use_bias=True)
        self.weight_x = jax.random.normal(k_conv, (P, s))
        self.weight_y = jax.random.normal(k_conv, (P, s))

        dummy = jnp.ones((width, s))
        keys_c = jax.random.split(k_conv, 28); ci = 0
        def _sc(k):
            nonlocal ci; m = WaveConv1d(width, width, level, dummy, k, num_param_levels=self.num_param_levels); ci += 1; return m
        def _cr(k):
            nonlocal ci; m = Conv1dResidual(width, k); ci += 1; return m

        self.conv_x0=_sc(keys_c[ci]);self.conv_x1=_sc(keys_c[ci]);self.conv_x2=_sc(keys_c[ci])
        self.conv_x3=_sc(keys_c[ci]);self.conv_x4=_sc(keys_c[ci]);self.conv_x5=_sc(keys_c[ci])
        self.wx0=_cr(keys_c[ci]);self.wx1=_cr(keys_c[ci]);self.wx2=_cr(keys_c[ci])
        self.wx3=_cr(keys_c[ci]);self.wx4=_cr(keys_c[ci]);self.wx5=_cr(keys_c[ci])
        self.conv_y0=_sc(keys_c[ci]);self.conv_y1=_sc(keys_c[ci]);self.conv_y2=_sc(keys_c[ci])
        self.conv_y3=_sc(keys_c[ci]);self.conv_y4=_sc(keys_c[ci]);self.conv_y5=_sc(keys_c[ci])
        self.wy0=_cr(keys_c[ci]);self.wy1=_cr(keys_c[ci]);self.wy2=_cr(keys_c[ci])
        self.wy3=_cr(keys_c[ci]);self.wy4=_cr(keys_c[ci]);self.wy5=_cr(keys_c[ci])

        self.branch_head = eqx.nn.Linear(width, self.p, key=k_bh, use_bias=True)
        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [trunk_layers[-1]], key=k_tx)
        self.trunk_y = MLP([1] + hidden + [trunk_layers[-1]], key=k_ty)
        self.output_bias = jnp.zeros(())
        self.coords_x = jnp.linspace(0, 1, nx)
        self.coords_y = jnp.linspace(0, 1, ny)

    def _dw_no_feature_extract(self, x):
        B, s1, s2, _ = x.shape
        w = self.width
        x = x.reshape(-1, 1); x = jax.vmap(self.fc0)(x); x = x.reshape(B, s1, s2, w)
        v0 = jnp.transpose(x, (0, 3, 1, 2))

        Vx = jnp.einsum('bchw,ph->bpcw', v0, self.weight_x[:, :s1]); Vx = Vx.reshape(-1, w, s1)
        Vy = jnp.einsum('bchw,pw->bpch', v0, self.weight_y[:, :s2]); Vy = Vy.reshape(-1, w, s2)

        v = jax.vmap
        def _block(cx, wx, cy, wy, Vx, Vy):
            Vx = jax.nn.gelu(v(cx)(Vx) + v(wx)(Vx))
            Vy = jax.nn.gelu(v(cy)(Vy) + v(wy)(Vy))
            return Vx, Vy

        Vx, Vy = _block(self.conv_x0,self.wx0,self.conv_y0,self.wy0,Vx,Vy)
        Vx, Vy = _block(self.conv_x1,self.wx1,self.conv_y1,self.wy1,Vx,Vy)
        Vx, Vy = _block(self.conv_x2,self.wx2,self.conv_y2,self.wy2,Vx,Vy)
        Vx, Vy = _block(self.conv_x3,self.wx3,self.conv_y3,self.wy3,Vx,Vy)
        Vx, Vy = _block(self.conv_x4,self.wx4,self.conv_y4,self.wy4,Vx,Vy)
        Vx = v(self.conv_x5)(Vx) + v(self.wx5)(Vx)
        Vy = v(self.conv_y5)(Vy) + v(self.wy5)(Vy)

        Vx = Vx.reshape(B, self.P, w, s1)
        Vy = Vy.reshape(B, self.P, w, s2)
        return jnp.einsum('bpcw,bpch->bchw', Vx, Vy)

    def __call__(self, source, coords_x=None, coords_y=None):
        B = source.shape[0]
        if coords_x is None: coords_x = self.coords_x
        if coords_y is None: coords_y = self.coords_y

        a = source[..., None]
        features = self._dw_no_feature_extract(a)       
        Bf = features.transpose((0, 2, 3, 1))            
        Bf = jax.vmap(jax.vmap(jax.vmap(self.branch_head)))(Bf)  

        phi = jax.vmap(self.trunk_x)(coords_x[:, None])   
        psi = jax.vmap(self.trunk_y)(coords_y[:, None])   

        u = jnp.einsum('bijp,ip,jp->bij', Bf, phi, psi)   
        return u + self.output_bias


def count_params(model):
    return sum(l.size for l in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    m = DWNOSeparableDeepONet2d(trunk_layers=[50]*7, nx=64, ny=64, width=64, level=4, P=25, key=key)
    print(f'Params: {count_params(m):,}')
    src = jnp.ones((2, 64, 64))
    out = m(src)
    print(f'{src.shape} -> {out.shape}')
    print('OK')
