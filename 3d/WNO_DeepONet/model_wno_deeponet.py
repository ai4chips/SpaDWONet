








import jax, jax.numpy as jnp, equinox as eqx
from typing import Sequence, Optional
import os, sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from waveconv2d_jax import WaveConv2d, Conv2dResidual


class MLP(eqx.Module):
    layers: tuple
    def __init__(self, layer_sizes: Sequence[int], *, key):
        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layers = tuple(eqx.nn.Linear(layer_sizes[i], layer_sizes[i+1], key=keys[i], use_bias=True)
                           for i in range(len(layer_sizes) - 1))
    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x


class WNOSeparableDeepONet3d(eqx.Module):


    fc0: eqx.nn.Linear
    conv: tuple
    w: tuple
    branch_head: eqx.nn.Linear

    trunk_x: MLP; trunk_y: MLP; trunk_z: MLP
    output_bias: jnp.ndarray
    coords_x: jnp.ndarray; coords_y: jnp.ndarray; coords_z: jnp.ndarray

    R: int = eqx.field(static=True)
    nx: int = eqx.field(static=True); ny: int = eqx.field(static=True); nz: int = eqx.field(static=True)
    p: int = eqx.field(static=True); width: int = eqx.field(static=True); level: int = eqx.field(static=True)
    num_param_levels: int = eqx.field(static=True)
    num_wavelet_layers: int = eqx.field(static=True)

    def __init__(self, trunk_layers: Sequence[int], R=50, nx=64, ny=64, nz=48,
                 width=64, level=4, num_param_levels=1, num_wavelet_layers=10, *, key):
        super().__init__()
        self.R, self.nx, self.ny, self.nz = R, nx, ny, nz
        self.p, self.width, self.level = trunk_layers[-1], width, level
        self.num_param_levels = num_param_levels
        self.num_wavelet_layers = num_wavelet_layers

        N = num_wavelet_layers

        k_fc0, k_bh, k_tx, k_ty, k_tz, k_conv = jax.random.split(key, 6)

        self.fc0 = eqx.nn.Linear(1, width, key=k_fc0, use_bias=True)

        dummy = jnp.ones((width, nx, ny))
        keys_c = jax.random.split(k_conv, 2 * N)

        conv_list = []
        w_list = []
        for i in range(N):
            conv_list.append(WaveConv2d(width, width, level, dummy, keys_c[i],
                                        num_param_levels=num_param_levels))
        for i in range(N):
            w_list.append(Conv2dResidual(width, keys_c[N + i]))
        self.conv = tuple(conv_list)
        self.w = tuple(w_list)

        self.branch_head = eqx.nn.Linear(width, self.p, key=k_bh, use_bias=True)

        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_tx)
        self.trunk_y = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_ty)
        self.trunk_z = MLP([1] + hidden + [trunk_layers[-1] * R], key=k_tz)

        self.output_bias = jnp.zeros(())
        self.coords_x = jnp.linspace(0, 1, nx)
        self.coords_y = jnp.linspace(0, 1, ny)
        self.coords_z = jnp.linspace(0, 0.75, nz)

    def _wno_feature_extract(self, x):

        B, s1, s2, _ = x.shape
        x = x.reshape(-1, 1)
        x = jax.vmap(self.fc0)(x)
        x = x.reshape(B, s1, s2, self.width)
        v = jnp.transpose(x, (0, 3, 1, 2))

        N = self.num_wavelet_layers
        for i in range(N):
            v = jax.vmap(self.conv[i])(v) + jax.vmap(self.w[i])(v)
            if i < N - 1:
                v = jax.nn.gelu(v)
        return v

    def __call__(self, source, coords_x=None, coords_y=None, coords_z=None):
        B = source.shape[0]
        nx, ny, nz, R, p = self.nx, self.ny, self.nz, self.R, self.p

        if coords_x is None: coords_x = self.coords_x
        if coords_y is None: coords_y = self.coords_y
        if coords_z is None: coords_z = self.coords_z

        a = source[..., None]
        features = self._wno_feature_extract(a)
        features = features.mean(axis=(2, 3))
        Bf = jax.vmap(self.branch_head)(features)

        phi = jax.vmap(self.trunk_x)(coords_x[:, None])
        phi = phi.reshape(nx, R, p, 1)
        psi = jax.vmap(self.trunk_y)(coords_y[:, None])
        psi = psi.reshape(ny, R, p, 1)
        zeta = jax.vmap(self.trunk_z)(coords_z[:, None])
        zeta = zeta.reshape(nz, R, p, 1)

        trunk_3d = jnp.einsum('xRpo,yRpo,zRpo->xyzpo', phi, psi, zeta)
        trunk_3d = jnp.squeeze(trunk_3d, axis=-1)
        u = jnp.einsum('bp,ijkp->bijk', Bf, trunk_3d)
        return u + self.output_bias


def count_params(model):
    return sum(l.size for l in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    model = WNOSeparableDeepONet3d(trunk_layers=[50]*7, R=50, nx=64, ny=64, nz=48, width=64, level=4,
                                   num_param_levels=1, num_wavelet_layers=10, key=key)
    print(f'Parameters: {count_params(model):,}')
    src = jnp.ones((2, 64, 64))
    out = model(src)
    print(f'Input: {src.shape}  ->  Output: {out.shape}')
    print('OK')
