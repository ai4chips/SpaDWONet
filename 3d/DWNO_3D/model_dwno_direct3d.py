








import jax, jax.numpy as jnp, equinox as eqx
import os, sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
from waveconv1d_jax import WaveConv1d, Conv1dResidual


class DWNODirect3d(eqx.Module):


    fc0: eqx.nn.Linear; fc_in: eqx.nn.Linear

    conv_x: tuple; w_x: tuple
    conv_y: tuple; w_y: tuple
    conv_z: tuple; w_z: tuple

    fc1: eqx.nn.Linear; fc2: eqx.nn.Linear

    grid_x: jnp.ndarray; grid_y: jnp.ndarray; grid_z: jnp.ndarray

    nx: int = eqx.field(static=True); ny: int = eqx.field(static=True); nz: int = eqx.field(static=True)
    width: int = eqx.field(static=True); level: int = eqx.field(static=True)
    num_param_levels: int = eqx.field(static=True)
    num_wavelet_layers: int = eqx.field(static=True)

    def __init__(self, nx=64, ny=64, nz=48, width=64, level=4,
                 num_param_levels=1, num_wavelet_layers=6, *, key):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.width, self.level = width, level
        self.num_param_levels = num_param_levels
        self.num_wavelet_layers = num_wavelet_layers

        N = num_wavelet_layers

        k0, k_in, k1, k2, kc = jax.random.split(key, 5)
        self.fc0 = eqx.nn.Linear(3, width, key=k0, use_bias=True)
        self.fc_in = eqx.nn.Linear(width + 1, width, key=k_in, use_bias=True)
        self.fc1 = eqx.nn.Linear(width, 128, key=k1, use_bias=True)
        self.fc2 = eqx.nn.Linear(128, 1, key=k2, use_bias=True)

        dx = jnp.ones((width, nx))
        dy = jnp.ones((width, ny))
        dz = jnp.ones((width, nz))
        keys = jax.random.split(kc, 6 * N)

        def _sc(d, k):
            return WaveConv1d(width, width, level, d, k, num_param_levels=num_param_levels)

        def _cr(k):
            return Conv1dResidual(width, k)

        ci = 0
        cx_list = []; wx_list = []
        cy_list = []; wy_list = []
        cz_list = []; wz_list = []
        for _ in range(N):
            cx_list.append(_sc(dx, keys[ci])); ci += 1
        for _ in range(N):
            wx_list.append(_cr(keys[ci])); ci += 1
        for _ in range(N):
            cy_list.append(_sc(dy, keys[ci])); ci += 1
        for _ in range(N):
            wy_list.append(_cr(keys[ci])); ci += 1
        for _ in range(N):
            cz_list.append(_sc(dz, keys[ci])); ci += 1
        for _ in range(N):
            wz_list.append(_cr(keys[ci])); ci += 1

        self.conv_x = tuple(cx_list); self.w_x = tuple(wx_list)
        self.conv_y = tuple(cy_list); self.w_y = tuple(wy_list)
        self.conv_z = tuple(cz_list); self.w_z = tuple(wz_list)

        self.grid_x = jnp.linspace(0, 1, nx)
        self.grid_y = jnp.linspace(0, 1, ny)
        self.grid_z = jnp.linspace(0, 0.75, nz)

    def __call__(self, source):
        B = source.shape[0]
        nx, ny, nz, w = self.nx, self.ny, self.nz, self.width


        a = source[..., None]
        gx = jnp.broadcast_to(self.grid_x[None, :, None, None], (B, nx, ny, 1))
        gy = jnp.broadcast_to(self.grid_y[None, None, :, None], (B, nx, ny, 1))
        x = jnp.concatenate([a, gx, gy], axis=-1).reshape(-1, 3)
        x = jax.vmap(self.fc0)(x).reshape(B, nx, ny, w)


        x = jnp.repeat(x[:, None, ...], nz, axis=1)
        gz = jnp.broadcast_to(self.grid_z[None, :, None, None, None], (B, nz, nx, ny, 1))
        x = jnp.concatenate([x, gz], axis=-1).reshape(-1, w + 1)
        x = jax.vmap(self.fc_in)(x).reshape(B, nz, nx, ny, w)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        v = jax.vmap
        N = self.num_wavelet_layers

        for i in range(N):

            xx = jnp.transpose(x, (0, 2, 4, 1, 3)).reshape(-1, w, nx)
            xx = v(self.conv_x[i])(xx) + v(self.w_x[i])(xx)
            xx = xx.reshape(B, nz, ny, w, nx).transpose(0, 3, 1, 4, 2)


            xy = jnp.transpose(x, (0, 2, 3, 1, 4)).reshape(-1, w, ny)
            xy = v(self.conv_y[i])(xy) + v(self.w_y[i])(xy)
            xy = xy.reshape(B, nz, nx, w, ny).transpose(0, 3, 1, 2, 4)


            xz = jnp.transpose(x, (0, 3, 4, 1, 2)).reshape(-1, w, nz)
            xz = v(self.conv_z[i])(xz) + v(self.w_z[i])(xz)
            xz = xz.reshape(B, nx, ny, w, nz).transpose(0, 3, 4, 1, 2)

            if i < N - 1:
                x = jax.nn.gelu(xx + xy + xz)
            else:
                x = xx + xy + xz


        x = jnp.transpose(x, (0, 2, 3, 4, 1)).reshape(-1, w)
        x = jax.vmap(self.fc1)(x); x = jax.nn.gelu(x); x = jax.vmap(self.fc2)(x)
        x = x.reshape(B, nz, nx, ny).transpose(0, 2, 3, 1)
        return x


def count_params(model):
    return sum(l.size for l in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    m = DWNODirect3d(nx=64, ny=64, nz=48, width=64, level=4,
                     num_param_levels=1, num_wavelet_layers=6, key=key)
    print(f'Params: {count_params(m):,}')
    src = jnp.ones((2, 64, 64))
    out = m(src)
    print(f'{src.shape} -> {out.shape}')
