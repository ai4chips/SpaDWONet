












import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Optional


class Conv2dBlock(eqx.Module):

    conv: eqx.nn.Conv
    norm: eqx.nn.GroupNorm

    def __init__(self, in_ch, out_ch, kernel_size, *, key):
        k1, k2 = jax.random.split(key)
        self.conv = eqx.nn.Conv(2, in_ch, out_ch, kernel_size, padding=kernel_size//2, key=k1)
        groups = 8
        while out_ch % groups != 0:
            groups -= 1
        self.norm = eqx.nn.GroupNorm(groups=groups, channels=out_ch)

    def __call__(self, x):
        return self.norm(self.conv(x))


class Conv2dResidual(eqx.Module):

    conv: eqx.nn.Conv

    def __init__(self, channels, *, key):
        self.conv = eqx.nn.Conv(2, channels, channels, 1, key=key)

    def __call__(self, x):
        return self.conv(x)


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, layer_sizes: Sequence[int], *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(layer_sizes) - 1)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(eqx.nn.Linear(
                layer_sizes[i], layer_sizes[i + 1], key=keys[i], use_bias=True))
        self.layers = tuple(layers)

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x


class CNNDeepONet3d(eqx.Module):



    fc0: eqx.nn.Linear
    conv0: Conv2dBlock; conv1: Conv2dBlock; conv2: Conv2dBlock
    conv3: Conv2dBlock; conv4: Conv2dBlock; conv5: Conv2dBlock
    w0: Conv2dResidual; w1: Conv2dResidual; w2: Conv2dResidual
    w3: Conv2dResidual; w4: Conv2dResidual; w5: Conv2dResidual
    branch_head: eqx.nn.Linear


    trunk_x: MLP
    trunk_y: MLP
    trunk_z: MLP

    output_bias: jnp.ndarray
    coords_x_default: jnp.ndarray
    coords_y_default: jnp.ndarray
    coords_z_default: jnp.ndarray

    r: int = eqx.field(static=True)
    nx: int = eqx.field(static=True)
    ny: int = eqx.field(static=True)
    nz: int = eqx.field(static=True)
    p: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(
        self,
        trunk_layers: Sequence[int],
        width: int = 64,
        r: int = 50,
        nx: int = 64, ny: int = 64, nz: int = 48,
        *,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        self.r = r
        self.nx = nx; self.ny = ny; self.nz = nz
        self.p = trunk_layers[-1]
        self.width = width

        k_fc0, k_bh, k_tx, k_ty, k_tz, k_conv = jax.random.split(key, 6)
        keys_c = jax.random.split(k_conv, 14)
        ci = 0

        self.fc0 = eqx.nn.Linear(1, width, key=k_fc0, use_bias=True)

        self.conv0 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1
        self.conv1 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1
        self.conv2 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1
        self.conv3 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1
        self.conv4 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1
        self.conv5 = Conv2dBlock(width, width, 3, key=keys_c[ci]); ci += 1

        self.w0 = Conv2dResidual(width, key=keys_c[ci]); ci += 1
        self.w1 = Conv2dResidual(width, key=keys_c[ci]); ci += 1
        self.w2 = Conv2dResidual(width, key=keys_c[ci]); ci += 1
        self.w3 = Conv2dResidual(width, key=keys_c[ci]); ci += 1
        self.w4 = Conv2dResidual(width, key=keys_c[ci]); ci += 1
        self.w5 = Conv2dResidual(width, key=keys_c[ci]); ci += 1

        self.branch_head = eqx.nn.Linear(width, self.p, key=k_bh, use_bias=True)

        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_tx)
        self.trunk_y = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_ty)
        self.trunk_z = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_tz)

        self.output_bias = jnp.zeros(())
        self.coords_x_default = jnp.linspace(0, 1, nx)
        self.coords_y_default = jnp.linspace(0, 1, ny)
        self.coords_z_default = jnp.linspace(0, 0.75, nz)

    def _cnn_feature_extract(self, x):

        B = x.shape[0]
        x = x.reshape(-1, 1)
        x = jax.vmap(self.fc0)(x)
        x = x.reshape(B, self.nx, self.ny, self.width)
        v0 = jnp.transpose(x, (0, 3, 1, 2))


        def block0(x): return jax.nn.gelu(self.conv0(x) + self.w0(x))
        def block1(x): return jax.nn.gelu(self.conv1(x) + self.w1(x))
        def block2(x): return jax.nn.gelu(self.conv2(x) + self.w2(x))
        def block3(x): return jax.nn.gelu(self.conv3(x) + self.w3(x))
        def block4(x): return jax.nn.gelu(self.conv4(x) + self.w4(x))
        def block5(x): return self.conv5(x) + self.w5(x)

        v0 = jax.vmap(block0)(v0)
        v0 = jax.vmap(block1)(v0)
        v0 = jax.vmap(block2)(v0)
        v0 = jax.vmap(block3)(v0)
        v0 = jax.vmap(block4)(v0)
        v0 = jax.vmap(block5)(v0)
        return v0

    def __call__(
        self,
        source: jnp.ndarray,
        coords_x: Optional[jnp.ndarray] = None,
        coords_y: Optional[jnp.ndarray] = None,
        coords_z: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        B = source.shape[0]
        nx, ny, nz = self.nx, self.ny, self.nz
        r, p = self.r, self.p

        if coords_x is None: coords_x = self.coords_x_default
        if coords_y is None: coords_y = self.coords_y_default
        if coords_z is None: coords_z = self.coords_z_default


        a = source[..., None]
        features = self._cnn_feature_extract(a)
        features = features.mean(axis=(2, 3))
        Bf = jax.vmap(self.branch_head)(features)
        Bf = Bf.reshape(B, p, 1)


        x = coords_x[:, None]
        phi = jax.vmap(self.trunk_x)(x)
        phi = phi.reshape(nx, r, p, 1)

        y = coords_y[:, None]
        psi = jax.vmap(self.trunk_y)(y)
        psi = psi.reshape(ny, r, p, 1)

        z = coords_z[:, None]
        zeta = jax.vmap(self.trunk_z)(z)
        zeta = zeta.reshape(nz, r, p, 1)


        trunk_3d = jnp.einsum('xrpo,yrpo,zrpo->xyzpo', phi, psi, zeta)                      

        u = jnp.einsum('bpo,xyzpo->bxyzo', Bf, trunk_3d)                      
        u = u.squeeze(-1)
        u = u + self.output_bias
        return u


def count_params(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    trunk_layers = [50, 50, 50, 50, 50, 50, 50]
    model = CNNDeepONet3d(trunk_layers=trunk_layers, width=64, r=50, key=key)
    print(f'Params: {count_params(model):,}')
    src = jnp.ones((2, 64, 64))
    out = model(src)
    print(f'Input: {src.shape}  ->  Output: {out.shape}')
    print('OK')
