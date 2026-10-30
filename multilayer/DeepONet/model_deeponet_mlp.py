












import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, Optional


class MLP(eqx.Module):
    layers: tuple

    def __init__(self, layer_sizes: Sequence[int], *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(layer_sizes) - 1)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(
                eqx.nn.Linear(layer_sizes[i], layer_sizes[i + 1],
                              key=keys[i], use_bias=True))
        self.layers = tuple(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x


class MLPDeepONet3d(eqx.Module):



    branch: MLP


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

    def __init__(
        self,
        branch_layers: Sequence[int],
        trunk_layers: Sequence[int],
        r: int = 50,
        nx: int = 64, ny: int = 64, nz: int = 48,
        *,
        key: jax.random.PRNGKey,
    ):
        super().__init__()
        self.r = r
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.p = trunk_layers[-1]

        k_br, k_tx, k_ty, k_tz = jax.random.split(key, 4)


        self.branch = MLP([nx * ny] + list(branch_layers), key=k_br)


        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_tx)
        self.trunk_y = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_ty)
        self.trunk_z = MLP([1] + hidden + [trunk_layers[-1] * r], key=k_tz)

        self.output_bias = jnp.zeros(())

        self.coords_x_default = jnp.linspace(0, 1, nx)
        self.coords_y_default = jnp.linspace(0, 1, ny)
        self.coords_z_default = jnp.linspace(0, 0.75, nz)

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

        if coords_x is None:
            coords_x = self.coords_x_default
        if coords_y is None:
            coords_y = self.coords_y_default
        if coords_z is None:
            coords_z = self.coords_z_default


        a = source.reshape(B, -1)
        Bf = jax.vmap(self.branch)(a)
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


        trunk_3d = jnp.einsum(
            'xrpo,yrpo,zrpo->xyzpo', phi, psi, zeta,
        )


        u = jnp.einsum('bpo,xyzpo->bxyzo', Bf, trunk_3d)
        u = u.squeeze(-1)
        u = u + self.output_bias
        return u


def count_params(model: eqx.Module) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return sum(leaf.size for leaf in leaves)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    branch_layers = [256, 256, 256, 50]
    trunk_layers = [50, 50, 50, 50, 50, 50, 50]

    model = MLPDeepONet3d(
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        r=50, nx=64, ny=64, nz=48,
        key=key,
    )

    print(f'Total parameters: {count_params(model):,}')
    print(f'Backend: {jax.default_backend()}')

    src = jnp.ones((2, 64, 64))
    out = model(src)
    print(f'Input:  {src.shape}  ->  Output: {out.shape}')
    print('Forward pass OK.')
