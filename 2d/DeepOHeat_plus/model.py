import jax, jax.numpy as jnp, equinox as eqx, numpy as np
from typing import Sequence


def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


class MLP(eqx.Module):
    layers: tuple
    def __init__(self, layer_sizes: Sequence[int], *, key):
        keys = jax.random.split(key, len(layer_sizes)-1)
        self.layers = tuple(eqx.nn.Linear(layer_sizes[i], layer_sizes[i+1], key=keys[i], use_bias=True)
                            for i in range(len(layer_sizes)-1))
    def __call__(self, x):
        for fc in self.layers[:-1]:
            x = jnp.tanh(fc(x))
        return self.layers[-1](x)


class MLPDeepONet2d(eqx.Module):
    branch: MLP
    trunk_x: MLP
    trunk_y: MLP
    output_bias: jnp.ndarray
    nx: int = eqx.field(static=True); ny: int = eqx.field(static=True)
    r: int = eqx.field(static=True); p: int = eqx.field(static=True)
    coords_x: jnp.ndarray; coords_y: jnp.ndarray

    def __init__(self, branch_layers, trunk_layers, r=49, nx=64, ny=64, *, key):
        self.nx, self.ny, self.r = nx, ny, r
        self.p = trunk_layers[-1]
        kb, ktx, kty = jax.random.split(key, 3)

        self.branch = MLP([nx*ny] + list(branch_layers), key=kb)
        hidden = list(trunk_layers[:-1])
        self.trunk_x = MLP([1] + hidden + [self.p * r], key=ktx)
        self.trunk_y = MLP([1] + hidden + [self.p * r], key=kty)
        self.output_bias = jnp.zeros(())

        self.coords_x = jnp.linspace(0, 1, nx)
        self.coords_y = jnp.linspace(0, 1, ny)

    def __call__(self, source, coords_x=None, coords_y=None):
        if coords_x is None: coords_x = self.coords_x
        if coords_y is None: coords_y = self.coords_y

        B = source.shape[0]
        
        Bf = jax.vmap(self.branch)(source.reshape(B, -1))  
        
        phi = jax.vmap(self.trunk_x)(coords_x[:, None]).reshape(self.nx, self.r, self.p)  
        psi = jax.vmap(self.trunk_y)(coords_y[:, None]).reshape(self.ny, self.r, self.p)  
        trunk_2d = jnp.einsum('irp,jrp->ijp', phi, psi)  
        u = jnp.einsum('bp,ijp->bij', Bf, trunk_2d)       
        return u + self.output_bias
