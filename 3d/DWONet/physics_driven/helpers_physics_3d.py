




import jax
import jax.numpy as jnp


@jax.jit
def mse_single(y_pred):
    return jnp.mean(jnp.square(y_pred))



def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jax.jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out
