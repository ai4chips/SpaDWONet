





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






def _precompute_c_prime(n_int, a, b, c):

    cp = jnp.zeros(n_int)
    cp = cp.at[0].set(c / b)
    def body(k, arr):
        return arr.at[k].set(c / (b - a * arr[k - 1]))
    return jax.lax.fori_loop(1, n_int, body, cp)


def _thomas_const(d, a, b, c, c_prime):



    n_int = d.shape[0]
    d_prime = jnp.zeros(n_int)
    d_prime = d_prime.at[0].set(d[0] / b)

    def fwd_body(k, dp):
        m = b - a * c_prime[k - 1]
        return dp.at[k].set((d[k] - a * dp[k - 1]) / m)

    d_prime = jax.lax.fori_loop(1, n_int, fwd_body, d_prime)

    x = jnp.zeros(n_int)
    x = x.at[n_int - 1].set(d_prime[n_int - 1])

    def bwd_body(k, xs):
        idx = n_int - 2 - k
        return xs.at[idx].set(d_prime[idx] - c_prime[idx] * xs[idx + 1])

    x = jax.lax.fori_loop(0, n_int - 1, bwd_body, x)
    return x


def _compact1d_6o_first(x, dx, cp):





    n = x.shape[0]
    n_int = n - 4

    df0  = (-25 * x[0] + 48 * x[1] - 36 * x[2] + 16 * x[3] - 3 * x[4]) / (12 * dx)
    df1  = ( -3 * x[0] - 10 * x[1] + 18 * x[2] - 6 * x[3] + x[4]) / (12 * dx)
    dfn2 = (   - x[n-5] + 6 * x[n-4] - 18 * x[n-3] + 10 * x[n-2] + 3 * x[n-1]) / (12 * dx)
    dfn1 = (  3 * x[n-5] - 16 * x[n-4] + 36 * x[n-3] - 48 * x[n-2] + 25 * x[n-1]) / (12 * dx)

    rhs = (x[4:n] - x[0:n-4] + 28 * (x[3:n-1] - x[1:n-3])) / (12 * dx)
    rhs = rhs.at[0].add(-df1)
    rhs = rhs.at[-1].add(-dfn2)

    interior = _thomas_const(rhs, 1.0, 3.0, 1.0, cp)
    return jnp.concatenate([jnp.array([df0, df1]), interior, jnp.array([dfn2, dfn1])])


def _compact1d_6o_second(x, dx, cp):






    n = x.shape[0]
    n_int = n - 4
    h2 = dx * dx

    d2f0  = (35 * x[0] - 104 * x[1] + 114 * x[2] - 56 * x[3] + 11 * x[4]) / (12 * h2)
    d2f1  = (11 * x[0] -  20 * x[1] +   6 * x[2] +  4 * x[3] -      x[4]) / (12 * h2)
    d2fn2 = (    - x[n-5] + 4 * x[n-4] + 6 * x[n-3] - 20 * x[n-2] + 11 * x[n-1]) / (12 * h2)
    d2fn1 = (11 * x[n-5] - 56 * x[n-4] + 114 * x[n-3] - 104 * x[n-2] + 35 * x[n-1]) / (12 * h2)

    d1 = x[3:n-1] - 2 * x[2:n-2] + x[1:n-3]
    d2 = x[4:n]   - 2 * x[2:n-2] + x[0:n-4]
    rhs = (12 * d1 + 0.75 * d2) / h2

    rhs = rhs.at[0].add(-2 * d2f1)
    rhs = rhs.at[-1].add(-2 * d2fn2)

    interior = _thomas_const(rhs, 2.0, 11.0, 2.0, cp)
    return jnp.concatenate([jnp.array([d2f0, d2f1]), interior, jnp.array([d2fn2, d2fn1])])


def compact_diff1d_6o(f, dx, axis):











    B, nx, ny, p = f.shape
    n = f.shape[axis]
    cp = _precompute_c_prime(n - 4, 1.0, 3.0, 1.0)

    if axis == 1:
        f_t = f.transpose(0, 2, 3, 1)
        f_flat = f_t.reshape(-1, n)
        df_flat = jax.vmap(lambda x: _compact1d_6o_first(x, dx, cp))(f_flat)
        df = df_flat.reshape(B, ny, p, nx).transpose(0, 3, 1, 2)
    else:
        f_t = f.transpose(0, 1, 3, 2)
        f_flat = f_t.reshape(-1, n)
        df_flat = jax.vmap(lambda x: _compact1d_6o_first(x, dx, cp))(f_flat)
        df = df_flat.reshape(B, nx, p, ny).transpose(0, 1, 3, 2)

    return df


def compact_diff2d_6o(f, dx, axis):












    B, nx, ny, p = f.shape
    n = f.shape[axis]
    cp = _precompute_c_prime(n - 4, 2.0, 11.0, 2.0)

    if axis == 1:
        f_t = f.transpose(0, 2, 3, 1)
        f_flat = f_t.reshape(-1, n)
        df_flat = jax.vmap(lambda x: _compact1d_6o_second(x, dx, cp))(f_flat)
        df = df_flat.reshape(B, ny, p, nx).transpose(0, 3, 1, 2)
    else:
        f_t = f.transpose(0, 1, 3, 2)
        f_flat = f_t.reshape(-1, n)
        df_flat = jax.vmap(lambda x: _compact1d_6o_second(x, dx, cp))(f_flat)
        df = df_flat.reshape(B, nx, p, ny).transpose(0, 1, 3, 2)

    return df
