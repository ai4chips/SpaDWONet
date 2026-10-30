













import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, List, Optional




DEC_LO = jnp.array([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
                    -0.18703481171889309, -0.027983769416859854, 0.6308807679298589,
                    0.7148465705529157, 0.2303778133088965])

DEC_HI = jnp.array([-0.2303778133088965, 0.7148465705529157, -0.6308807679298589,
                    -0.027983769416859854, 0.18703481171889309, 0.030841381835560764,
                    -0.0328830116668852, -0.010597401785069032])

REC_LO = jnp.array([0.2303778133088965, 0.7148465705529157, 0.6308807679298589,
                    -0.027983769416859854, -0.18703481171889309, 0.030841381835560764,
                    0.0328830116668852, -0.010597401785069032])

REC_HI = jnp.array([-0.010597401785069032, -0.0328830116668852, 0.030841381835560764,
                    0.18703481171909309, -0.027983769416859854, -0.6308807679298589,
                    0.7148465705529157, -0.2303778133088965])


def _circular_pad_1d(x: jnp.ndarray, left: int, right: int) -> jnp.ndarray:











    length = x.shape[-1]

    indices = (jnp.arange(length + left + right) - left) % length
    return x[..., indices]


def _dwt_1d_single_level(x: jnp.ndarray,
                         dec_lo: jnp.ndarray,
                         dec_hi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

















    channels, length = x.shape
    filter_len = dec_lo.shape[0]



    dec_lo_flipped = jnp.flip(dec_lo)
    dec_hi_flipped = jnp.flip(dec_hi)




    dec_lo_conv = jnp.broadcast_to(
        dec_lo_flipped.reshape(1, 1, filter_len),
        (channels, 1, filter_len)
    )
    dec_hi_conv = jnp.broadcast_to(
        dec_hi_flipped.reshape(1, 1, filter_len),
        (channels, 1, filter_len)
    )



    left_pad = 3
    right_pad = 4
    x_padded = _circular_pad_1d(x, left=left_pad, right=right_pad)



    x_padded_batched = x_padded[None, ...]



    dimension_numbers = lax.conv_dimension_numbers(
        x_padded_batched.shape, dec_lo_conv.shape, ('NCH', 'OIH', 'NCH')
    )


    approx = lax.conv_general_dilated(
        x_padded_batched, dec_lo_conv, window_strides=(2,), padding='VALID',
        dimension_numbers=dimension_numbers,
        feature_group_count=channels
    )

    detail = lax.conv_general_dilated(
        x_padded_batched, dec_hi_conv, window_strides=(2,), padding='VALID',
        dimension_numbers=dimension_numbers,
        feature_group_count=channels
    )


    approx = approx[0]
    detail = detail[0]



    output_length = (length + 1) // 2
    approx = approx[..., :output_length]
    detail = detail[..., :output_length]

    return approx, detail


def _idwt_1d_single_level(approx: jnp.ndarray,
                          detail: jnp.ndarray,
                          rec_lo: jnp.ndarray,
                          rec_hi: jnp.ndarray) -> jnp.ndarray:
















    channels, length = approx.shape
    filter_len = rec_lo.shape[0]


    rec_lo_flipped = jnp.flip(rec_lo)
    rec_hi_flipped = jnp.flip(rec_hi)


    rec_lo_conv = jnp.broadcast_to(
        rec_lo_flipped.reshape(1, 1, filter_len),
        (channels, 1, filter_len)
    )
    rec_hi_conv = jnp.broadcast_to(
        rec_hi_flipped.reshape(1, 1, filter_len),
        (channels, 1, filter_len)
    )


    def upsample(x):

        upsampled = jnp.zeros((channels, length * 2), dtype=x.dtype)
        upsampled = upsampled.at[..., ::2].set(x)
        return upsampled

    approx_up = upsample(approx)
    detail_up = upsample(detail)



    left_pad = 4
    right_pad = 3
    approx_padded = _circular_pad_1d(approx_up, left=left_pad, right=right_pad)
    detail_padded = _circular_pad_1d(detail_up, left=left_pad, right=right_pad)


    approx_padded_batched = approx_padded[None, ...]
    detail_padded_batched = detail_padded[None, ...]


    dimension_numbers = lax.conv_dimension_numbers(
        approx_padded_batched.shape, rec_lo_conv.shape, ('NCH', 'OIH', 'NCH')
    )


    recon_approx = lax.conv_general_dilated(
        approx_padded_batched, rec_lo_conv, window_strides=(1,), padding='VALID',
        dimension_numbers=dimension_numbers,
        feature_group_count=channels
    )

    recon_detail = lax.conv_general_dilated(
        detail_padded_batched, rec_hi_conv, window_strides=(1,), padding='VALID',
        dimension_numbers=dimension_numbers,
        feature_group_count=channels
    )


    recon_approx = recon_approx[0]
    recon_detail = recon_detail[0]


    reconstructed = recon_approx + recon_detail


    output_length = length * 2
    reconstructed = reconstructed[..., :output_length]

    return reconstructed


def jax_dwt1d(x: jnp.ndarray, level: int = 1) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:














    if level <= 0:
        raise ValueError(f"level must be positive, got {level}")

    approx = x
    detail_coeffs = []

    for i in range(level):
        approx, detail = _dwt_1d_single_level(approx, DEC_LO, DEC_HI)
        detail_coeffs.append(detail)

    return approx, detail_coeffs


def jax_idwt1d(approx: jnp.ndarray, detail_coeffs: List[jnp.ndarray]) -> jnp.ndarray:












    if not detail_coeffs:
        return approx


    current_approx = approx


    for detail in reversed(detail_coeffs):
        current_approx = _idwt_1d_single_level(current_approx, detail, REC_LO, REC_HI)

    return current_approx



jax_dwt1d_jit = jax.jit(jax_dwt1d, static_argnums=(1,))
jax_idwt1d_jit = jax.jit(jax_idwt1d)
