
MODEL_NAME = "DeepOHeat"
CONFIG = {
    "size": 64,
    "branch_layers": [4096] + [2167] * 6 + [50],
    "trunk_layers": [50] + [657] * 5 + [653, 50],
    "fourier_frequencies": 25,
    "fourier_std": 1.0,
    "fourier_frequency_scale": "2*pi",
    "activation": "silu",
}
EXPECTED_ALLOCATION = {
    "branch": 32_476_879,
    "joint_trunk": 2_225_105,
    "output_bias": 1,
    "total": 34_701_985,
}
