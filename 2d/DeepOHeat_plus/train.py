#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import MLPDeepONet2d, count_params

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "2d"
OUT = Path(__file__).resolve().parent / "results" / "deepoheat_plus"
KWARGS = dict(branch_layers=[2254, 2247, 2247, 2247, 2254, 2261, 50],
              trunk_layers=[50] * 7, r=1, nx=64, ny=64)
EXPECTED_PARAMS = 34_713_711
UPDATES, BATCH, NTRAIN, NTEST = 1000, 25, 1000, 100


def load_data(path):
    arrays = {name: np.load(path / f"{name}.npy", allow_pickle=False)
              for name in ("f_train", "u_train", "f_test", "u_test")}
    if arrays["f_train"].shape != (NTRAIN, 64, 64) or arrays["u_train"].shape != (NTRAIN, 64, 64):
        raise ValueError("Expected 1000 training maps and fields on 64x64 grids")
    if arrays["f_test"].shape != (NTEST, 64, 64) or arrays["u_test"].shape != (NTEST, 64, 64):
        raise ValueError("Expected 100 held-out maps and fields on 64x64 grids")
    fmax = float(arrays["f_train"].max())
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError("Invalid training-source normalization scale")
    return arrays, fmax


def evaluate(model, arrays, fmax, batch=25):
    pred = []
    forward = eqx.filter_jit(lambda m, x: m(x))
    for start in range(0, NTEST, batch):
        x = jax.device_put(np.asarray(arrays["f_test"][start:start + batch] / fmax,
                                     dtype=np.float32))
        pred.append(np.asarray(jax.block_until_ready(forward(model, x))))
    pred = np.concatenate(pred, axis=0)
    truth = arrays["u_test"].astype(np.float64)
    err = pred.astype(np.float64) - truth
    mae = np.mean(np.abs(err), axis=(1, 2))
    peakerr = 100 * np.abs(pred.max(axis=(1, 2)) - truth.max(axis=(1, 2))) / np.maximum(np.abs(truth.max(axis=(1, 2))), 1e-12)
    return pred, mae, peakerr


def main():
    ap = argparse.ArgumentParser(description="Fixed-budget DeepOHeat+ (separable-trunk MLP) reproduction entrypoint.\n\nThis is based on the project's existing MLPDeepONet2d and the final 34.7M\nparameter experiment configuration. It is provided as reproduction source,\nnot as a claim that an independent run has been fully validated.\n")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--train", action="store_true", help="explicitly run the 1000-update training job")
    ap.add_argument("--evaluate", action="store_true", help="evaluate an existing final checkpoint")
    args = ap.parse_args()
    if args.train and args.evaluate:
        ap.error("choose --train or --evaluate")
    print(f"DeepOHeat+ MLPDeepONet2d: kwargs={KWARGS}")
    if not args.train and not args.evaluate:
        print(f"Expected trainable array count: {EXPECTED_PARAMS:,}; no training started.")
        return
    arrays, fmax = load_data(args.data_dir)
    key = jax.random.PRNGKey(0)
    model = MLPDeepONet2d(**KWARGS, key=key)
    nparams = count_params(model)
    if nparams != EXPECTED_PARAMS:
        raise ValueError(f"Parameter-count mismatch: {nparams:,} != {EXPECTED_PARAMS:,}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "deepoheat_plus_final.eqx"
    if args.evaluate:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing final checkpoint: {checkpoint}")
        model = eqx.tree_deserialise_leaves(checkpoint, model)
    else:
        if checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite {checkpoint}; choose a fresh --output-dir")
        xtrain = jax.device_put(np.asarray(arrays["f_train"] / fmax, dtype=np.float32))
        ytrain = jax.device_put(np.asarray(arrays["u_train"], dtype=np.float32))
        schedule = optax.cosine_decay_schedule(0.001, UPDATES, 0.0)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                                optax.adamw(schedule, weight_decay=1e-4))
        state = optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def step(net, opt_state, x, y):
            loss, grads = eqx.filter_value_and_grad(lambda m: jnp.mean((m(x) - y) ** 2))(net)
            updates, opt_state = optimizer.update(grads, opt_state, net)
            return eqx.apply_updates(net, updates), opt_state, loss

        rng = np.random.default_rng(0)
        losses = []
        for i in range(UPDATES):
            ids = rng.choice(NTRAIN, BATCH, replace=False)
            model, state, loss = step(model, state, xtrain[ids], ytrain[ids])
            value = float(jax.block_until_ready(loss))
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite training loss at update {i + 1}")
            losses.append(value)
            if (i + 1) % 100 == 0:
                print(f"update {i + 1}/{UPDATES}: training MSE {value:.8g}", flush=True)
        tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
        eqx.tree_serialise_leaves(tmp, model)
        tmp.replace(checkpoint)
        np.save(args.output_dir / "training_mse.npy", np.asarray(losses), allow_pickle=False)
    pred, mae, peakerr = evaluate(model, arrays, fmax)
    np.save(args.output_dir / "predictions_test.npy", pred, allow_pickle=False)
    record = {
        "status": "evaluated", "implementation_status": "organized from existing project experiment code; independent full training not verified",
        "historical_trial_note": "The prior rectangular 34.7M trial was recorded as mean-field-like with negligible input response; this rerun is not claimed numerically equivalent to its checkpoint.",
        "model": "DeepOHeat+", "model_class": "MLPDeepONet2d", "kwargs": KWARGS,
        "parameters_including_coordinate_arrays": nparams, "updates": UPDATES,
        "batch_size": BATCH, "optimizer": "AdamW", "initial_learning_rate": 0.001,
        "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
        "input_scaling": "divide by maximum of f_train", "target_scaling": "none",
        "checkpoint": str(checkpoint), "checkpoint_selection": "final update; test set evaluated only after training",
        "test_MAE_K": float(mae.mean()), "test_PeakErr_percent": float(peakerr.mean()),
        "per_sample_MAE_K": mae.tolist(), "per_sample_PeakErr_percent": peakerr.tolist(),
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Final test MAE={mae.mean():.6f}; PeakErr={peakerr.mean():.4f}%")


if __name__ == "__main__":
    main()
