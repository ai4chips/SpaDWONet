#!/usr/bin/env python3






import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import equinox as eqx
import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DEFAULT = ROOT / "data" / "multilayer"
LAYERS = (1, 4, 8, 12, 16)
T_AMBIENT, T_SCALE, F_SCALE = 298.15, 100.0, 0.03
SHAPE_SOURCE, SHAPE_FIELD = (64, 64), (64, 64, 48)


CONFIG = {
    "SpaDWONet": {
        "source": HERE / "SpaDWONet",
        "module": "model.py", "class": "DWNOSeparableDeepONet3d",
        "kwargs": dict(trunk_layers=[50]*7, R=50, nx=64, ny=64, nz=48, width=64,
                        level=4, P=25, num_param_levels=1, num_wavelet_layers=10),
        "results": "SpaDWONet", "stem": "dwno_bpss_z", "extension": ".eqqx",
    },
    "DWONet": {
        "source": HERE / "DWONet", "module": "model.py",
        "class": "DWNOSeparableDeepONet3d",
        "kwargs": dict(trunk_layers=[50]*7, R=50, nx=64, ny=64, nz=48, width=64,
                        level=4, P=25, num_param_levels=1, num_wavelet_layers=10),
        "results": "DWONet", "stem": "dwno_pooled_z", "extension": ".eqqx",
    },
    "SpaCNN_DeepONet": {
        "source": HERE / "SpaCNN_DeepONet", "module": "model.py", "class": "CNNDeepONet3d",
        "kwargs": dict(trunk_layers=[50]*7, width=121, r=30, nx=64, ny=64, nz=48),
        "results": "SpaCNN_DeepONet", "checkpoint": "model.eqx",
    },
    "CNN_DeepONet": {
        "source": HERE / "CNN_DeepONet", "module": "model.py", "class": "CNNDeepONet3d",
        "kwargs": dict(trunk_layers=[50]*7, width=121, r=30, nx=64, ny=64, nz=48),
        "results": "CNN_DeepONet", "checkpoint": "model.eqx",
    },
    "Spa_DeepONet": {
        "source": HERE / "Spa_DeepONet", "module": "model_mlp_bpss.py",
        "class": "MLPDeepONet3dBPSS",
        "kwargs": dict(encoder_layers=[190,190,190,95], head_layers=[97,32,32,50],
                        trunk_layers=[50]*7, r=30, nx=64, ny=64, nz=48),
        "results": "Spa_DeepONet", "stem": "mlp_bpss_z", "extension": ".eqqx",
    },
    "DeepONet": {
        "source": HERE / "DeepONet", "module": "model_deeponet_mlp.py",
        "class": "MLPDeepONet3d",
        "kwargs": dict(branch_layers=[190,190,190,190,50], trunk_layers=[50]*7,
                        r=30, nx=64, ny=64, nz=48),
        "results": "DeepONet", "stem": "mlp_deeponet_z", "extension": ".eqqx",
    },
}


def checkpoint_path(name, z, checkpoint_root=HERE):
    c = CONFIG[name]
    filename = c["checkpoint"] if "checkpoint" in c else f"{c['stem']}{z}{c['extension']}"
    return checkpoint_root / c["results"] / "results" / f"z{z}" / filename


def load_model_class(name):
    c = CONFIG[name]
    source = c["source"]
    if not source.is_dir():
        raise FileNotFoundError(f"Model source directory is missing: {source}")

    sys.path.insert(0, str(source))
    module_path = source / c["module"]
    spec = importlib.util.spec_from_file_location(f"benchmark_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, c["class"])


def predict_superposition(name, xs, checkpoint_root):
    c = CONFIG[name]
    cls = load_model_class(name)
    models = []
    for z in LAYERS:
        path = checkpoint_path(name, z, checkpoint_root)
        if not path.is_file():
            raise FileNotFoundError(f"Missing final checkpoint for {name}, z={z}: {path}")
        model = cls(**c["kwargs"], key=jax.random.PRNGKey(0))
        models.append(eqx.tree_deserialise_leaves(path, model))

    forwards = [eqx.filter_jit(model) for model in models]
    predictions = np.empty((len(xs),) + SHAPE_FIELD, dtype=np.float64)
    started = time.perf_counter()
    for i in range(len(xs)):


        temperature_sum = np.zeros(SHAPE_FIELD, dtype=np.float64)
        for plane, forward in enumerate(forwards):
            source = jax.device_put(np.asarray(xs[i:i+1, plane] / F_SCALE, dtype=np.float32))
            normalized_rise = np.asarray(jax.block_until_ready(forward(source)))[0]
            temperature_sum += normalized_rise * T_SCALE + T_AMBIENT
        predictions[i] = temperature_sum - (len(LAYERS)-1) * T_AMBIENT
    elapsed = time.perf_counter() - started
    return predictions, elapsed


def main():
    parser = argparse.ArgumentParser(description='Evaluate five-source superposition for the six paper models.\n\nThis script evaluates only final per-plane checkpoints. It does not train or\nselect checkpoints. Use --evaluate to run inference; without it, it only lists\nexpected checkpoints and exits.\n')
    parser.add_argument("--data-dir", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--checkpoint-root", type=Path, default=HERE,
                        help="directory containing the six named model folders (default: this multilayer folder)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "evaluation")
    parser.add_argument("--model", action="append", choices=tuple(CONFIG), help="select a model; repeat to select several (default: all six)")
    parser.add_argument("--evaluate", action="store_true", help="load final checkpoints and evaluate the 100 multi-source test cases")
    args = parser.parse_args()
    names = args.model or list(CONFIG)
    if not args.evaluate:
        for name in names:
            print(name)
            for z in LAYERS:
                print(f"  {checkpoint_path(name, z, args.checkpoint_root)}")
        print("No model was loaded or evaluated. Add --evaluate to run inference.")
        return

    xs = np.load(args.data_dir / "f_test_multi.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(args.data_dir / "u_test_multi.npy", mmap_mode="r", allow_pickle=False)
    expected_x = (100, 5) + SHAPE_SOURCE
    if xs.shape != expected_x or truth.shape != (100,) + SHAPE_FIELD:
        raise ValueError(f"Unexpected data shapes: {xs.shape}, {truth.shape}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        pred, elapsed = predict_superposition(name, xs, args.checkpoint_root)
        diff = pred.astype(np.float64) - np.asarray(truth, dtype=np.float64)
        sample_mae = np.mean(np.abs(diff), axis=(1,2,3))
        sample_rl2 = np.linalg.norm(diff.reshape(len(diff), -1), axis=1) / np.linalg.norm(
            np.asarray(truth, dtype=np.float64).reshape(len(diff), -1), axis=1)
        truth_peak = np.max(truth, axis=(1,2,3))
        if np.any(truth_peak <= T_AMBIENT):
            raise ValueError("PeakErr is undefined for a test field with no rise above ambient temperature")
        pred_peak = np.max(pred, axis=(1,2,3))
        sample_peak_err = 100 * np.abs(pred_peak - truth_peak) / (truth_peak - T_AMBIENT)
        slug = name.lower()
        pred_file = args.output_dir / f"{slug}_predictions.npy"
        metrics_file = args.output_dir / f"{slug}_metrics.json"
        np.save(pred_file, pred, allow_pickle=False)
        row = {
            "model": name, "test_MAE_K": float(sample_mae.mean()),
            "test_RelL2_percent": float(sample_rl2.mean() * 100.0),
            "test_MaxAE_K": float(np.max(np.abs(diff))),
            "test_PeakErr_percent": float(sample_peak_err.mean()),
            "per_sample_MAE_K": sample_mae.tolist(),
            "per_sample_RelL2_percent": (sample_rl2 * 100.0).tolist(),
            "per_sample_PeakErr_percent": sample_peak_err.tolist(),
            "prediction_file": str(pred_file),
            "elapsed_forward_seconds_including_first_compile": elapsed,
            "checkpoint_selection": "final saved checkpoint; evaluator does not select on test metrics",
            "temperature_rule": "sum five denormalized layer predictions in float64, then subtract four ambient-temperature terms",
        }
        metrics_file.write_text(json.dumps(row, indent=2) + "\n")
        summary.append(row)
        print(f"{name}: Rel.L2 {sample_rl2.mean()*100:.4f}%; MAE {sample_mae.mean():.6f} K; PeakErr {sample_peak_err.mean():.4f}%")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
