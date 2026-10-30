#!/usr/bin/env python3







from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AMBIENT_K = 298.15
MODELS = (
    "SpaDWONet",
    "DWONet",
    "SpaCNN_DeepONet",
    "CNN_DeepONet",
    "Spa_DeepONet",
    "DeepONet",
)
PREDICTION_FILENAMES = {
    "SpaDWONet": "spadwonet_predictions.npy",
    "DWONet": "dwonet_predictions.npy",
    "SpaCNN_DeepONet": "spa_cnn_deeponet_predictions.npy",
    "CNN_DeepONet": "cnn_deeponet_predictions.npy",
    "Spa_DeepONet": "spa_deeponet_predictions.npy",
    "DeepONet": "deeponet_predictions.npy",
}


def _load_temperature_array(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load {label} array {path}: {exc}") from exc
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a numeric .npy array: {path}")
    if array.ndim != 4 or any(size <= 0 for size in array.shape):
        raise ValueError(
            f"{label} must have non-empty shape (samples, x, y, z); "
            f"got {array.shape} from {path}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinite values: {path}")
    return array


def score_prediction(prediction: np.ndarray, reference: np.ndarray, ambient_k: float = AMBIENT_K):

    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match reference shape {reference.shape}"
        )
    if not np.isfinite(ambient_k):
        raise ValueError("ambient temperature must be finite")


    sample_mae = []
    sample_peak_err = []
    for index in range(reference.shape[0]):
        pred = np.asarray(prediction[index], dtype=np.float64)
        truth = np.asarray(reference[index], dtype=np.float64)
        reference_peak = float(np.max(truth))
        reference_peak_rise = reference_peak - ambient_k
        if reference_peak_rise <= 0:
            raise ValueError(
                "PeakErr denominator max(u_test)-298.15 K must be positive for every sample; "
                f"invalid sample index: {index}"
            )
        sample_mae.append(float(np.mean(np.abs(pred - truth))))
        prediction_peak = float(np.max(pred))
        sample_peak_err.append(
            100.0 * abs(prediction_peak - reference_peak) / reference_peak_rise
        )
    return {
        "per_sample_MAE_K": sample_mae,
        "per_sample_PeakErr_percent": sample_peak_err,
        "mean_MAE_K": float(np.mean(sample_mae)),
        "mean_PeakErr_percent": float(np.mean(sample_peak_err)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Summarize MAE and peak-rise error from saved 3D temperature predictions.\n\nThis script evaluates prediction arrays only. It does not load model weights,\nrun inference, or convert normalized predictions: inputs must be temperatures\nin kelvin with shape (samples, x, y, z).\n')
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="directory containing the named prediction .npy files")
    parser.add_argument("--reference", type=Path, required=True,
                        help="reference u_test.npy in absolute kelvin")
    parser.add_argument("--model", choices=MODELS, action="append",
                        help="evaluate this model; repeat as needed (default: all six)")
    parser.add_argument("--output", type=Path,
                        help="metrics JSON path (default: <predictions-dir>/benchmark_3d_metrics.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing metrics JSON file")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reference = _load_temperature_array(args.reference, "reference")
    models = args.model or list(MODELS)
    missing = [name for name in models
               if not (args.predictions_dir / PREDICTION_FILENAMES[name]).is_file()]
    if missing:
        details = ", ".join(
            f"{name}: {args.predictions_dir / PREDICTION_FILENAMES[name]}" for name in missing
        )
        raise FileNotFoundError(f"missing prediction file(s): {details}")

    output = args.output or (args.predictions_dir / "benchmark_3d_metrics.json")
    protected_inputs = {args.reference.resolve()} | {
        (args.predictions_dir / PREDICTION_FILENAMES[name]).resolve() for name in models
    }
    if output.resolve() in protected_inputs:
        raise ValueError("metrics output must not overwrite the reference or a prediction input")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"metrics output already exists: {output} (pass --overwrite to replace)")

    results = {
        "evaluation_type": "metrics from supplied predictions; no model inference performed",
        "reference_file": str(args.reference),
        "reference_shape": list(reference.shape),
        "ambient_temperature_K": AMBIENT_K,
        "metrics": {},
    }
    for name in models:
        pred_path = args.predictions_dir / PREDICTION_FILENAMES[name]
        prediction = _load_temperature_array(pred_path, f"{name} prediction")
        scores = score_prediction(prediction, reference)
        results["metrics"][name] = {
            "prediction_file": str(pred_path),
            "shape": list(prediction.shape),
            **scores,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics: {output}")
    for name, score in results["metrics"].items():
        print(f"{name}: MAE={score['mean_MAE_K']:.6f} K; "
              f"PeakErr={score['mean_PeakErr_percent']:.6f}%")


if __name__ == "__main__":
    main()
