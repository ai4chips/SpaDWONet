#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ("DWNO", "SpaDWONet", "WNO", "FNO", "DeepOHeat_plus",
          "DeepOHeat", "ONO", "Transolver", "LNO")
DEFAULT_PREDICTIONS = {
    "DWNO": ROOT / "DWNO/results/baseline/predictions_test.npy",
    "SpaDWONet": ROOT / "SpaDWONet/results/baseline/predictions_test.npy",
    "WNO": ROOT / "WNO/results/baseline/predictions_test.npy",
    "FNO": ROOT / "FNO/results/baseline/predictions_test.npy",
    "DeepOHeat_plus": ROOT / "DeepOHeat_plus/results/deepoheat_plus/predictions_test.npy",
    "DeepOHeat": ROOT / "common/results/fig3/deepoheat_34m_20261027/DeepOHeat/train/predictions.npy",
    "ONO": ROOT / "common/results/fig3/operator_baselines_34m_20261026/ONO/train/predictions.npy",
    "Transolver": ROOT / "common/results/fig3/operator_baselines_34m_20261026/Transolver/train/predictions.npy",
    "LNO": ROOT / "common/results/fig3/operator_baselines_34m_20261026/LNO/train/predictions.npy",
}


def prediction_map(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Prediction mapping must be MODEL=PATH: {item}")
        name, raw_path = item.split("=", 1)
        if name not in MODELS:
            raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODELS)}")
        if not raw_path:
            raise ValueError(f"Missing prediction path for {name}")
        if name in result:
            raise ValueError(f"Prediction file specified more than once for {name}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def maps(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{label} must be a .npy array; got {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim == 4 and array.shape[1:] == (1, 64, 64):
        array = array[:, 0]
    elif array.ndim == 4 and array.shape[1:] == (64, 64, 1):
        array = array[..., 0]
    if array.ndim != 3 or array.shape[1:] != (64, 64):
        raise ValueError(f"{label} must have shape (N,64,64), (N,1,64,64), or (N,64,64,1); got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} contains no samples: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains NaN or infinity: {path}")
    return np.asarray(array, dtype=np.float64)


def summarize(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match target shape {truth.shape}")
    difference = pred - truth
    per_mae = np.mean(np.abs(difference), axis=(1, 2))
    truth_norm = np.sqrt(np.sum(truth * truth, axis=(1, 2)))
    per_rel_l2 = np.divide(np.sqrt(np.sum(difference * difference, axis=(1, 2))), truth_norm,
                           out=np.full_like(truth_norm, np.nan), where=truth_norm > 0)
    true_peak = np.max(truth, axis=(1, 2))
    pred_peak = np.max(pred, axis=(1, 2))
    per_peak_error = 100.0 * np.divide(np.abs(pred_peak - true_peak), true_peak,
                                       out=np.full_like(true_peak, np.nan), where=true_peak > 0)

    def stats(values):
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return {"mean": None, "std": None}
        return {"mean": float(valid.mean()), "std": float(valid.std(ddof=0))}

    return {"samples": int(pred.shape[0]), "mae": stats(per_mae),
            "relative_l2": stats(per_rel_l2), "peak_error_percent": stats(per_peak_error),
            "max_absolute_error": float(np.max(np.abs(difference)))}


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare generated held-out predictions for the nine published 2D models.\n\nThis utility evaluates existing prediction arrays only. It does not instantiate\nmodels or infer predictions from checkpoints. Arrays must have shape\n(N, 64, 64), (N, 1, 64, 64), or (N, 64, 64, 1).\n')
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data" / "2d",
                        help="directory containing u_test.npy (targets only; no model inputs are needed)")
    parser.add_argument("--target", type=Path, help="explicit target .npy file; overrides --data-dir")
    parser.add_argument("--model", choices=MODELS, action="append",
                        help="model(s) to evaluate; repeat to select several, defaults to all nine")
    parser.add_argument("--predictions", action="append", default=[], metavar="MODEL=PATH",
                        help="use an existing prediction .npy; repeat to override defaults")
    parser.add_argument("--output", type=Path, help="optional JSON output file")
    args = parser.parse_args()
    selected = args.model or list(MODELS)
    supplied = prediction_map(args.predictions)
    target_path = args.target.expanduser().resolve() if args.target else args.data_dir.expanduser().resolve() / "u_test.npy"
    truth = maps(target_path, "Target")
    results = {}
    for name in selected:
        pred_path = supplied.get(name, DEFAULT_PREDICTIONS[name])
        prediction = maps(pred_path, f"{name} prediction")
        results[name] = {"prediction_file": str(pred_path), **summarize(prediction, truth)}
    output = {"target_file": str(target_path), "metrics": results,
              "metric_note": "Per-sample MAE, relative L2 and peak absolute-value error; mean and population standard deviation. Zero-norm target samples are excluded from ratio means."}
    serialized = json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False)
    print(serialized)
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
