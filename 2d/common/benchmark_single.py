#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
SCRIPTS = ROOT / "scripts"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))
import train_operator_baselines as runner
import train_deepoheat_baseline as deepoheat_runner
from operator_baselines import resource_guard as guard

MODELS = ("DeepOHeat", "ONO", "Transolver", "LNO")
OPERATOR_MODELS = ("LNO", "Transolver", "ONO")
DEFAULT_OPERATOR_RUN = ROOT / "runs" / "operator_baselines"
DEFAULT_DEEPOHEAT_RUN = ROOT / "runs" / "deepoheat"
GLOBAL_LOCK = ROOT / ".operator_worker.lock"
QUEUE_STATUS = ROOT / "timing_single_sample_status.json"
WARMUP, REPEATS, SAMPLE = 10, 30, 0
GIB = guard.GIB


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    if path.stat().st_size == 0:
        raise ValueError(f"Empty run record requires review: {path}")
    return json.loads(path.read_text())


def run_for(model: str, operator_run: Path, deepoheat_run: Path) -> Path:
    return deepoheat_run if model == "DeepOHeat" else operator_run


def load_run_plan(model: str, run_dir: Path):
    plan_path = run_dir / "plan.json"
    plan = read_json(plan_path)
    if not plan:
        raise FileNotFoundError(f"Prepared run plan not found: {plan_path}")
    template_path = ROOT / "plans" / ("deepoheat_plan.json" if model == "DeepOHeat"
                                      else "operator_baselines_plan.json")
    template = read_json(template_path)
    for key in ("order", "settings", "models", "batch_indices_sha256", "resource_policy"):
        if plan.get(key) != template.get(key):
            raise ValueError(f"Prepared {model} plan field {key!r} differs from the published release plan")
    expected_names = ("DeepOHeat",) if model == "DeepOHeat" else OPERATOR_MODELS
    if tuple(plan.get("order", ())) != expected_names:
        raise ValueError(f"Run plan model order does not match {model}: {plan.get('order')}")
    settings = plan["settings"]
    if (settings.get("updates"), settings.get("ntrain"), settings.get("ntest"),
            settings.get("input_normalization"), settings.get("seed")) != (
            1000, 1000, 100, "power / training-set maximum", 0):
        raise ValueError(f"Unexpected completed-training plan for {model}")
    if set(plan.get("data", {})) != {"f_train", "u_train", "f_test", "u_test"}:
        raise ValueError("Run plan does not include the four required arrays")
    for entry in plan["data"].values():
        path = Path(entry["path"])
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise ValueError(f"Dataset path/hash differs from the prepared plan: {path}")
    if sha(run_dir / "batch_indices.npy") != plan["batch_indices_sha256"]:
        raise ValueError("Fixed batch-index file differs from the prepared plan")
    if model not in plan.get("models", {}):
        raise ValueError(f"Model {model} is absent from its run plan")
    return plan


def completed_model(model: str, run_dir: Path, plan):
    directory = run_dir / model / "train"
    result = read_json(directory / "training_result.json")
    status = read_json(directory / "worker_status.json")
    checkpoint = directory / "model.pt"
    predictions = directory / "predictions.npy"
    if (not result or result.get("state") != "complete" or result.get("model") != model
            or result.get("completed_updates") != 1000):
        raise ValueError(f"{model} has no complete fixed-budget training record")
    if not status or status.get("state") != "complete" or not status.get("scope_released"):
        raise ValueError(f"{model} training worker has not completed guarded cleanup")
    expected_signature = (deepoheat_runner.signature(run_dir) if model == "DeepOHeat"
                          else runner.signature(run_dir))
    if result.get("signature") != expected_signature:
        raise ValueError(f"{model} training result signature differs from the current plan/runner/guard sources")
    if not checkpoint.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"{model} final model/predictions are missing")
    expected = plan["models"][model]["trainable_parameters"]
    if result.get("trainable_parameters") != expected:
        raise ValueError(f"{model} checkpoint parameter count differs from its plan")
    maximum = result.get("training_heat_source_maximum")
    if not isinstance(maximum, (int, float)) or maximum <= 0:
        raise ValueError(f"{model} has no valid training-source normalization maximum")
    if sha(checkpoint) == "":
        raise ValueError("Cannot hash model checkpoint")
    return {"result": result, "checkpoint": checkpoint, "predictions": predictions,
            "maximum": float(maximum), "parameters": expected}


def measurement_policy(plan):
    base = dict(plan["resource_policy"])
    return guard.ResourcePolicy(**{
        **base,
        "torch_allocator_fraction": 0.85,
        "worker_memory_high_bytes": 3 * GIB,
        "worker_memory_max_bytes": 4 * GIB,
        "worker_swap_max_bytes": GIB // 2,
        "windows_start_min_commit_free_bytes": max(base.get("windows_start_min_commit_free_bytes", 0), 16 * GIB),
    })


def source_hashes(model: str):
    paths = [Path(__file__), SCRIPTS / "train_operator_baselines.py",
             SCRIPTS / "operator_baselines" / "resource_guard.py",
             SCRIPTS / "operator_baselines" / "models.py"]
    if model == "DeepOHeat":
        paths += [PACKAGE_ROOT / "DeepOHeat" / "model.py",
                  SCRIPTS / "train_deepoheat_baseline.py"]
    else:
        paths += [PACKAGE_ROOT / "LNO/lno.py", PACKAGE_ROOT / "ONO/ono.py",
                  PACKAGE_ROOT / "Transolver/transolver.py",
                  PACKAGE_ROOT / "Transolver/physics_attention.py",
                  PACKAGE_ROOT / "Transolver/embedding.py"]
    return {str(path.relative_to(PACKAGE_ROOT)): sha(path) for path in paths}


def worker(args):
    if not args.guard_token or args.guard_token != os.environ.get("OPERATOR_GUARD_TOKEN"):
        raise guard.ResourceUnavailable("Direct CUDA benchmarking is disabled; use --action queue")
    if any(value is None for value in (args.expected_checkpoint_sha256, args.expected_predictions_sha256,
                                       args.expected_training_result_sha256, args.expected_plan_sha256,
                                       args.expected_source_sha256_json)):
        raise guard.ResourceUnavailable("Worker is missing the queue-frozen provenance hashes")
    runner.check_parent()
    watchdog = runner.start_parent_watchdog()
    try:
        run_dir = args.run_dir.resolve()
        plan = load_run_plan(args.model, run_dir)
        record = completed_model(args.model, run_dir, plan)
        frozen = {
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "predictions_sha256": args.expected_predictions_sha256,
            "training_result_sha256": args.expected_training_result_sha256,
            "plan_sha256": args.expected_plan_sha256,
            "source_sha256": json.loads(args.expected_source_sha256_json),
        }
        if (sha(frozen["checkpoint"]) != frozen["checkpoint_sha256"]
                or sha(record["predictions"]) != frozen["predictions_sha256"]
                or sha(run_dir / args.model / "train" / "training_result.json") != frozen["training_result_sha256"]
                or sha(run_dir / "plan.json") != frozen["plan_sha256"]
                or source_hashes(args.model) != frozen["source_sha256"]):
            raise ValueError("Queued checkpoint, prediction, plan, training record or source hash changed before worker start")
        policy = measurement_policy(plan)
        unit = os.environ.get("OPERATOR_GUARD_UNIT")
        if not unit:
            raise guard.ResourceUnavailable("Worker has no resource-limited systemd scope")
        limits = guard.verify_worker_limits(policy, expected_unit=unit)
        
        
        plan = dict(plan)
        plan["resource_policy"] = policy.to_dict()
        torch, np = runner.setup_torch(plan)
        spec = plan["models"][args.model]
        config = spec["config"]
        if args.model == "DeepOHeat":
            from DeepOHeat.model import DeepOHeat2D
            model = DeepOHeat2D(**config)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            from operator_baselines.models import Operator2D, count_parameters
            model = Operator2D(args.model, config)
            count = count_parameters(model)
        if count != record["parameters"]:
            raise ValueError(f"Instantiated {args.model} parameter count {count} differs from plan")
        runner.check_parent()
        state = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected the final model.pt state dictionary")
        
        
        for module_name, module in model.named_modules():
            for attr, buffer in list(module._buffers.items()):
                key = f"{module_name}.{attr}" if module_name else attr
                if buffer is None and key in state:
                    module._buffers[attr] = state[key].clone()
        model.load_state_dict(state, strict=True)
        del state
        if sha(record["checkpoint"]) != frozen["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while it was being loaded")
        if source_hashes(args.model) != frozen["source_sha256"]:
            raise ValueError("Model source changed while the worker was initializing")
        model = model.to("cuda").eval()
        arrays = {key: np.load(plan["data"][key]["path"], mmap_mode="r", allow_pickle=False)
                  for key in ("f_train", "f_test")}
        maximum = float(np.max(arrays["f_train"]))
        if not np.isclose(maximum, record["maximum"], rtol=0.0, atol=0.0):
            raise ValueError("Training-data maximum differs from the recorded normalization scale")
        
        
        x_host = np.array(arrays["f_test"][SAMPLE:SAMPLE + 1], dtype=np.float32, copy=True)
        x = torch.from_numpy(x_host[:, None] / maximum).to("cuda")
        saved = np.load(record["predictions"], mmap_mode="r", allow_pickle=False)[SAMPLE:SAMPLE + 1]
        if sha(record["predictions"]) != frozen["predictions_sha256"]:
            raise ValueError("Saved predictions changed while sample zero was being loaded")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            probe = model(x)
            torch.cuda.synchronize()
            actual = probe[:, 0].detach().cpu().numpy()
            if actual.shape != (1, 64, 64) or not np.isfinite(actual).all():
                raise ValueError("Invalid sample-zero model output")
            difference = float(np.max(np.abs(actual - saved)))
            if not np.allclose(actual, saved, rtol=2e-4, atol=1e-4):
                raise ValueError(f"Sample-zero prediction differs from the saved final prediction: max abs {difference}")
            del probe, actual
            for _ in range(WARMUP):
                output = model(x)
                torch.cuda.synchronize()
                del output
            elapsed = []
            for _ in range(REPEATS):
                runner.check_parent()
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                output = model(x)
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
                del output
        if (sha(record["checkpoint"]) != frozen["checkpoint_sha256"]
                or sha(record["predictions"]) != frozen["predictions_sha256"]
                or sha(run_dir / "plan.json") != frozen["plan_sha256"]
                or source_hashes(args.model) != frozen["source_sha256"]):
            raise ValueError("Benchmark inputs or source files changed during measurement")
        output_dir = run_dir / args.model / "benchmark_single"
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "state": "complete", "kind": "single_sample_latency", "model": args.model,
            "framework": "PyTorch", "sample_index_zero_based": SAMPLE, "batch_size": 1,
            "warmup_forwards": WARMUP, "timed_forwards": REPEATS,
            "per_forward_ms": elapsed, "mean_ms": statistics.mean(elapsed),
            "std_ms": statistics.stdev(elapsed), "parameters": count,
            "normalization": "f_test / maximum(f_train)",
            "checkpoint_sha256": sha(record["checkpoint"]),
            "training_plan_sha256": sha(run_dir / "plan.json"),
            "benchmark_script_sha256": sha(Path(__file__)),
            "test_data_sha256": plan["data"]["f_test"]["sha256"],
            "max_abs_difference_from_saved_prediction": difference,
            "excluded": ["model loading", "input transfer", "sample-check forward", "warmup"],
            "precision": "FP32; AMP and TF32 disabled",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "scope": limits,
            "model_config": config, "training_result_sha256": sha(run_dir / args.model / "train" / "training_result.json"),
            "saved_predictions_sha256": sha(record["predictions"]), "source_sha256": source_hashes(args.model),
            "interpretation": "A new guarded measurement; it is not a previously measured paper value until run and reviewed.",
        }
        guard.durable_json(output_dir / "measurement.json", result)
        print(f"{args.model}: {result['mean_ms']:.6f} ms/sample; guarded measurement complete", flush=True)
    finally:
        watchdog.set()


def verify_pair(operator_run: Path, deepoheat_run: Path):
    plans = {}
    for model in MODELS:
        directory = deepoheat_run if model == "DeepOHeat" else operator_run
        plans[model] = load_run_plan(model, directory)
        completed_model(model, directory, plans[model])
    reference = plans["DeepOHeat"]
    for name in MODELS[1:]:
        plan = plans[name]
        for data_name in ("f_train", "u_train", "f_test", "u_test"):
            if plan["data"][data_name] != reference["data"][data_name]:
                raise ValueError(f"{name} and DeepOHeat use different {data_name} inputs")
        if plan["batch_indices_sha256"] != reference["batch_indices_sha256"]:
            raise ValueError("The four-model run plans do not share the fixed batch-index sequence")
    return plans


def queue(operator_run: Path, deepoheat_run: Path):
    plans = verify_pair(operator_run, deepoheat_run)
    frozen = {}
    for model in MODELS:
        run_dir = deepoheat_run if model == "DeepOHeat" else operator_run
        record = completed_model(model, run_dir, plans[model])
        frozen[model] = {
            "checkpoint_sha256": sha(record["checkpoint"]),
            "predictions_sha256": sha(record["predictions"]),
            "training_result_sha256": sha(run_dir / model / "train" / "training_result.json"),
            "plan_sha256": sha(run_dir / "plan.json"),
            "source_sha256": source_hashes(model),
        }
    for model in MODELS:
        run_dir = deepoheat_run if model == "DeepOHeat" else operator_run
        output_dir = run_dir / model / "benchmark_single"
        prior = read_json(output_dir / "measurement.json")
        if prior and prior.get("state") == "complete":
            raise FileExistsError(
                f"A complete single-sample measurement already exists for {model}: {output_dir}. "
                "This run will not reuse or overwrite a completed measurement across devices or "
                "software versions. Move the entire benchmark_single directory elsewhere to "
                "preserve it, then rerun the queue."
            )
    old = os.environ.get("OPERATOR_GUARD_LOCK_PATH")
    with guard.QueueLock(GLOBAL_LOCK):
        reader = guard.WindowsMemoryReader()
        try:
            reader.start()
            guard.assert_no_operator_scopes()
            guard.durable_json(QUEUE_STATUS, {"state": "running", "models": list(MODELS),
                                               "batch_size": 1, "warmup": WARMUP, "repeats": REPEATS,
                                               "frozen_input_hashes": frozen})
            for model in MODELS:
                run_dir = deepoheat_run if model == "DeepOHeat" else operator_run
                output_dir = run_dir / model / "benchmark_single"
                frozen_model = frozen[model]
                completed_model(model, run_dir, plans[model])
                policy = measurement_policy(plans[model])
                command = [sys.executable, str(Path(__file__).resolve()), "--action", "_worker",
                           "--model", model, "--run-dir", str(run_dir),
                           "--expected-checkpoint-sha256", frozen_model["checkpoint_sha256"],
                           "--expected-predictions-sha256", frozen_model["predictions_sha256"],
                           "--expected-training-result-sha256", frozen_model["training_result_sha256"],
                           "--expected-plan-sha256", frozen_model["plan_sha256"],
                           "--expected-source-sha256-json", json.dumps(frozen_model["source_sha256"], separators=(",", ":"))]
                os.environ["OPERATOR_GUARD_LOCK_PATH"] = str(GLOBAL_LOCK)
                guard.run_guarded(command, output_dir, policy, reader, f"{model}-single-sample")
                row = read_json(output_dir / "measurement.json")
                status = read_json(output_dir / "worker_status.json")
                if (not row or row.get("state") != "complete" or not status
                        or status.get("state") != "complete" or not status.get("scope_released")
                        or row.get("checkpoint_sha256") != frozen_model["checkpoint_sha256"]
                        or row.get("saved_predictions_sha256") != frozen_model["predictions_sha256"]
                        or row.get("training_result_sha256") != frozen_model["training_result_sha256"]
                        or row.get("training_plan_sha256") != frozen_model["plan_sha256"]
                        or row.get("source_sha256") != frozen_model["source_sha256"]):
                    raise RuntimeError(f"Guarded measurement did not complete/release resources for {model}")
            guard.durable_json(QUEUE_STATUS, {"state": "complete", "models": list(MODELS),
                                               "batch_size": 1, "warmup": WARMUP, "repeats": REPEATS,
                                               "frozen_input_hashes": frozen,
                                               "results": {name: str((deepoheat_run if name == "DeepOHeat" else operator_run) / name / "benchmark_single" / "measurement.json")
                                                           for name in MODELS}})
        except BaseException as error:
            guard.durable_json(QUEUE_STATUS, {"state": "stopped", "reason": str(error),
                                               "frozen_input_hashes": frozen})
            raise
        finally:
            reader.close()
            if old is None:
                os.environ.pop("OPERATOR_GUARD_LOCK_PATH", None)
            else:
                os.environ["OPERATOR_GUARD_LOCK_PATH"] = old


def main():
    parser = argparse.ArgumentParser(description='Guarded batch-one latency measurement for the four added 2D baselines.\n\nThe default action only reads run status. `--action queue` requires completed\ntraining runs and launches one CUDA worker at a time through the bundled\nWindows/WSL telemetry, admission, systemd/cgroup and parent-watchdog guard.\nPass `--operator-run-dir` and `--deepoheat-run-dir` with the two completed run\ndirectories. There is intentionally no direct-worker or CPU timing fallback.\n')
    parser.add_argument("--action", choices=("status", "queue", "_worker"), default="status")
    parser.add_argument("--operator-run-dir", type=Path, default=DEFAULT_OPERATOR_RUN)
    parser.add_argument("--deepoheat-run-dir", type=Path, default=DEFAULT_DEEPOHEAT_RUN)
    parser.add_argument("--model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--guard-token", help=argparse.SUPPRESS)
    parser.add_argument("--expected-checkpoint-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-predictions-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-training-result-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-plan-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-sha256-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == "_worker":
        if not args.model or not args.run_dir:
            parser.error("Internal guarded worker requires model and run directory")
        worker(args)
    elif args.action == "queue":
        queue(args.operator_run_dir.resolve(), args.deepoheat_run_dir.resolve())
    else:
        print(json.dumps({"kind": "single_sample_latency", "models": list(MODELS),
                          "warmup": WARMUP, "repeats": REPEATS,
                          "operator_run_dir": str(args.operator_run_dir.resolve()),
                          "deepoheat_run_dir": str(args.deepoheat_run_dir.resolve()),
                          "queue_status": read_json(QUEUE_STATUS),
                          "run_records": {name: read_json((args.deepoheat_run_dir if name == "DeepOHeat" else args.operator_run_dir) / name / "benchmark_single" / "measurement.json")
                                          for name in MODELS}}, indent=2))


if __name__ == "__main__":
    main()
