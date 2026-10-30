#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
DEFAULT_RUN = ROOT / "results/fig3/operator_baselines_34m_20261026"
NAMES = ("LNO", "Transolver", "ONO")
sys.path.insert(0, str(ROOT / ".build/operator_deps"))

from operator_baselines.resource_guard import (
    QueueLock, ResourcePolicy, ResourceStop, ResourceUnavailable, WindowsMemoryReader,
    assert_no_operator_scopes, durable_json, run_guarded, snapshot, verify_worker_limits, violations,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        raise ValueError(f"Interrupted/empty JSON must be reviewed before reuse: {path}")
    return json.loads(path.read_text())


def policy_from(plan):
    return ResourcePolicy(**plan.get("resource_policy", {}))


def settings_for(plan, name):
    settings = dict(plan["settings"])
    override = plan.get("model_execution_overrides", {}).get(name, {})
    if set(override) - {"microbatch_size"}:
        raise ValueError("Only a documented per-model microbatch override is allowed")
    settings.update(override)
    
    
    if name == "ONO":
        settings["microbatch_size"] = 1
    return settings


def plan_for(run):
    plan = read_json(run / "plan.json")
    if not plan:
        raise ValueError("Missing experiment plan")
    settings = plan["settings"]
    if plan["order"] != list(NAMES) or not plan.get("serial_training"):
        raise ValueError("The four-model order must be explicitly serial")
    if settings["effective_batch_size"] != 25 or settings["microbatch_size"] != 5:
        raise ValueError("The original run requires effective batch 25 / microbatch 5")
    if settings["updates"] != 1000 or settings["ntrain"] != 1000 or settings["ntest"] != 100:
        raise ValueError("This experiment requires the fixed 1000-update / 1000-train / 100-test budget")
    if set(plan["models"]) != set(NAMES):
        raise ValueError("Incomplete four-model plan")
    for name, override in plan.get("model_execution_overrides", {}).items():
        if name not in NAMES or set(override) != {"microbatch_size"} or type(override["microbatch_size"]) is not int or not 1 <= override["microbatch_size"] <= 5:
            raise ValueError("Invalid per-model execution override")
        evidence = plan.get("resource_adjustments", {}).get(name, {})
        record = read_json(run / evidence.get("failure_record", "missing.json"))
        if evidence.get("reason") != "confirmed_cuda_out_of_memory" or not record or record.get("type") != "OutOfMemoryError":
            raise ValueError("A microbatch reduction requires its recorded actual CUDA OOM")
    for entry in plan["data"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Dataset differs from the saved plan: {entry['path']}")
    expected = plan.get("batch_indices_sha256")
    if not expected or sha(run / "batch_indices.npy") != expected:
        raise ValueError("The shared training-batch sequence is missing or changed")
    policy_from(plan)
    return plan


def signature(run):
    package = Path(__file__).resolve().parent / "operator_baselines"
    package_root = ROOT.parent
    methods = ("LNO/lno.py", "ONO/ono.py", "Transolver/transolver.py",
               "Transolver/physics_attention.py", "Transolver/embedding.py")
    return {"plan_sha256": sha(run / "plan.json"), "runner_sha256": sha(__file__),
            "guard_sha256": sha(package / "resource_guard.py"),
            "windows_monitor_sha256": sha(package / "windows_memory_watch.ps1"),
            "models_sha256": sha(package / "models.py"),
            "method_sources_sha256": {name: sha(package_root / name) for name in methods}}


def valid_preflight(run, name):
    data = read_json(run / name / "preflight" / "preflight.json")
    state = read_json(run / name / "preflight" / "worker_status.json")
    return bool(data and data.get("passed") is True and data.get("signature") == signature(run)
                and state and state.get("state") == "complete" and state.get("scope_released") is True)


def valid_result(run, name, updates):
    data = read_json(run / name / "train" / "training_result.json")
    state = read_json(run / name / "train" / "worker_status.json")
    return bool(data and data.get("state") == "complete" and data.get("completed_updates") == updates
                and data.get("signature") == signature(run)
                and state and state.get("state") == "complete" and state.get("scope_released") is True)


def status(run):
    plan = read_json(run / "plan.json")
    records = {}
    for name in NAMES:
        records[name] = {}
        for phase, filename in (("preflight", "preflight.json"), ("train", "progress.json"), ("train", "training_result.json")):
            path = run / name / phase / filename
            records[name][filename] = read_json(path)
    print(json.dumps({"run": str(run), "settings": plan["settings"] if plan else None,
                      "resource_policy": plan.get("resource_policy") if plan else None,
                      "models": records, "queue": read_json(run / "queue_status.json")}, ensure_ascii=False, indent=2))


def parent_tick(pid):
    
    return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[19]


def check_parent():
    pid = os.environ.get("OPERATOR_GUARD_PARENT_PID")
    expected = os.environ.get("OPERATOR_GUARD_PARENT_TICK")
    if not pid or not expected:
        raise ResourceUnavailable("CUDA workers cannot be launched directly")
    try:
        actual = parent_tick(int(pid))
    except (OSError, ValueError, IndexError) as error:
        raise ResourceUnavailable("The resource coordinator is no longer alive") from error
    if actual != expected:
        raise ResourceUnavailable("The resource coordinator process changed")
    raw_lock_path = os.environ.get("OPERATOR_GUARD_LOCK_PATH")
    if not raw_lock_path:
        raise ResourceUnavailable("The resource coordinator lock path is missing")
    lock_path = Path(raw_lock_path)
    try:
        inode = str(lock_path.stat().st_ino)
        held = any(len(fields) > 5 and fields[1] == "FLOCK" and fields[4] == str(pid)
                   and fields[5].rsplit(":", 1)[-1] == inode
                   for fields in (line.split() for line in Path("/proc/locks").read_text().splitlines()))
    except OSError as error:
        raise ResourceUnavailable("Cannot verify the coordinator's global lock") from error
    if not held:
        raise ResourceUnavailable("The coordinator no longer owns the global worker lock")


def start_parent_watchdog():
    check_parent()
    
    
    direct_parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  
        raise ResourceUnavailable("Cannot install worker parent-death protection")
    if os.getppid() != direct_parent:
        raise ResourceUnavailable("Worker launcher exited during initialization")
    stopped = threading.Event()

    def watch():
        while not stopped.wait(0.25):
            try:
                check_parent()
            except BaseException:
                os.write(2, b"Resource coordinator disappeared; terminating CUDA worker.\n")
                os._exit(125)

    threading.Thread(target=watch, name="operator-parent-watchdog", daemon=True).start()
    return stopped


def load_worker_context(args):
    if not args.guard_token or args.guard_token != os.environ.get("OPERATOR_GUARD_TOKEN"):
        raise ResourceUnavailable("Direct CUDA worker execution is disabled; use the guarded coordinator")
    check_parent()
    plan = plan_for(args.run_dir)
    unit = os.environ.get("OPERATOR_GUARD_UNIT")
    if not unit:
        raise ResourceUnavailable("The worker has no expected resource scope")
    limits = verify_worker_limits(policy_from(plan), expected_unit=unit)
    directory = args.run_dir / args.model / args.phase
    durable_json(directory / "worker_started.json", {"pid": os.getpid(), "scope": limits,
                                                    "signature": signature(args.run_dir), "time_unix": time.time()})
    return plan, limits


def setup_torch(plan):
    
    check_parent()
    import numpy as np
    import torch
    check_parent()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    seed = plan["settings"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(policy_from(plan).torch_allocator_fraction, 0)
    check_parent()
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch, np


def new_model(name, plan, torch):
    check_parent()
    from operator_baselines.models import Operator2D, count_parameters
    model = Operator2D(name, plan["models"][name]["config"])
    count = count_parameters(model)
    if count != plan["models"][name]["trainable_parameters"]:
        raise ValueError(f"Actual parameter count differs from plan: {name} {count}")
    check_parent()
    model = model.to("cuda")
    check_parent()
    return model, count


def training_arrays(plan, np):
    arrays = {key: np.load(plan["data"][key]["path"], mmap_mode="r") for key in ("f_train", "u_train")}
    shape = (plan["settings"]["ntrain"], 64, 64)
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("Unexpected training data shape")
    maximum = float(np.max(arrays["f_train"]))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid training heat-source maximum")
    return arrays, maximum


def batch_tensors(arrays, indices, maximum, torch, np, device="cuda"):
    
    power = np.array(arrays["f_train"][indices], dtype=np.float32, copy=True)[:, None] / maximum
    target = np.array(arrays["u_train"][indices], dtype=np.float32, copy=True)[:, None]
    return torch.from_numpy(power).to(device), torch.from_numpy(target).to(device)


def optimizer_for(model, plan, torch):
    settings = plan["settings"]
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                            betas=(0.9, 0.999), eps=1e-8, foreach=False)


def worker_environment(torch, plan, count, limits, name):
    settings = settings_for(plan,name)
    return {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "precision": "FP32; AMP and TF32 disabled",
            "cudnn_benchmark": False, "torch_threads": torch.get_num_threads(),
            "trainable_parameters": count, "scope": limits,
            "microbatch_size": settings["microbatch_size"],
            "effective_batch_size": settings["effective_batch_size"],
            "batch_statistic_note": f"BatchNorm/covariance statistics use physical microbatch {settings['microbatch_size']}; gradients accumulate to 25 samples.",
            "resource_policy": policy_from(plan).to_dict()}


class DisconnectedGradientError(ValueError):
    def __init__(self, names):
        self.names = names
        super().__init__("Trainable parameters disconnected from loss: " + ", ".join(names))


def optimizer_update(model, optimizer, arrays, batch, maximum, settings, step, torch, np, audit_gradients=False):
    check_parent()
    lr = settings["learning_rate"] * 0.5 * (1+math.cos(math.pi*step/settings["updates"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    loss_total = torch.zeros((), device=device)
    microbatch = settings["microbatch_size"]
    for offset in range(0, len(batch), microbatch):
        check_parent()
        subset = batch[offset:offset+microbatch]
        power, target = batch_tensors(arrays, subset, maximum, torch, np, device=device)
        prediction = model(power)
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction shape {tuple(prediction.shape)} differs from target {tuple(target.shape)}")
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at update {step+1}")
        weighted = loss * (len(subset)/len(batch))
        weighted.backward()
        loss_total.add_(weighted.detach())
        del power, target, prediction, loss, weighted
    if audit_gradients:
        missing = [name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise DisconnectedGradientError(missing)
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Short-run gradients are nonfinite")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"update": step+1, "mse": float(loss_total.item()), "learning_rate": lr,
            "gradient_norm": float(norm.item())}


def preflight_worker(args, plan, limits):
    torch, np = setup_torch(plan)
    directory = args.run_dir / args.model / "preflight"
    started = time.monotonic()
    model, count = new_model(args.model, plan, torch)
    optimizer = optimizer_for(model, plan, torch)
    arrays, maximum = training_arrays(plan, np)
    settings = settings_for(plan,args.model)
    batches = np.load(args.run_dir / "batch_indices.npy", mmap_mode="r")
    if batches.shape != (settings["updates"], settings["effective_batch_size"]):
        raise ValueError("Shared minibatch indices have the wrong shape")
    history = []
    model.train()
    for step in range(3):
        try:
            row = optimizer_update(model, optimizer, arrays, batches[step], maximum, settings, step, torch, np, audit_gradients=True)
        except DisconnectedGradientError as error:
            durable_json(directory / "gradient_audit.json", {"parameters_without_gradients": error.names})
            raise
        history.append(row)
        durable_json(directory / "short_run_progress.json", {"model":args.model,"completed_updates":step+1,
                     "effective_batch_size":25,"microbatch_size":settings["microbatch_size"],"history":history})
        print(json.dumps({"short_run":args.model, **row}), flush=True)
    model.eval()
    with torch.no_grad():
        response = []
        for index in (0, 1):
            check_parent()
            x, _ = batch_tensors(arrays, [index], maximum, torch, np)
            response.append(model(x).detach().cpu())
        response_difference = float((response[0]-response[1]).abs().mean())
    if not math.isfinite(response_difference) or response_difference == 0:
        raise ValueError("Preflight output does not respond to two distinct training inputs")
    torch.cuda.synchronize()
    record = {"passed": True, "model": args.model, "signature": signature(args.run_dir),
              "effective_batch_size":25,"microbatch_size":settings["microbatch_size"],
              "probe_uses_training_samples_only": True,"disposable_optimizer_steps":3,
              "history":history,"input_response_mae":response_difference,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "elapsed_seconds": time.monotonic()-started,
              "environment": worker_environment(torch, plan, count, limits,args.model)}
    durable_json(directory / "preflight.json", record)
    print(json.dumps(record), flush=True)


def durable_save(path, writer):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item, torch) for item in value)
    return value


def checkpoint(path, model, optimizer, completed, history, elapsed, args, torch, np):
    
    value = {"signature": signature(args.run_dir), "model_name": args.model,
             "model_state": cpu_tree(model.state_dict(), torch),
             "optimizer_state": cpu_tree(optimizer.state_dict(), torch),
             "completed_updates": completed, "history": history, "training_seconds": elapsed,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    durable_save(path, lambda stream: torch.save(value, stream))


def restore_checkpoint(path, model, optimizer, args, torch, np):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["signature"] != signature(args.run_dir) or value["model_name"] != args.model:
        raise ValueError("Checkpoint signature differs; refusing to mix experiments")
    completed = value.get("completed_updates")
    history = value.get("history")
    if type(completed) is not int or not 0 <= completed <= 1000 or not isinstance(history, list) or len(history) != completed:
        raise ValueError("Checkpoint update count/history is inconsistent")
    if any(type(row.get("update")) is not int or row.get("update") != index+1 or not math.isfinite(row.get("mse", math.nan))
           for index, row in enumerate(history)):
        raise ValueError("Checkpoint loss history is incomplete or nonfinite")
    
    
    for name, module in model.named_modules():
        for key, buffer in list(module._buffers.items()):
            state_key = f"{name}.{key}" if name else key
            if buffer is None and state_key in value["model_state"]:
                module._buffers[key] = value["model_state"][state_key].to("cuda")
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    random.setstate(value["python_rng"])
    np.random.set_state(value["numpy_rng"])
    torch.set_rng_state(value["torch_rng"])
    torch.cuda.set_rng_state_all(value["cuda_rng"])
    return value


def evaluate(model, plan, maximum, directory, torch, np):
    arrays = {key: np.load(plan["data"][key]["path"], mmap_mode="r") for key in ("f_test", "u_test")}
    ntest = plan["settings"]["ntest"]
    if any(value.shape != (ntest, 64, 64) for value in arrays.values()):
        raise ValueError("Unexpected held-out test shape")
    predictions = np.empty((ntest, 64, 64), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for index in range(ntest):
            check_parent()
            x = torch.from_numpy(np.array(arrays["f_test"][index:index+1], dtype=np.float32, copy=True)[:, None] / maximum).to("cuda")
            prediction = model(x)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite held-out prediction at sample {index}")
            predictions[index] = prediction[0, 0].cpu().numpy()
    truth = np.asarray(arrays["u_test"], dtype=np.float64)
    difference = predictions.astype(np.float64)-truth
    mae = np.mean(np.abs(difference), axis=(1, 2))
    peaks = truth.max(axis=(1, 2))
    if np.any(peaks <= 0):
        raise ValueError("PeakErr denominator is nonpositive")
    peak_error = 100 * np.abs(predictions.max(axis=(1, 2))-peaks) / peaks
    relative_l2 = np.linalg.norm(difference.reshape(ntest, -1), axis=1) / np.linalg.norm(truth.reshape(ntest, -1), axis=1)
    durable_save(directory / "predictions.npy", lambda stream: np.save(stream, predictions))
    values = {"mae_k": mae, "peak_error_percent": peak_error, "relative_l2": relative_l2}
    summary = {key: {"mean": float(value.mean()), "std": float(value.std()), "per_sample": value.tolist()}
               for key, value in values.items()}
    durable_json(directory / "test_metrics.json", {"ntest": ntest, "metrics": summary,
                 "selection": "Final fixed-budget checkpoint; no held-out tuning or checkpoint selection."})
    return summary


def train_worker(args, plan, limits):
    if not valid_preflight(args.run_dir, args.model):
        raise ValueError("A matching successful preflight is required before training")
    torch, np = setup_torch(plan)
    settings = settings_for(plan,args.model)
    directory = args.run_dir / args.model / "train"
    model, count = new_model(args.model, plan, torch)
    optimizer = optimizer_for(model, plan, torch)
    arrays, maximum = training_arrays(plan, np)
    batches = np.load(args.run_dir / "batch_indices.npy", mmap_mode="r")
    if batches.shape != (settings["updates"], settings["effective_batch_size"]):
        raise ValueError("Shared minibatch indices have the wrong shape")
    path = directory / "checkpoint.pt"
    completed, history, prior_seconds = 0, [], 0.0
    if path.exists():
        if not args.resume:
            raise ValueError("Checkpoint already exists; use --resume rather than overwriting it")
        saved = restore_checkpoint(path, model, optimizer, args, torch, np)
        completed, history, prior_seconds = saved["completed_updates"], saved["history"], saved["training_seconds"]
        del saved
    elif args.resume and (directory / "progress.json").exists():
        raise ValueError("Progress exists without a durable checkpoint; refusing a silent restart")
    durable_json(directory / "environment.json", worker_environment(torch, plan, count, limits,args.model))
    started = time.monotonic()
    model.train()
    for step in range(completed, settings["updates"]):
        row = optimizer_update(model, optimizer, arrays, batches[step], maximum, settings, step, torch, np)
        completed = step+1
        history.append(row)
        if completed == 1 or completed % 10 == 0 or completed == settings["updates"]:
            progress = {"model": args.model, "completed_updates": completed, "target_updates": settings["updates"],
                        "last": history[-1], "training_seconds": prior_seconds+time.monotonic()-started,
                        "signature": signature(args.run_dir)}
            durable_json(directory / "progress.json", progress)
            durable_json(directory / "loss_history.json", history)
            print(json.dumps(progress), flush=True)
        if completed == 1 or completed % 100 == 0 or completed == settings["updates"]:
            checkpoint(path, model, optimizer, completed, history, prior_seconds+time.monotonic()-started, args, torch, np)
    elapsed = prior_seconds+time.monotonic()-started
    metrics = evaluate(model, plan, maximum, directory, torch, np)
    durable_save(directory / "model.pt", lambda stream: torch.save(cpu_tree(model.state_dict(), torch), stream))
    durable_json(directory / "training_result.json", {
        "state": "complete", "model": args.model, "signature": signature(args.run_dir),
        "trainable_parameters": count, "completed_updates": completed, "training_seconds": elapsed,
        "effective_batch_size": settings["effective_batch_size"], "microbatch_size": settings["microbatch_size"],
        "training_heat_source_maximum": maximum, "metrics": metrics,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "budget_note": "1000 fixed optimizer updates; this does not establish convergence.",
        "environment": worker_environment(torch, plan, count, limits,args.model),
    })


def worker(args):
    directory = args.run_dir / args.model / args.phase
    watchdog = None
    try:
        plan, limits = load_worker_context(args)
        watchdog = start_parent_watchdog()
        if args.phase == "preflight":
            preflight_worker(args, plan, limits)
        else:
            train_worker(args, plan, limits)
    except BaseException as error:
        resources = {}
        loaded_torch = sys.modules.get("torch")
        if loaded_torch is not None:
            try:
                if loaded_torch.cuda.is_initialized():
                    resources = {"cuda_allocated_bytes":loaded_torch.cuda.memory_allocated(),
                                 "cuda_reserved_bytes":loaded_torch.cuda.memory_reserved(),
                                 "peak_cuda_allocated_bytes":loaded_torch.cuda.max_memory_allocated(),
                                 "peak_cuda_reserved_bytes":loaded_torch.cuda.max_memory_reserved()}
            except Exception:
                pass
        durable_json(directory / "failure.json", {"type": type(error).__name__, "reason": str(error),
                                                   "traceback": traceback.format_exc(), "time_unix": time.time(),
                                                   "resources":resources,
                                                   "note":"An allocator/cgroup cap or protection stop is not proof of exhausting physical system memory."})
        raise
    finally:
        if watchdog is not None:
            watchdog.set()


def coordinator(args):
    plan = plan_for(args.run_dir)
    policy = policy_from(plan)
    with QueueLock(ROOT / ".operator_worker.lock"):
        reader = WindowsMemoryReader()
        try:
            reader.start()
            sample = snapshot(reader, policy)
            if args.action == "check-resources":
                print(json.dumps({"snapshot": sample, "policy": policy.to_dict(),
                                  "start_blockers": violations(sample, policy, starting=True)}, indent=2))
                return
            assert_no_operator_scopes()
            
            names = ([args.model] if args.action == "queue" and args.model else list(NAMES)) if args.action == "queue" else [args.model]
            durable_json(args.run_dir / "queue_status.json", {"state": "running", "action": args.action,
                                                               "models": names, "signature": signature(args.run_dir)})
            phases = ["preflight", "train"] if args.action == "queue" else [args.action]
            for phase in phases:
                for name in names:
                    if phase == "preflight" and valid_preflight(args.run_dir, name):
                        continue
                    if phase == "train":
                        if not valid_preflight(args.run_dir, name):
                            raise ValueError(f"{name} has no matching successful preflight")
                        if valid_result(args.run_dir, name, plan["settings"]["updates"]):
                            continue
                    command = [sys.executable, str(Path(__file__).resolve()), "--action", "_worker", "--model", name,
                               "--phase", phase, "--run-dir", str(args.run_dir)]
                    if args.resume:
                        command.append("--resume")
                    assert_no_operator_scopes()
                    durable_json(args.run_dir / "queue_status.json", {
                        "state": "running", "action": args.action, "current_model": name,
                        "current_phase": phase, "time_unix": time.time(), "signature": signature(args.run_dir)})
                    run_guarded(command, args.run_dir/name/phase, policy, reader, f"{name}-{phase}", preflight=phase=="preflight")
                    if phase == "preflight" and not valid_preflight(args.run_dir, name):
                        raise ValueError(f"{name}'s worker exited without a complete matching preflight result")
                    if phase == "train" and not valid_result(args.run_dir, name, plan["settings"]["updates"]):
                        raise ValueError(f"{name}'s worker exited without a complete matching training result")
            durable_json(args.run_dir / "queue_status.json", {"state": "complete", "action": args.action,
                                                               "models": names, "signature": signature(args.run_dir)})
        except BaseException as error:
            if args.action != "check-resources":
                durable_json(args.run_dir / "queue_status.json", {"state": "stopped", "reason": str(error),
                                                                    "action": args.action, "time_unix": time.time()})
            raise
        finally:
            reader.close()


def main():
    parser = argparse.ArgumentParser(description="Guarded, serial 2D operator experiments. The default action only reads status.\n\nNo torch/model import occurs in the coordinator. CUDA workers must enter a\nbounded systemd scope and pass host-resource admission before importing torch.\nThis replaces the empty launcher left by the interrupted 2026-10-26 attempt;\nthe previous launcher's complete contents could not be recovered reliably.\n")
    parser.add_argument("--action", choices=("status", "check-resources", "preflight", "train", "queue", "_worker"),
                        metavar="{status,check-resources,preflight,train,queue}", default="status")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--model", choices=NAMES)
    parser.add_argument("--phase", choices=("preflight", "train"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--guard-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.action in {"preflight", "train", "_worker"} and not args.model:
        parser.error("Select exactly one --model for this action")
    if args.action == "_worker" and not args.phase:
        parser.error("Internal worker phase is missing")
    if args.action == "status":
        status(args.run_dir)
    elif args.action == "_worker":
        worker(args)
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
