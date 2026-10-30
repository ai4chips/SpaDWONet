#!/usr/bin/env python3









import argparse
import importlib.util
import json
import os
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from cnn_experiment_support import (digest, environment, prepare_jax, provenance_matches,
                                    windows_headroom, write_json)

DATA = Path(os.environ.get("MULTILAYER_DATA_DIR", HERE.parent/"data"/"multilayer")).resolve()
MODEL_DIR = HERE
OUTPUT = HERE
LAYERS = [1, 4, 8, 12, 16]
SPECS = {"cnn_s_deeponet": {"label": "CNN-S-DeepONet", "model_dir": "SpaCNN_DeepONet"},
         "cnn_deeponet": {"label": "CNN-DeepONet", "model_dir": "CNN_DeepONet"}}
KWARGS = dict(trunk_layers=[50]*7, width=121, r=30, nx=64, ny=64, nz=48)
SETTINGS = {"ntrain": 500, "ntest": 100, "batch_size": 25, "updates": 2500,
            "learning_rate": 0.001, "weight_decay": 1e-4, "shuffle_seed": 42,
            "initialization_key": 0, "gradient_clip_norm": 1.0,
            "optimizer": "AdamW", "schedule": "cosine decay to zero",
            "input_scale": 0.03, "ambient_K": 298.15, "temperature_scale_K": 100.0}
RSS_CAP_GIB = 2.5
START_HEADROOM_GIB = 6.0
MIN_HEADROOM_GIB = 3.0


def folder(name, z):
    return OUTPUT/SPECS[name]["model_dir"]/"results"/f"z{z}"


def signature(name, z):
    paths = [MODEL_DIR/SPECS[name]["model_dir"]/"model.py", DATA/"f_train.npy",
             DATA/f"u_train_z{z}.npy", DATA/"f_test.npy", DATA/f"u_test_z{z}.npy"]
    return {"architecture": name, "source_z_index": z, "model_kwargs": KWARGS,
            "training_settings": SETTINGS,
            "sources": [{"path": str(p), "sha256": digest(p)} for p in paths]}


def complete(name, z):
    p = folder(name, z)
    meta = p/"training_metadata.json"
    if not meta.exists():
        return False
    d = json.loads(meta.read_text())
    if d["status"] != "complete" or not provenance_matches(d["signature"], signature(name, z)):
        raise ValueError(f"Saved configuration differs: {name}, z={z}")
    if digest(p/"model.eqx") != d["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum differs")
    return True


def load_class(name):
    path = MODEL_DIR/SPECS[name]["model_dir"]/"model.py"
    spec = importlib.util.spec_from_file_location(f"fig5_{name}_model", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CNNDeepONet3d


def worker(name, z, microbatch):
    if resource is not None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20",
                       "CUDA_CACHE_PATH": "/tmp/fig5_cuda_cache",
                       "JAX_COMPILATION_CACHE_DIR": "/tmp/fig5_cnn_jax_cache",
                       "OMP_NUM_THREADS": "2"})
    jax, eqx, np = prepare_jax("gpu")
    import optax
    p = folder(name, z)
    p.mkdir(parents=True, exist_ok=True)
    sig = signature(name, z)
    cls = load_class(name)
    model = cls(**KWARGS, key=jax.random.PRNGKey(0))
    nparam = int(sum(a.size for a in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))))
    if nparam != 1_155_933:
        raise ValueError(f"CNN architecture parameter count differs: {nparam}")
    schedule = optax.cosine_decay_schedule(0.001, 2500, alpha=0.0)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    rng = np.random.RandomState(42)
    loss_log = []
    completed = 0
    resume_path = p/"resume.json"
    if resume_path.exists():
        saved = json.loads(resume_path.read_text())
        if not provenance_matches(saved["signature"], sig):
            raise ValueError("Resume data/configuration differs")
        for file, sha in saved["checkpoint_sha256"].items():
            if digest(p/file) != sha:
                raise ValueError("Resume checkpoint checksum differs")
        model = eqx.tree_deserialise_leaves(p/"resume_model.eqx", model)
        opt_state = eqx.tree_deserialise_leaves(p/"resume_optimizer.eqx", opt_state)
        state = saved["numpy_random_state"]
        rng.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        completed, loss_log = saved["completed_updates"], saved["loss_log"]
    train_x = np.load(DATA/"f_train.npy", mmap_mode="r", allow_pickle=False)
    train_y = np.load(DATA/f"u_train_z{z}.npy", mmap_mode="r", allow_pickle=False)
    if train_x.shape != (500, 64, 64) or train_y.shape != (500, 64, 64, 48):
        raise ValueError("Original multilayer training data shape differs")
    chunks = 25//microbatch

    @eqx.filter_jit
    def train_step(network, state, x, target):
        zero = jax.tree_util.tree_map(jax.numpy.zeros_like, eqx.filter(network, eqx.is_inexact_array))

        def accumulate(carry, xy):
            grads_sum, loss_sum = carry
            val, grad = eqx.filter_value_and_grad(
                lambda m: jax.numpy.mean((m(xy[0])-xy[1])**2))(network)
            grads_sum = jax.tree_util.tree_map(lambda a, b: a+b, grads_sum, grad)
            return (grads_sum, loss_sum+val), None

        (grads, loss), _ = jax.lax.scan(
            accumulate, (zero, jax.numpy.array(0., dtype=jax.numpy.float32)),
            (x.reshape(chunks, microbatch, 64, 64),
             target.reshape(chunks, microbatch, 64, 64, 48)))
        grads = jax.tree_util.tree_map(lambda a: a/chunks, grads)
        updates, state = optimizer.update(grads, state, eqx.filter(network, eqx.is_inexact_array))
        return eqx.apply_updates(network, updates), state, loss/chunks

    def save_resume():
        hashes = {}
        for filename, tree in (("resume_model.eqx", model), ("resume_optimizer.eqx", opt_state)):
            tmp = p/(filename+".tmp")
            eqx.tree_serialise_leaves(tmp, tree)
            tmp.replace(p/filename)
            hashes[filename] = digest(p/filename)
        s = rng.get_state()
        write_json(resume_path, {"signature": sig, "completed_updates": len(loss_log),
                   "microbatch_size": microbatch, "checkpoint_sha256": hashes,
                   "numpy_random_state": [s[0], s[1].tolist(), s[2], s[3], s[4]],
                   "loss_log": loss_log})

    print(f"{SPECS[name]['label']} z={z}: GPU, batch=25, microbatch={microbatch}, resume={completed}/2500", flush=True)
    started = time.monotonic()
    for step in range(completed, 2500):
        ids = rng.choice(500, 25, replace=False)
        x = jax.device_put(np.array(train_x[ids]/0.03, dtype=np.float32))
        y = jax.device_put((np.array(train_y[ids], dtype=np.float32)-298.15)/100.)
        model, opt_state, loss = train_step(model, opt_state, x, y)
        value = float(jax.block_until_ready(loss))
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite loss at update {step+1}")
        loss_log.append(value)
        if (step+1) % 100 == 0 or step == completed:
            elapsed = time.monotonic()-started
            print(f"{name} z={z} | update {step+1}/2500 | MSE {value:.8f} | {elapsed:.1f}s", flush=True)
            write_json(p/"progress.json", {"model": SPECS[name]["label"], "z": z,
                       "completed_updates": step+1, "target_updates": 2500,
                       "loss": value, "elapsed_seconds_this_worker": elapsed})
        if (step+1) % 250 == 0:
            save_resume()
        del x, y
    final_path = p/"model.eqx"
    tmp = p/"model.eqx.tmp"
    eqx.tree_serialise_leaves(tmp, model)
    tmp.replace(final_path)
    np.save(p/"train_loss.npy", np.asarray(loss_log), allow_pickle=False)

    @eqx.filter_jit
    def forward(m, x):
        return m(x)

    tx = np.load(DATA/"f_test.npy", mmap_mode="r", allow_pickle=False)
    ty = np.load(DATA/f"u_test_z{z}.npy", mmap_mode="r", allow_pickle=False)
    errors = []
    for first in range(0, 100, 5):
        pred = np.asarray(jax.block_until_ready(forward(model, jax.device_put(
            np.array(tx[first:first+5]/0.03, dtype=np.float32)))))*100.+298.15
        errors.extend(np.mean(np.abs(pred-np.asarray(ty[first:first+5])), axis=(1, 2, 3)).tolist())
    write_json(p/"training_metadata.json", {
        "status": "complete", "model": SPECS[name]["label"], "signature": sig,
        "completed_updates": len(loss_log), "parameters": nparam, "microbatch_size": microbatch,
        "effective_batch_size": 25, "gradient_accumulation_chunks": chunks,
        "gradient_clipping_and_optimizer_steps": "Once per effective batch, after gradient averaging",
        "checkpoint_sha256": digest(final_path), "per_sample_test_MAE_K": errors,
        "test_MAE_K": float(np.mean(errors)), "environment": environment(jax, eqx, np),
        "elapsed_seconds_this_worker": time.monotonic()-started,
        "training_script": str(Path(__file__).resolve()), "training_script_sha256": digest(Path(__file__)),
        "checkpoint_selection": "Final update 2500; test set is not used for model selection",
    })
    print(f"COMPLETE {name} z={z}: test MAE {np.mean(errors):.6f} K", flush=True)


def guarded(name, z, microbatch=5):
    if complete(name, z):
        print(f"Reuse complete checkpoint: {name} z={z}", flush=True)
        return
    before = windows_headroom()
    if before["headroom_GiB"] < START_HEADROOM_GIB:
        raise RuntimeError(f"Windows headroom {before['headroom_GiB']:.2f} GiB below {START_HEADROOM_GIB} GiB; no worker started")
    p = folder(name, z)
    p.mkdir(parents=True, exist_ok=True)
    if (p/"resources.json").exists():
        history_path = p/"resources_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        previous = json.loads((p/"resources.json").read_text())
        if previous not in history:
            write_json(history_path, history+[previous])
    log = (p/"run_log.txt").open("a", buffering=1)
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", name,
                             "--z", str(z), "--microbatch", str(microbatch)], cwd=ROOT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
                            stdout=log, stderr=subprocess.STDOUT)
    started = checked = time.monotonic()
    peak = 0
    try:
        while proc.poll() is None:
            try:
                d = dict(line.split(":", 1) for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines() if ":" in line)
                rss = sum(int(d.get(k, "0 kB").split()[0]) for k in ("VmRSS", "VmSwap"))*1024
                peak = max(peak, rss)
                if rss > RSS_CAP_GIB*1024**3:
                    raise RuntimeError(f"Training worker reached the {RSS_CAP_GIB} GiB RSS+swap cap")
            except FileNotFoundError:
                pass
            if time.monotonic()-started > 3600:
                raise RuntimeError("Layer training exceeded one hour")
            if time.monotonic()-checked >= 30:
                memory = windows_headroom()
                if memory["headroom_GiB"] < MIN_HEADROOM_GIB:
                    raise RuntimeError(f"Windows headroom below {MIN_HEADROOM_GIB} GiB")
                checked = time.monotonic()
            time.sleep(0.25)
        if proc.returncode:
            raise RuntimeError(f"Training worker exited with {proc.returncode}; see {p/'run_log.txt'}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()
        write_json(p/"resources.json", {"windows_before": before,
                   "peak_RSS_plus_swap_MiB": peak/1024**2, "rss_cap_GiB": RSS_CAP_GIB,
                   "gpu_memory_fraction": 0.2, "single_worker": True, "microbatch_size": microbatch,
                   "wall_seconds": time.monotonic()-started, "returncode": proc.returncode})
    if not complete(name, z):
        raise RuntimeError("No verified final checkpoint after worker completion")
    print(f"Completed {name} z={z}; peak RSS+swap {peak/1024**2:.0f} MiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the two CNN architectures on each of five source layers.\n\nCode and outputs stay inside this publication package. --train runs guarded GPU\nworkers sequentially; default only reports progress. An effective batch of\n25 is retained via five microbatches of five, one gradient clip and AdamW\nupdate per batch. Models use per-sample GroupNorm, so accumulation preserves\nthe batch objective. Original source files and weights are read-only.\n')
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model", choices=list(SPECS))
    parser.add_argument("--worker", choices=list(SPECS), help=argparse.SUPPRESS)
    parser.add_argument("--z", type=int, choices=LAYERS)
    parser.add_argument("--microbatch", type=int, choices=[1, 5, 25], default=5)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.z, args.microbatch)
    else:
        for name in ([args.model] if args.model else SPECS):
            for z in ([args.z] if args.z else LAYERS):
                if args.train:
                    guarded(name, z, args.microbatch)
                else:
                    print(name, z, "complete" if complete(name, z) else "pending")
