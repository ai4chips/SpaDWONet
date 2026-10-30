from __future__ import annotations

import base64
from collections import deque
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourcePolicy:
    sample_interval_seconds: float = 1.0
    telemetry_max_age_seconds: float = 4.0
    windows_start_max_commit_fraction: float = 0.75
    windows_stop_max_commit_fraction: float = 0.85
    windows_start_min_commit_free_bytes: int = 16 * GIB
    windows_stop_min_commit_free_bytes: int = 8 * GIB
    windows_start_min_physical_free_bytes: int = 6 * GIB
    windows_stop_min_physical_free_bytes: int = 4 * GIB
    linux_start_min_available_bytes: int = 6 * GIB
    linux_stop_min_available_bytes: int = 4 * GIB
    gpu_start_min_free_bytes: int = 6 * GIB
    gpu_stop_min_free_bytes: int = GIB
    worker_memory_high_bytes: int = 7 * GIB
    worker_memory_max_bytes: int = 8 * GIB
    worker_swap_max_bytes: int = GIB
    worker_tasks_max: int = 256
    torch_allocator_fraction: float = 0.70
    worker_timeout_seconds: float = 1800.0  

    def __post_init__(self):
        if not 0 < self.windows_start_max_commit_fraction < self.windows_stop_max_commit_fraction < 1:
            raise ValueError("Windows start/stop thresholds must provide headroom and hysteresis")
        if not 0 < self.torch_allocator_fraction < 1:
            raise ValueError("Invalid PyTorch allocator fraction")
        if self.worker_memory_high_bytes >= self.worker_memory_max_bytes:
            raise ValueError("MemoryHigh must be less than MemoryMax")

    def to_dict(self):
        return asdict(self)


class ResourceUnavailable(RuntimeError):
    pass


class ResourceStop(RuntimeError):
    pass


def durable_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
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


def durable_json(path, value):
    durable_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def append_event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode())
        os.fsync(stream.fileno())


class QueueLock:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise ResourceUnavailable("Another operator queue owns the global worker lock") from error
        return self

    def __exit__(self, *args):
        self.stream.close()


class WindowsMemoryReader:
    def __init__(self, script=None, executable=None):
        self.script = Path(script or Path(__file__).with_name("windows_memory_watch.ps1"))
        self.executable = executable or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        self.process = None
        self.latest = None
        self.received_at = 0.0
        self.errors = deque(maxlen=8)
        self.mutex = threading.Lock()

    def start(self, timeout=15.0):
        encoded = base64.b64encode(self.script.read_text(encoding="utf-8").encode("utf-16-le")).decode("ascii")
        try:
            self.process = subprocess.Popen(
                [self.executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            raise ResourceUnavailable(f"Cannot start Windows memory monitor: {error}") from error
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.mutex:
                if self.latest is not None:
                    return self
            if self.process.poll() is not None:
                break
            time.sleep(0.05)
        error = " | ".join(self.errors)
        self.close()
        raise ResourceUnavailable("Windows commit telemetry is unavailable; refusing to load CUDA. " + error)

    def _read_stdout(self):
        for line in self.process.stdout:
            try:
                data = json.loads(line.lstrip("\ufeff").strip())
                required = ("commit_used_bytes", "commit_limit_bytes", "physical_available_bytes", "physical_total_bytes")
                if any(not isinstance(data.get(key), (int, float)) for key in required):
                    raise ValueError("missing memory counters")
                if data["commit_limit_bytes"] <= 0 or data["physical_total_bytes"] <= 0:
                    raise ValueError("invalid memory counter denominator")
                with self.mutex:
                    self.latest = data
                    self.received_at = time.monotonic()
            except (ValueError, TypeError) as error:
                self.errors.append(f"Invalid Windows telemetry: {error}")

    def _read_stderr(self):
        for line in self.process.stderr:
            self.errors.append(line.strip())

    def read(self, max_age):
        with self.mutex:
            age = time.monotonic() - self.received_at
            if self.latest is None or age > max_age or self.process.poll() is not None:
                raise ResourceUnavailable("Windows memory monitor stopped or its sample became stale")
            return {**self.latest, "sample_age_seconds": age}

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def linux_memory(path="/proc/meminfo"):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.split()[0]) * 1024
    return {"available_bytes": values["MemAvailable"], "total_bytes": values["MemTotal"],
            "swap_free_bytes": values["SwapFree"], "swap_total_bytes": values["SwapTotal"]}


def gpu_memory(executable="/usr/lib/wsl/lib/nvidia-smi", device=0):
    try:
        result = subprocess.run(
            [executable, f"--id={device}", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=3,
        )
        lines = result.stdout.strip().splitlines()
        if len(lines) != 1:
            raise ValueError("Expected exactly one selected GPU")
        total, used, free = (int(x.strip()) * 1024 ** 2 for x in lines[0].split(","))
        return {"total_bytes": total, "used_bytes": used, "free_bytes": free}
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise ResourceUnavailable(f"GPU memory telemetry is unavailable: {error}") from error


def snapshot(reader, policy):
    
    
    linux, gpu = linux_memory(), gpu_memory()
    return {"time_unix": time.time(), "windows": reader.read(policy.telemetry_max_age_seconds),
            "linux": linux, "gpu": gpu}


def violations(sample, policy, starting=False):
    win, linux, gpu = sample["windows"], sample["linux"], sample["gpu"]
    phase = "start" if starting else "stop"
    ratio = win["commit_used_bytes"] / win["commit_limit_bytes"]
    free_commit = win["commit_limit_bytes"] - win["commit_used_bytes"]
    checks = [
        (win.get("sample_age_seconds", 0) > policy.telemetry_max_age_seconds,
         "Windows commit telemetry is stale"),
        (ratio >= getattr(policy, f"windows_{phase}_max_commit_fraction"),
         f"Windows commit is {ratio:.1%}"),
        (free_commit < getattr(policy, f"windows_{phase}_min_commit_free_bytes"),
         f"Windows commit headroom is {free_commit/GIB:.2f} GiB"),
        (win["physical_available_bytes"] < getattr(policy, f"windows_{phase}_min_physical_free_bytes"),
         f"Windows physical memory available is {win['physical_available_bytes']/GIB:.2f} GiB"),
        (linux["available_bytes"] < getattr(policy, f"linux_{phase}_min_available_bytes"),
         f"WSL memory available is {linux['available_bytes']/GIB:.2f} GiB"),
        (gpu["free_bytes"] < getattr(policy, f"gpu_{phase}_min_free_bytes"),
         f"GPU memory free is {gpu['free_bytes']/GIB:.2f} GiB"),
    ]
    return [reason for failed, reason in checks if failed]


def scope_command(command, unit, policy):
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}",
            f"--property=MemoryHigh={policy.worker_memory_high_bytes}",
            f"--property=MemoryMax={policy.worker_memory_max_bytes}",
            f"--property=MemorySwapMax={policy.worker_swap_max_bytes}",
            f"--property=TasksMax={policy.worker_tasks_max}", "--", *command]


def verify_worker_limits(policy, proc_cgroup="/proc/self/cgroup", cgroup_root="/sys/fs/cgroup", expected_unit=None):
    entry = next((line.split(":", 2)[2] for line in Path(proc_cgroup).read_text().splitlines()
                  if line.startswith("0::")), None)
    leaf = Path(entry).name if entry else ""
    if not leaf.startswith("operator-") or not leaf.endswith(".scope"):
        raise ResourceUnavailable("Worker is outside its dedicated operator cgroup")
    if expected_unit is not None and leaf != expected_unit:
        raise ResourceUnavailable("Worker is in a different scope from the coordinator's expected unit")
    directory = Path(cgroup_root) / entry.lstrip("/")
    limits = {}
    for key, maximum in (("memory.high", policy.worker_memory_high_bytes),
                         ("memory.max", policy.worker_memory_max_bytes),
                         ("memory.swap.max", policy.worker_swap_max_bytes),
                         ("pids.max", policy.worker_tasks_max)):
        raw = (directory / key).read_text().strip()
        if raw == "max" or int(raw) > maximum:
            raise ResourceUnavailable(f"Worker cgroup has no required hard limit: {key}={raw}")
        limits[key] = int(raw)
    return {"path": str(directory), "limits": limits}


def scope_empty(unit):
    try:
        result = subprocess.run(["systemctl", "--user", "show", "--property=ActiveState,ControlGroup", unit],
                                capture_output=True, text=True, timeout=2, check=False)
        if result.returncode == 4:  
            return True
        if result.returncode:
            return False
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        group = values.get("ControlGroup", "")
        if group:
            events = Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.events"
            if events.exists():
                pairs = dict(line.split() for line in events.read_text().splitlines())
                return pairs.get("populated") == "0"
        return not group and values.get("ActiveState") in {"inactive", "failed"}
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def assert_no_operator_scopes():
    try:
        result = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--type=scope", "--plain", "--no-legend",
             "--no-pager", "operator-*.scope"], capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise ResourceUnavailable(f"Cannot check for workers left by an earlier coordinator: {error}") from error
    units = [line.split()[0] for line in result.stdout.splitlines() if line.split()]
    populated = [unit for unit in units if not scope_empty(unit)]
    if populated:
        raise ResourceUnavailable("An earlier operator scope is still populated or unverifiable: " + ", ".join(populated))


def worker_cgroup_memory(directory, unit):
    try:
        started = json.loads((Path(directory)/"worker_started.json").read_text())
        group = Path(started["scope"]["path"])
        if group.name != unit or not group.is_relative_to("/sys/fs/cgroup"):
            return None
        data = {key:int((group/key).read_text()) for key in ("memory.current","memory.peak","memory.swap.current")}
        data["memory.events"] = {key:int(value) for key,value in
                                 (line.split() for line in (group/"memory.events").read_text().splitlines())}
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def terminate_scope(process, unit):
    diagnostics = []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            subprocess.run(["systemctl", "--user", "kill", "--kill-whom=all", f"--signal={sig.name}", unit],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            diagnostics.append(f"{sig.name} unit signal failed: {error}")
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        except OSError as error:
            diagnostics.append(f"{sig.name} process-group signal failed: {error}")
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if scope_empty(unit):
                return {"scope_empty": True, "diagnostics": diagnostics}
            time.sleep(0.1)
    return {"scope_empty": scope_empty(unit), "diagnostics": diagnostics}


def run_guarded(command, directory, policy, reader, label, preflight=False):
    directory = Path(directory)
    sample = snapshot(reader, policy)
    reasons = violations(sample, policy, starting=True)
    append_event(directory / "resources.jsonl", {"event": "launch_gate", "snapshot": sample, "reasons": reasons})
    if reasons:
        raise ResourceStop("Start blocked: " + "; ".join(reasons))
    unit = "operator-" + label.lower().replace("_", "-") + "-" + uuid.uuid4().hex[:12] + ".scope"
    token = uuid.uuid4().hex
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               OPERATOR_GUARD_TOKEN=token, OPERATOR_GUARD_PARENT_PID=str(os.getpid()),
               OPERATOR_GUARD_UNIT=unit,
               OPERATOR_GUARD_PARENT_TICK=Path(f"/proc/{os.getpid()}/stat").read_text().rpartition(")")[2].split()[19])
    launch = scope_command([*command, "--guard-token", token], unit, policy)
    status_path = directory / "worker_status.json"
    durable_json(status_path, {"state": "launching", "unit": unit, "policy": policy.to_dict(), "time_unix": time.time()})
    started = time.monotonic()
    process = None
    try:
        with (directory / "worker.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(launch, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            durable_json(status_path, {"state": "running", "unit": unit, "launcher_pid": process.pid,
                                       "time_unix": time.time(), "policy": policy.to_dict()})
            while process.poll() is None:
                sample = snapshot(reader, policy)
                append_event(directory / "resources.jsonl", {"event": "running", "snapshot": sample,
                                                             "worker_cgroup":worker_cgroup_memory(directory,unit)})
                reasons = violations(sample, policy)
                if preflight and time.monotonic() - started > policy.worker_timeout_seconds:
                    reasons.append("Preflight exceeded its time limit")
                if reasons:
                    raise ResourceStop("Resource stop: " + "; ".join(reasons))
                time.sleep(policy.sample_interval_seconds)
            result = process.returncode
            if result:
                raise ResourceStop(f"Bounded worker exited with code {result}; see {directory/'worker.log'}")
        durable_json(status_path, {"state": "worker_exited", "unit": unit, "time_unix": time.time(),
                                   "elapsed_seconds": time.monotonic()-started})
        empty_deadline = time.monotonic() + 5
        while not scope_empty(unit):
            if time.monotonic() >= empty_deadline:
                raise ResourceStop("Launcher exited but the worker scope is still populated or cannot be verified")
            time.sleep(0.1)
        
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            sample = snapshot(reader, policy)
            if not violations(sample, policy, starting=True):
                durable_json(status_path, {"state": "complete", "scope_released": True, "unit": unit,
                                           "time_unix": time.time(), "elapsed_seconds": time.monotonic()-started})
                return
            time.sleep(policy.sample_interval_seconds)
        raise ResourceStop("Worker exited, but resources did not return to the next-worker start thresholds")
    except BaseException as error:
        cleanup = None
        if process is not None:
            cleanup = terminate_scope(process, unit)
        durable_json(status_path, {"state": "stopped", "unit": unit, "reason": str(error),
                                   "time_unix": time.time(), "elapsed_seconds": time.monotonic()-started,
                                   "cleanup": cleanup})
        append_event(directory / "resources.jsonl", {"event": "stopped", "reason": str(error),
                                                     "time_unix": time.time(), "cleanup": cleanup})
        raise
