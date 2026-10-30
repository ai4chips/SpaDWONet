
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance_matches(saved, current):

    return saved == current


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def prepare_jax(backend):

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda" if backend == "gpu" else "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import jax
    import equinox as eqx
    import numpy as np
    jax.config.update("jax_enable_x64", False)
    devices = jax.devices()
    if backend == "gpu" and any(device.platform != "gpu" for device in devices):
        raise RuntimeError("GPU requested, but JAX did not select a GPU; refusing CPU fallback.")
    return jax, eqx, np


def environment(jax, eqx, np):
    devices = jax.devices()
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "jax": jax.__version__,
        "equinox": eqx.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "device_name": devices[0].device_kind,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def windows_headroom():

    ps_script = (
        '[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); '
        '$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; '
        '[PSCustomObject]@{TimeLocal=(Get-Date).ToString("yyyy-MM-dd HH:mm:ss");'
        'CommittedBytes=$m.CommittedBytes;CommitLimitBytes=$m.CommitLimit;'
        'CommitPercent=$m.PercentCommittedBytesInUse} | ConvertTo-Json'
    )
    if platform.system() == "Windows":
        powershell = "powershell.exe"
    elif Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").is_file():
        powershell = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    else:
        powershell = None
    if powershell:
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", ps_script],
                                capture_output=True, timeout=30, check=True)
        memory = json.loads(result.stdout.decode("utf-8-sig"))
        memory["headroom_GiB"] = (int(memory["CommitLimitBytes"]) - int(memory["CommittedBytes"])) / 1024**3
        return memory
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size
        total = os.sysconf("SC_PHYS_PAGES") * page_size
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError("Cannot read system memory headroom for the training guard.") from exc
    return {"platform": platform.platform(), "available_bytes": available, "total_bytes": total,
            "headroom_GiB": available / 1024**3,
            "measurement": "available physical memory; Windows commit headroom unavailable"}
