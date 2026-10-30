#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent
FILES = ("f_train", "u_train", "f_test", "u_test")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Prepare a fresh run directory from a pinned release plan and local arrays.')
    parser.add_argument("--plan", choices=("operator_baselines", "deepoheat"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty run directory: {destination}")
    expected = {"f_train": (1000, 64, 64), "u_train": (1000, 64, 64),
                "f_test": (100, 64, 64), "u_test": (100, 64, 64)}
    plan_path = ROOT / "plans" / ("operator_baselines_plan.json" if args.plan == "operator_baselines"
                                   else "deepoheat_plan.json")
    plan = json.loads(plan_path.read_text())
    for name in FILES:
        path = (args.data_dir / f"{name}.npy").resolve()
        if not path.is_file():
            raise SystemExit(f"Missing dataset array: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != expected[name] or array.dtype.kind not in "fiu":
            raise SystemExit(f"Unexpected shape or dtype for {name}: {array.shape}, {array.dtype}")
        plan["data"][name] = {"path": str(path), "sha256": digest(path)}
    plan["batch_indices_sha256"] = digest(ROOT / "batch_indices.npy")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "batch_indices.npy", destination / "batch_indices.npy")
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Prepared {args.plan} plan in {destination}")


if __name__ == "__main__":
    main()
