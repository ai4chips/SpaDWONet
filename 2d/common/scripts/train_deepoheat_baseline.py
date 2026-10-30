#!/usr/bin/env python3
from pathlib import Path
import sys

import train_operator_baselines as runner

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
ENTRYPOINT = Path(__file__).resolve()
SHARED_ENTRYPOINT = ROOT / "scripts/train_operator_baselines.py"
DEFAULT_RUN = ROOT / "results/fig3/deepoheat_34m_20261027"


def signature(run):
    return {
        "plan_sha256": runner.sha(run / "plan.json"),
        "runner_sha256": runner.sha(ENTRYPOINT),
        "shared_runner_sha256": runner.sha(SHARED_ENTRYPOINT),
        "model_sha256": runner.sha(PACKAGE_ROOT / "DeepOHeat/model.py"),
        "guard_sha256": runner.sha(ROOT / "scripts/operator_baselines/resource_guard.py"),
        "windows_monitor_sha256": runner.sha(ROOT / "scripts/operator_baselines/windows_memory_watch.ps1"),
    }


def new_model(name, plan, torch):
    runner.check_parent()
    if name != "DeepOHeat":
        raise ValueError(f"Unexpected model in the DeepOHeat-only entrypoint: {name}")
    from DeepOHeat.model import DeepOHeat2D
    model = DeepOHeat2D(**plan["models"][name]["config"])
    allocation = model.parameter_allocation()
    if allocation != plan["models"][name]["parameter_allocation"]:
        raise ValueError(f"DeepOHeat parameter allocation differs from the plan: {allocation}")
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != plan["models"][name]["trainable_parameters"]:
        raise ValueError("DeepOHeat parameter count differs from the plan")
    runner.check_parent()
    model = model.to("cuda")
    runner.check_parent()
    return model, count


def main():
    runner.NAMES = ("DeepOHeat",)
    runner.DEFAULT_RUN = DEFAULT_RUN
    runner.signature = signature
    runner.new_model = new_model
    
    
    runner.__file__ = str(ENTRYPOINT)
    runner.main()


if __name__ == "__main__":
    main()
