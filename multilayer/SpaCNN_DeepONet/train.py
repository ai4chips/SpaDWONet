#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SCRIPT = HERE.parent / "train_cnn_experiments.py"
if "--model" not in sys.argv[1:]:
    sys.argv[1:1] = ["--model", "cnn_s_deeponet"]
runpy.run_path(str(SCRIPT), run_name="__main__")
