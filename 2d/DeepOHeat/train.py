from pathlib import Path
import runpy
import sys

_scripts = Path(__file__).resolve().parents[1] / "common" / "scripts"
sys.path.insert(0, str(_scripts))
_entry = _scripts / "train_deepoheat_baseline.py"
if __name__ == "__main__":
    sys.argv[0] = str(_entry)
    runpy.run_path(str(_entry), run_name="__main__")
