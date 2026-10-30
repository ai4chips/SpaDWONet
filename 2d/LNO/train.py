from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "common" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import train_operator_baselines as runner


def main():
    args = sys.argv[1:]
    supplied = next((arg.split("=", 1)[1] for arg in args if arg.startswith("--model=")), None)
    if "--model" in args:
        index = args.index("--model")
        supplied = args[index + 1] if index + 1 < len(args) else None
    if supplied is not None and supplied != "LNO":
        raise SystemExit("This entrypoint trains LNO, not " + supplied)
    if supplied is None:
        args += ["--model", "LNO"]
    sys.argv = [str(Path(runner.__file__).resolve()), *args]
    runner.main()


if __name__ == "__main__":
    main()
