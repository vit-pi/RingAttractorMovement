import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "Fig_S4"
    cmd = [
        sys.executable,
        str(root / "Fig_S4CFJ.py"),
        "--config",
        str(root / "figureS4F_config.json"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
