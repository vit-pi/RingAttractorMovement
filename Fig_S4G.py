import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "stochastic-ring-attractor-simulations"
    cmd = [
        sys.executable,
        str(root / "Fig_S4GK.py"),
        "--config",
        str(root / "figureS4G_config.json"),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
