"""Run every experiment of the study, one process per (experiment, seed), in parallel.

    python experiments/run_all.py              # everything
    python experiments/run_all.py e2 e5        # a subset
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nsp import config

HERE = Path(__file__).resolve().parent
EXPERIMENTS = {"e0": "e0_calibrate", "e1": "e1_decompose", "e2": "e2_transfer", "e3": "e3_reconfigure",
               "e4": "e4_large_swings", "e5": "e5_detect", "e6": "e6_recall", "e7": "e7_capstone"}


def _run(name: str, seed: int) -> str:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / f"{EXPERIMENTS[name]}.py"), "--seed", str(seed)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{name} seed {seed} failed:\n{r.stderr}")
    return f"{name} seed {seed}: {time.time() - t0:.1f} s"


def main(names: list[str]) -> None:
    jobs = [(name, seed) for name in names
            for seed in config.load(HERE.parent / "configs" / f"{name}.toml")["run"]["seeds"]]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        for line in pool.map(lambda job: _run(*job), jobs):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
