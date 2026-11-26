"""Shared plumbing for the study's experiment scripts: paths, the command line, saving runs.

Every script is `config and seed in, results out`:

    python experiments/a2_oriented.py                      # every seed in configs/a2.toml
    python experiments/a2_oriented.py --seed 0 --seed 1    # chosen seeds
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from nsp import config
from nsp.results import save_run

STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
CONFIGS = STUDY / "configs"


def main(experiment: str, run) -> None:
    ap = argparse.ArgumentParser(description=f"{experiment}: config and seed in, results out")
    ap.add_argument("--config", default=str(CONFIGS / f"{experiment}.toml"))
    ap.add_argument("--seed", type=int, action="append", help="seed to run (repeatable); default: [run] seeds")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    cfg = config.load(args.config)
    for seed in args.seed or cfg["run"]["seeds"]:
        t0 = time.time()
        metrics, arrays = run(cfg, seed)
        out = save_run(Path(args.out) / experiment / f"seed-{seed}", metrics, arrays, cfg)
        print(f"{experiment} seed {seed}: {time.time() - t0:.1f} s -> {out}")
