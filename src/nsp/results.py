"""Results on disk: one run, one folder.

    results/<experiment>/seed-<n>/metrics.json   scalars and small tables
    results/<experiment>/seed-<n>/arrays.npz     trajectories and curves for the figures
    results/<experiment>/seed-<n>/meta.json      config, code version, time, library versions

Parallel runs never share a file, so seeds and configs can spread across machines later.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy


def to_jsonable(obj):
    """Convert NumPy containers and scalars (recursively) to plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def git_version(cwd=None) -> str:
    """Short commit hash, with '+dirty' when the working tree differs from it; 'unknown' outside git.

    Untracked source files count as differences (new code that is not committed yet); generated outputs
    (results, notebooks, HTML exports) do not.
    """
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True,
                           check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=cwd,
                                capture_output=True, text=True).stdout.splitlines()
        generated = ("/results/", "/notebooks/", "notebooks/")
        dirty = [ln for ln in status if not any(g in ln[3:] or ln[3:].startswith(g) for g in generated)]
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_run(out_dir, metrics: dict, arrays: dict | None = None, config: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(to_jsonable(metrics), indent=1))
    if arrays:
        np.savez_compressed(out / "arrays.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    meta = {"code": git_version(out), "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "config": config}
    (out / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1))
    return out


def load_run(out_dir) -> dict:
    out = Path(out_dir)
    run = {"metrics": json.loads((out / "metrics.json").read_text()),
           "meta": json.loads((out / "meta.json").read_text()), "dir": out}
    npz = out / "arrays.npz"
    run["arrays"] = dict(np.load(npz, allow_pickle=False)) if npz.exists() else {}
    return run


def load_runs(root, experiment: str) -> list[dict]:
    """Every seed's run for one experiment, sorted by seed."""
    base = Path(root) / experiment
    dirs = sorted(base.glob("seed-*"), key=lambda p: int(p.name.split("-")[1]))
    return [load_run(p) for p in dirs]
