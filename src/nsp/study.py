"""Plumbing shared by study folders: configs, results, ledgers, logs, and building the presentation notebooks.

A study folder holds configs/<exp>.toml, experiments/<script>.py (config and seed in, results out), results/,
figures.py and build_notebooks.py. Notebooks only display and trigger; the machinery stays in `nsp`.
(Study 01 predates this module and keeps its own copy of the same plumbing.)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from . import config, viz
from .results import load_runs


class Study:
    """One study folder: where its configs, scripts and results live, and what its config keys mean."""

    def __init__(self, root, scripts: dict[str, str], describe: dict[str, str]):
        self.root = Path(root)
        self.results = self.root / "results"
        self.scripts = scripts
        self.describe = describe

    def cfg(self, exp: str) -> dict:
        return config.load(self.root / "configs" / f"{exp}.toml")

    def ensure_results(self, exp: str, rerun: bool = False) -> list[dict]:
        """Run the experiment script if its results are missing (or rerun=True), then load every seed."""
        seeds = self.cfg(exp)["run"]["seeds"]
        have = all((self.results / exp / f"seed-{s}" / "metrics.json").exists() for s in seeds)
        if rerun or not have:
            r = subprocess.run([sys.executable, str(self.root / "experiments" / f"{self.scripts[exp]}.py")],
                               capture_output=True, text=True, cwd=self.root)
            print(r.stdout.strip() or r.stderr.strip())
        runs = load_runs(self.results, exp)
        print(f"{exp}: {len(runs)} seeds loaded (code {runs[0]['meta']['code']}, run {runs[0]['meta']['time_utc']})")
        return runs

    def ledger(self, exp: str):
        c = self.cfg(exp)
        rows = [(f"world.{k}", v, self.describe.get(f"world.{k}", "")) for k, v in c["world"].items()]
        rows.append(("run.seeds", c["run"]["seeds"], self.describe.get("run.seeds", "seeds run (kept low on purpose)")))
        rows += [(k, v, self.describe.get(k, "")) for k, v in c[exp].items()]
        return viz.ledger(rows)


def log(runs: list[dict]):
    from IPython.display import Markdown
    lines = ["| Seed | Code version | Run at (UTC) | Python | NumPy | SciPy |", "| --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        m = r["meta"]
        lines.append(f"| {r['dir'].name.split('-')[1]} | `{m['code']}` | {m['time_utc']} | {m['python']} | "
                     f"{m['numpy']} | {m['scipy']} |")
    return Markdown("\n".join(lines))


def spread(values) -> str:
    """'mean (min to max)' across seeds."""
    v = np.asarray(values, float)
    if len(v) == 1:
        return viz.fmt(v[0])
    return f"{viz.fmt(v.mean())} ({viz.fmt(v.min())} to {viz.fmt(v.max())})"


def say(text: str):
    from IPython.display import display
    display(viz.finding(text))


# ------------------------------------------------------------------------------------------ building the notebooks

def md(text: str, slide: bool = False):
    import nbformat as nbf
    c = nbf.v4.new_markdown_cell(text.strip())
    if slide:
        c.metadata["slideshow"] = {"slide_type": "slide"}
    return c


def code(src: str):
    import nbformat as nbf
    return nbf.v4.new_code_cell(src.strip())


def headers(at: str, now: str, want: str, how: str):
    """The four headers every experiment notebook opens with."""
    return md(f"""**Where we're at.** {at}

**Where the system is after the last experiments.** {now}

**Where we want it.** {want}

**How this set moves it.** {how}""")


def setup_cell(exp: str, imports: str = "from nsp import config, graphs, viz, world as W"):
    return code(f"""%config InlineBackend.figure_format = 'retina'
import sys
from pathlib import Path
import numpy as np
from IPython.display import display
STUDY = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(STUDY))
import figures as F
{imports}
F.setup()
cfg = F.cfg("{exp}")
world = config.world_from(cfg)
p = cfg["{exp}"]""")


def build(study_root, notebooks: dict, names=None, execute: bool = False, html: bool = False) -> list[Path]:
    """Write each notebook from its cell list; optionally execute it in place and export code-free HTML."""
    import nbformat as nbf
    out = Path(study_root) / "notebooks"
    out.mkdir(exist_ok=True)
    paths = []
    for name, cells in notebooks.items():
        if names and name not in names and name.split("_")[0] not in names:
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        path = out / f"{name}.ipynb"
        nbf.write(nb, path)
        paths.append(path)
        print("wrote", path.relative_to(study_root))
        if execute:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=600", str(path)], check=True, cwd=out)
        if html:
            (out / "html").mkdir(exist_ok=True)
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--no-input",
                            "--output-dir", str(out / "html"), str(path)], check=True, cwd=out)
    return paths
