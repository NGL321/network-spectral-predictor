# network-spectral-predictor
The goal of this project is to create a fully linear network approach to solving non-linear, non-stationary dynamical systems. This is a spectral graph-theoretic approach to a learning model. Areas to investigate are sheaves over the graph, reservoir approaches to predictors, and compositional multigraphs for changes in level of abstraction.

---

*Everything below this line was added by Claude on 25 November 2026; the paragraph above is Noah's.*

## Layout

```
src/nsp/                  the library: graphs, world, model, fit, lifts, detect, context, metrics, viz, results, config
tests/                    what the code is held to (pytest)
tools/build_math_notebook.py   builds notebooks/00_math_walkthrough.ipynb from the library
notebooks/                00_math_walkthrough.ipynb: the math, next to the library's own code (runs no Python)
studies/01-multibody/     step 1: pendulums on springs
    configs/              one TOML per experiment, E0 to E6
    experiments/          one script per experiment; run_all.py runs every (experiment, seed) in parallel
    results/              per experiment and seed: metrics.json, arrays.npz, meta.json (code version, time, versions)
    figures.py            the study's figures
    build_notebooks.py    builds, executes and exports the presentation notebooks
    notebooks/            e0_calibrate ... e7_capstone, executed
studies/02-hidden/        step 2: hidden information; only angles are measured, each body's sense is clamped
    configs/, experiments/, results/, figures.py, build_notebooks.py
    notebooks/            h1_clamped, executed
```

`nsp.study` holds the plumbing a study folder needs (configs, results, ledgers, logs, building the notebooks). Study 02 uses it; study 01 predates it and keeps its own copy.

## Running

```bash
pip install -e ".[notebooks,dev]"
python -m pytest                                    # the closed forms, fitting, detection and recall
cd studies/01-multibody
python experiments/run_all.py                       # every experiment and seed (or a subset: run_all.py e2 e5)
python build_notebooks.py --execute --html          # build, run and export the eight notebooks
python ../../tools/build_math_notebook.py --html    # rebuild the math walkthrough after changing the library
cd ../02-hidden
python experiments/run_all.py && python build_notebooks.py --execute --html
```

A notebook whose results are missing runs its experiment first, so `build_notebooks.py --execute` also works from a clean checkout.

## Viewing the notebooks

- Each notebook reads like a deck: title, the four status headers, a configuration ledger, the question with its prediction and pass criterion, the world, nominal vs actual, the analysis, an inspection report and a log.
- The players (play or scrub through a run, magnify the deviation) are HTML and JavaScript. They run in VS Code or Cursor, in Jupyter once the notebook has been run, and in the HTML exports (`--html` writes them to `notebooks/html/`, which is not committed).
- GitHub's notebook viewer strips scripts, so the players are blank there. Every player is followed by a static film strip that does render.

## Vocabulary

Metrology terms, inverted on purpose: **actual** is the generated world (deterministic, exact by construction) and **nominal** is the model's prediction. **Deviation** = actual − nominal, judged against a **tolerance**: in, near (within twice the tolerance) or out.

## Notes on this first build

- Three seeds per experiment, on purpose: a first visual pass, not a statistics run.
- One notebook per experiment rather than one per rung.
- Gate detection defaults to non-negative least squares: plain least squares has an exact blind spot when several identical pendulums hang free and the data is one trajectory (derived in Part 9 of the math walkthrough, measured in E5).
- E6 decides whether a structure is new with a likelihood ratio against a freshly detected structure, not an absolute fit threshold, which flagged too few new structures.
