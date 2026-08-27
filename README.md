# Neutral-Atom QC for Multi-Hypothesis Tracking

This repository is a compact, simulation-only tracking workflow. It creates
synthetic image sequences from explicit frame ground truth, detects objects,
builds a weighted association-conflict graph, solves the graph with either an
exact classical solver or a Pulser/QuTiP neutral-atom simulator, and applies the
selected associations through one shared tracking update.

## Quick Start

```bash
python3 -m pip install -e ".[test,notebook]"
python3 -m pytest
jupyter lab user_notebook.ipynb
```

The quantum path is optional and imported lazily:

```bash
python3 -m pip install -e ".[quantum]"
NEUTRAL_ATOM_INTEGRATION=1 python3 -m pytest tests/test_neutral_atom_integration.py -q
```

## Simulation-Only Data

`SyntheticDataGenerator` is the only data source. Every frame is built in two
steps:

1. `SyntheticScene` creates deterministic object positions, labels, and a clean
   image in a `GroundTruthFrame`.
2. `SyntheticNoiseModel` applies only `gaussian_bluriness` and `grainyness`,
   returning a `SimulatedFrame`.

```python
from neutral_atom_mht import SyntheticDataConfig, SyntheticDataGenerator

config = SyntheticDataConfig(
    frame_count=8,
    object_count=4,
    gaussian_bluriness=1.0,
    grainyness=2.0,
    seed=7,
)

generator = SyntheticDataGenerator(config)
frames = list(generator.iter_simulated_frames())
image = frames[0].image
labels = frames[0].labels
```

For tools that expect Cell Tracking Challenge-style files, materialize a
simulated sequence explicitly:

```python
dataset = generator.generate()
image = dataset.load_frame(0)
truth = dataset.load_tracking_labels(0)
```

No real dataset loader, bundled data, generated figure assets, timing analysis,
or experiment campaign is part of the package.

## Tracking API

`HPC` is the stateful controller. It owns detection, prediction, gating,
Bayesian association weights, graph encoding, result validation, and track
state advancement.

```python
from neutral_atom_mht import ClassicalSolver, HPC, HPCConfig

tracker = HPC(HPCConfig())
solver = ClassicalSolver()
sequence = tracker.run_sequence((frame.image for frame in frames), solver)
```

For inspection or notebooks, split one step into the same public stages:

```python
prepared = tracker.prepare_frame(frames[0].image, frame=0)
solver_result = tracker.solve(prepared, solver)
frame_result = tracker.advance(prepared, solver_result)
```

`prepare_frame()` does not mutate retained tracks. `advance()` validates that
the solver result matches the exact prepared-frame fingerprint before applying
Kalman and Bayesian updates.

## Solver Contract

A solver receives one immutable full-frame `SolverInput` and returns one
`SolverResult`:

```text
schema_version, problem_id, input_fingerprint, solver_name,
selected_ids, objective, feasible, status, diagnostics
```

`ComponentSolver` is the object-oriented base for solvers that factor the graph
into deterministic connected components. `ClassicalSolver` solves each bounded
component exactly. `QuantumSolver` prepares supported components for the
neutral-atom simulator, decodes sampled bitstrings back to original graph-node
IDs, and returns `completed` for usable samples because sampling is not a proof
of optimality.

## Project Layout

```text
README.md
user_notebook.ipynb
pyproject.toml
src/
  neutral_atom_mht.py    public facade
  synthetic_data.py      ground-truth-first simulator
  hpc.py                 stateful tracking controller
  solver.py              solver contract and component base class
  classical_solver.py    exact maximum-weight independent set solver
  neutral_atom.py        optional Pulser/QuTiP solver
  detection.py           deterministic frame detector
  graph.py               conflict graph encoding and components
  models.py              observations, tracks, and hypotheses
  filtering.py           Kalman prediction and update helpers
  gating.py              validation gates
  likelihood.py          Bayesian weights and posterior updates
tests/
```

## Limits

- Synthetic sequences are controlled simulations, not biological ground truth.
- Detection parameters are fixed for interpretability rather than learned.
- The exact classical solver is exponential and has a declared component cap.
- The neutral-atom path is a local simulator with a component-size cap; it is
  not a hardware timing or optimality claim.
