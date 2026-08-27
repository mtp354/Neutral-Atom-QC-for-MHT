"""Keep the root notebook concise, simulation-only, and executable."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "user_notebook.ipynb"


def _payload() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def test_notebook_has_short_clean_cell_order() -> None:
    payload = _payload()
    cells = payload["cells"]

    assert payload["nbformat"] == 4
    assert [cell["id"] for cell in cells] == [
        "overview",
        "setup",
        "simulate",
        "track",
        "inspect",
    ]
    assert cells[0]["cell_type"] == "markdown"
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    for cell in code_cells:
        compile(_source(cell), f"user_notebook:{cell['id']}", "exec")


def test_notebook_contains_only_simulation_tracking_workflow() -> None:
    source = "\n".join(_source(cell) for cell in _payload()["cells"])

    required = (
        "SyntheticDataConfig",
        "SyntheticDataGenerator",
        "iter_simulated_frames",
        "gaussian_bluriness",
        "grainyness",
        "HPC",
        "ClassicalSolver",
        "run_sequence",
    )
    forbidden = (
        "cell_data",
        "load_tiff",
        "run_overnight_benchmark",
        "OvernightBenchmarkConfig",
        "benchmark.sqlite3",
        "outputs/",
        "matplotlib",
        "savefig",
        "NeutralAtomVisualizer",
        "runtime_seconds",
    )

    assert all(token in source for token in required)
    assert all(token not in source for token in forbidden)


def test_notebook_executes_from_a_clean_kernel() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=90,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    executed = client.execute()

    assert executed["cells"][-1]["outputs"]
