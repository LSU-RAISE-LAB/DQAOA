# DQAOA-QUBO Simulator

A Qiskit-compatible research simulator for solving quadratic unconstrained binary optimization (QUBO) problems with classical baselines, monolithic QAOA, and distributed QAOA execution models.

The software accepts a dense quadratic matrix `H`, a linear vector `f`, and a constant `c0`, canonicalizes the QUBO, constructs QAOA circuits, allocates variables across user-configurable QPUs, identifies local and cross-QPU couplings, optimizes variational parameters, and reports sampled distribution metrics in a common workflow.

## Main capabilities

- Exact brute-force reference for small QUBOs.
- Optional CPLEX MIQP reference.
- Monolithic QAOA on one QPU.
- Abstract distributed QAOA with cross-QPU interaction accounting.
- Explicit TeleGate-style distributed QAOA with communication qubits, mid-circuit measurement, reset, and classical feedforward.
- Contiguous, graph-aware, and manual variable-to-QPU allocation.
- Parameterized-circuit reuse and transpile-once/bind-many execution.
- Batched circuit evaluation and parallel multi-start optimization.
- SPSA-gradient Adam optimization, depth progression, random restarts, and warm starts.
- Exact-reference, convergence, elite-bitstring, consensus, and distribution-level metrics.
- Plotting, Qiskit circuit rendering, paper-style schematics, and fixed-angle validation.

## Repository layout

```text
dqaoa-qubo-simulator/
├── dqaoa_qubo_multi_mode_callable_v5_batched_parallel.py  # implementation
├── dqaoa_qubo.py                                           # short import alias
├── examples/                                               # runnable examples
├── tests/                                                  # dependency-light tests
├── docs/                                                   # mode, API, configuration, and troubleshooting docs
├── outputs/                                                # generated files; ignored by Git
├── pyproject.toml                                          # installable package metadata
├── requirements.txt                                        # Qiskit + plotting environment
├── requirements-optional-cplex.txt                         # optional CPLEX modeling layer
├── environment.yml                                         # Conda environment
├── CITATION.cff
└── LICENSE
```

## Requirements

Recommended:

- Python 3.10–3.12
- NumPy
- Qiskit 2.4.2
- Qiskit Aer 0.17.2
- Matplotlib
- `pylatexenc` for Qiskit Matplotlib circuit drawings
- Optional: DOcplex and a compatible IBM ILOG CPLEX runtime

The simulator catches missing optional dependencies at import time. Classical brute force can still be used when Qiskit, Matplotlib, or DOcplex is absent.

## Installation

### Option A: Conda on Windows, macOS, or Linux

```bash
conda env create -f environment.yml
conda activate dqaoa-qubo
pip install -e .
```

### Option B: Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Optional CPLEX mode

```bash
pip install -r requirements-optional-cplex.txt
```

`docplex` alone is not the CPLEX optimizer. A compatible IBM ILOG CPLEX runtime and license must also be available. Remove `miqp_cplex` from `modes` when CPLEX is unavailable.

## Quick start

```bash
python examples/quickstart.py
```

Minimal Python usage:

```python
import numpy as np

from dqaoa_qubo import (
    AdamConfig,
    MultiStartConfig,
    TrainConfig,
    compare_qubo_modes,
    print_mode_comparison_table,
)

H = np.array([
    [-0.5, 0.8, 0.0],
    [0.0, -0.2, 0.6],
    [0.0, 0.0, 0.1],
])
f = np.array([0.1, -0.4, 0.2])

results = compare_qubo_modes(
    H=H,
    f=f,
    c0=0.0,
    modes=["bruteforce", "monolithic_qaoa", "telegate_explicit_qaoa"],
    p_max=1,
    num_qpus=2,
    capacities=[2, 1],
    train_cfg=TrainConfig(shots_train=128),
    final_shots=256,
    adam_cfg=AdamConfig(iters=2, seed=123),
    multistart_cfg=MultiStartConfig(num_random_starts=2),
)

print_mode_comparison_table(results)
```

The original long module name remains supported:

```python
from dqaoa_qubo_multi_mode_callable_v5_batched_parallel import compare_qubo_modes
```

## QUBO input convention

The accepted dense input is

```text
F(z) = c0 + f^T z + z^T H z,    z in {0,1}^n.
```

Internally, it is converted to

```text
F(z) = c0 + Σ_i ell_i z_i + Σ_{i<j} b_ij z_i z_j,
ell_i = f_i + H_ii,
b_ij = H_ij + H_ji.
```

This means that if both `H[i,j]` and `H[j,i]` are nonzero, the implemented pairwise coefficient is their sum. Upper-triangular input is therefore convenient when each interaction should be specified once.

## Supported modes

```python
modes = [
    "bruteforce",
    "miqp_cplex",
    "monolithic_qaoa",
    "abstract_distributed_qaoa",
    "telegate_explicit_qaoa",
]
```

See [`docs/MODES.md`](docs/MODES.md) for the interpretation and limitations of each mode.

## Configuration objects

The main configuration dataclasses are:

- `TrainConfig`
- `AdamConfig`
- `QuboObjectiveConfig`
- `MultiStartConfig`
- `AnalysisConfig`
- `ConvergenceConfig`

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for field-by-field descriptions.

## Full six-variable example

The cleaned version of the supplied manuscript example is available at:

```bash
python examples/example_01_full_workflow.py
```

It performs mode comparison, prints result summaries, creates elite-bitstring figures, saves Qiskit and paper-style circuit figures, plots depth progression, and creates a mode-comparison figure. Generated files are written under `outputs/example_01/` and are not tracked by Git.

The example requests `miqp_cplex`. If CPLEX is unavailable, that row is reported as an error while the other modes continue.

## Plotting sampled solutions

```python
from dqaoa_qubo import plot_top_k_elite_bitstrings

for result in results:
    if result.get("status") == "finished" and result.get("elite_rows_top_k"):
        plot_top_k_elite_bitstrings(
            result,
            top_k=10,
            title=f"Top-10 elite bitstrings for {result['mode']}",
            save_path=f"outputs/elite_topk_{result['mode']}.png",
        )
```

## Circuit figures

The repository provides two types of circuit figures:

1. **Qiskit circuit drawings**, created by `save_qubo_mode_circuit_figure`.
2. **Compact paper-style schematics**, created by `save_qubo_mode_paper_schematic`, `save_monolithic_qaoa_style_circuit_figure`, and `save_dqaoa_style_qaoa_circuit_figure`.

These figures are visualizations of the implementation and allocation. The compact paper-style schematics are not replacements for the full executable circuit object.

## Validation

To compare abstract and explicit distributed implementations at fixed angles:

```bash
python examples/example_02_fixed_angle_validation.py
```

The reported total variation distance is shot-dependent and should not be expected to be exactly zero at finite shots.

## Result dictionaries

Common fields include:

- `mode`, `status`
- `best_bitstring`, `best_cost`
- `mean_cost`
- `runtime_to_solution_seconds`
- `remote_cx_count`, `cross_qpu_terms`
- `angles`, `gammas`, `betas`
- `elite_rows_top_k`, `elite_mass_top_k`
- `final_p_best_observed`
- `final_p_exact_cost`, `final_p_exact_bitstring`
- `optimality_gap_to_exact`
- `details`

See [`docs/API.md`](docs/API.md) for an overview.

## Reproducibility guidance

For every reported run, record:

- `H`, `f`, and `c0`;
- requested modes;
- QPU count, capacities, and manual assignment if used;
- every configuration dataclass;
- `p_max`, training shots, final shots, and exact threshold;
- Python, NumPy, Qiskit, Qiskit Aer, and operating-system versions;
- simulator/transpiler seeds;
- whether restarts were parallel;
- the final result dictionary and generated figures.

Shot-based QAOA and stochastic optimization can vary between runs. Exact agreement on a small benchmark does not imply a general quantum advantage or guarantee the same behavior on larger or noisy hardware.

## Testing

The default test suite validates dependency-light classical functionality:

```bash
pip install -e ".[dev]"
pytest
```

The GitHub Actions workflow runs these tests on Python 3.10, 3.11, and 3.12. Quantum integration tests are intentionally not run on every commit because Qiskit Aer installation and explicit TeleGate simulations are substantially heavier.

## Current limitations

- The code is a simulator and research framework, not a cloud-hardware scheduler.
- Explicit TeleGate mode models remote gates inside a Qiskit Aer circuit; it does not model every physical network effect, latency source, calibration error, or entanglement-generation process.
- The abstract distributed mode uses direct simulator CX gates for cross-QPU operations while counting them as remote operations.
- Brute force scales exponentially.
- CPLEX use depends on a separate installation and license.
- Candidate selection can use exact references for small instances when enabled; disable this behavior for a strictly reference-free quantum selection study.
- Runtime results depend strongly on hardware, simulator method, software versions, shots, depth, restarts, and QUBO structure.
- This release contains the attached solver module and examples. A separate Streamlit GUI is not included unless its source files are added to the repository.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## Citation

GitHub displays a **Cite this repository** button because the repository includes `CITATION.cff`. Update that file after the associated article receives its final IEEE citation and DOI.

Suggested software citation before journal publication:

> A. Rajabi, M. Hasanzadeh, and A. Kargarian, “DQAOA-QUBO Simulator,” version 1.0.0, 2026, software.

## License

Released under the MIT License. Before publishing under an institutional organization, confirm the copyright and licensing language with LSU/RAISE LAB.

## Maintainers and links

- RAISE LAB website: https://sites.google.com/site/aminkargarian/home
- Intended GitHub organization: https://github.com/LSU-RAISE-LAB/

## Publishing to GitHub

A complete first-push walkthrough is in [`docs/PUBLISHING_TO_GITHUB.md`](docs/PUBLISHING_TO_GITHUB.md), and a final review list is in [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
