# Troubleshooting

## `ImportError: No module named qiskit`

Install the full environment:

```bash
pip install -r requirements.txt
```

Confirm the active interpreter:

```bash
python -c "import sys; print(sys.executable)"
python -c "import qiskit, qiskit_aer; print(qiskit.__version__, qiskit_aer.__version__)"
```

## CPLEX mode reports an error

`docplex` is the modeling layer; a compatible local CPLEX runtime is also required. Install `requirements-optional-cplex.txt`, then follow IBM's CPLEX installation and licensing instructions. Otherwise, remove `miqp_cplex` from the requested mode list.

## Circuit drawing asks for `pylatexenc`

Install it with:

```bash
pip install pylatexenc
```

## Distributed mode requires at least two QPUs

Use `num_qpus >= 2`, and ensure `sum(capacities) >= n`. Each capacity must be a nonnegative integer and the selected allocation must use at least two QPUs.

## Runs are slow

Start with smaller development settings:

- `p_max=1`
- `shots_train=128`
- `final_shots=256`
- `iters=2`
- `num_random_starts=2`
- `parallel_restarts=1`

Then increase one setting at a time. Explicit TeleGate mode is expected to be considerably slower than the monolithic and abstract modes.

## A result differs between runs

The optimizer and final distribution are stochastic. Check every seed and setting, increase shots, store history, and compare probability mass rather than only one bitstring.
