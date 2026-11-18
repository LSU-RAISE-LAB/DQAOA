# QUBO examples: 3–15 binary variables

This collection contains **39 new synthetic input instances**, three for each size from 3 through 15. These are additional examples, not a reconstruction of the historical experiments in the paper. Only cases explicitly listed in `tested_results/summary.txt` have been run with QAOA here.

| Case family | Interaction pattern | Maximum depth | Training shots | Final shots |
|---|---|---:|---:|---:|
| ring | Nearest neighbors plus closing edge | 1 | 128 | 512 |
| sparse | Ring plus random extra edges | 2 | 256 | 1024 |
| dense | All variable pairs | 3 | 512 | 2048 |

All cases use six SPSA–Adam iterations per restart per depth, learning rate 0.08, two random restarts, seed 7, one warm-start perturbation, perturbation standard deviation 0.10, and one restart worker. Warm starts add candidates at later depths. `p_max=3` means the backend proceeds through depths 1, 2, and 3. Its selected result can come from an earlier depth; the per-case result field `p` records that selected depth. `final_shots` is the backend sampling budget per evaluated candidate, not a single total budget for the entire optimization.

Each family has a different matrix and vector. These settings provide varied examples; differences between cases cannot isolate the effect of depth or shots because the inputs also change. For a controlled comparison, reuse one H/f pair and change one setting at a time. Dense large distributed cases may take substantially longer than small ring cases.

## Files and objective

Each case folder contains:

- `H.txt`: numeric, comma-separated n-by-n matrix, no header.
- `f.txt`: numeric, comma-separated vector of length n, no header.
- `settings.txt`: readable JSON text containing the constant, all case settings, seeds, and explicit QPU assignments.

The minimized objective is **C(x) = x^T H x + f^T x + c0**, with binary x and `c0=0.25`. There is no factor of one half in this convention. H is symmetric, so the coefficient of x_i*x_j for i<j is H_ij+H_ji. Bitstrings list x_0 first, then x_1, and so on, in matrix row order.

Generator: NumPy `default_rng`, seed `20261118 + 100*n + family_index`, where ring=0, sparse=1, dense=2. Linear-vector entries are uniform integers from -12 through 12 divided by 10; diagonal entries are integers from -4 through 4 divided by 10. Edge coefficients are nonzero integers from -12 through 12 divided by 10, split equally between symmetric matrix entries. Sparse non-ring edges are included with probability 0.20. `generate_inputs.py DESTINATION` regenerates the 78 matrix/vector files; byte-for-byte reproduction was checked. No optimum was planted, and instances were not filtered by QAOA success. For three variables, ring and dense have the same topology but different coefficients.

QPU assignments are round-robin: variable i belongs to QPU `i % num_qpus`. The files list all assignments and capacities explicitly; QPU IDs start at zero. Capacities count data qubits. The explicit telegate simulator adds its two reusable communication qubits. This is an ideal circuit simulation, not a network hardware timing model.

## Open a case in the GUI

1. Start the app using `gui/Open App.cmd` after installing its requirements.
2. Choose the CSV/TXT input-upload option.
3. Upload the case's `H.txt` and `f.txt` to their respective fields.
4. Set the constant to 0.25 and enter the settings from `settings.txt`, including depth, shots, seed, restarts, and QPU allocation.

The existing GUI does **not** automatically import `settings.txt`. The batch runner below applies the saved configuration through the existing GUI job runner and is the preferred way to reproduce the recorded runs. No GUI changes are required for matrix uploads.

## Run the saved configurations

Copy this `examples/qubo_txt` directory into the DQAOA repository alongside its existing `gui` directory. From the repository root on Windows, after installing GUI requirements:

```cmd
gui\.venv\Scripts\python.exe examples\qubo_txt\run_cases.py
```

The default subset is n03_ring, n03_sparse, n03_dense, n04_ring, n05_sparse, n06_ring, n08_ring, and n12_ring. Run another case with:

```cmd
gui\.venv\Scripts\python.exe examples\qubo_txt\run_cases.py --cases n10_dense --timeout 7200
```

Use `--all --timeout 7200` to request all 39 cases. The timeout applies to each case's complete subprocess, including both monolithic and explicit distributed modes. Results are written to `local_results`; the included `tested_results` are preserved. On Linux/macOS, use the Python interpreter with the GUI requirements installed in place of the Windows interpreter path. `--gui PATH` selects an alternate GUI folder.

The runner requires the repository GUI backend and job runner. Settings that are constants in that job runner (e.g. transpiler seed 1234, warm-start sigma 0.10, elite count 3, optimization level 1) match the files supplied here. Editing those fields in a settings file alone will not override job-runner constants. Other fields are passed through that runner.

Both `select_by_exact_when_available` and `compute_exact_if_small` are false. Only quantum modes are requested. Classical enumeration occurs after both quantum runs and is used solely to evaluate the returned cost. `classical_references.txt` contains separately enumerated references for all 39 inputs; the runner never reads that file.

## What the tests establish

`case_index.txt` lists all configurations and identifies the tested subset. `tested_results/summary.txt` reports best bitstrings, costs, cost gaps, and backend simulation times. Per-case files include settings and subprocess elapsed time; logs are retained. `environment.txt` records software versions, host-visible hardware, thread limits, and SHA-256 hashes of the backend and job runner.

These are single-seed functional checks, not a statistical benchmark, circuit-equivalence test, or evidence of hardware acceleration. Final backend samples participate in selecting candidates; an independent post-selection sample is not added by this runner. Mean cost remains the backend training objective, but mean-cost and probability metrics are not included in the result summaries.

Runtime figures are single illustrative measurements of classical simulation. The backend per-mode timer and end-to-end subprocess timer have different scopes; neither is a quantum hardware runtime. The subprocess timer includes imports and both quantum modes, and excludes the subsequent classical reference enumeration. Larger untested cases carry no claim of recovering a reference optimum.
