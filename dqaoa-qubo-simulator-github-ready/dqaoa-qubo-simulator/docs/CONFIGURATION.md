# Configuration reference

## `TrainConfig`

Controls shot-based objective evaluation and transpilation.

- `shots_train`: shots per training evaluation.
- `avg_k`: independent evaluations averaged per objective call.
- `seed_trans`: transpiler seed.
- `seed_base`: base simulator seed used during training.
- `optimization_level`: Qiskit transpiler optimization level.
- `sim_method`: Qiskit Aer simulation method, such as `automatic` or `matrix_product_state`.
- `batch_evaluations`: submit several parameter-bound circuits in one backend call when possible.

## `AdamConfig`

Controls the Adam update and SPSA gradient estimate.

- `iters`: optimizer iterations per restart.
- `lr`: base learning rate.
- `fd_eps0`: initial SPSA perturbation magnitude.
- `seed`: restart seed base.
- `spsa_alpha`, `spsa_gamma`, `spsa_A`: SPSA schedules.
- `eval_at_each_iter`: evaluate the unperturbed point each iteration for history reporting.

## `MultiStartConfig`

- `num_random_starts`: independent random initializations per depth.
- `warm_start_perturbations`: perturbed starts around the lifted previous-depth solution.
- `warm_start_sigma`: perturbation standard deviation.
- `add_plain_warm_start`: include the unperturbed lifted point.
- `keep_top_k_per_depth`: candidates retained in depth summaries.
- `select_by_exact_when_available`: use exact-reference agreement in candidate ranking when available.
- `parallel_restarts`: number of concurrent restart workers.

## `AnalysisConfig`

- `elite_top_k`: number of low-cost sampled bitstrings retained.
- `certificate_tol`: tolerance for lower-bound certificate comparisons.

## `ConvergenceConfig`

Defines tolerances used in the practical training-stability report.

## Reproducibility

Record all configuration objects, QUBO arrays, Qiskit versions, Python version, operating system, simulator method, and seeds. Shot-based and parallel execution can produce stochastic variation even when the same high-level settings are used.
