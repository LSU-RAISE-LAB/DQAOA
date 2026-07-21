# Public API overview

## Main entry points

- `solve_qubo_mode(...)`: run one solver mode.
- `compare_qubo_modes(...)`: run several modes on the same QUBO and return a list of result dictionaries.
- `print_mode_comparison_table(results)`: compact terminal comparison.
- `print_result_summary(result)`: detailed terminal summary.

## QUBO input convention

The input is

```text
F(z) = c0 + f^T z + z^T H z,    z in {0,1}^n.
```

The implementation canonicalizes it as

```text
F(z) = c0 + sum_i ell_i z_i + sum_{i<j} b_ij z_i z_j,
ell_i = f_i + H_ii,
b_ij = H_ij + H_ji.
```

Therefore, both triangular and full dense matrices are accepted, but users must avoid unintentionally duplicating a pairwise coefficient in both triangles unless that sum is intended.

## Important result fields

- `status`
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

## Visualization helpers

- `plot_top_k_elite_bitstrings`
- `plot_depth_progression`
- `plot_mode_comparison`
- `plot_training_trace`
- `save_qubo_mode_circuit_figure`
- `save_qubo_mode_paper_schematic`
- `save_monolithic_qaoa_style_circuit_figure`
- `save_dqaoa_style_qaoa_circuit_figure`

## Validation helper

`validate_fixed_angle_equivalence` compares decoded distributions produced by the abstract and explicit TeleGate implementations at fixed QAOA angles.
