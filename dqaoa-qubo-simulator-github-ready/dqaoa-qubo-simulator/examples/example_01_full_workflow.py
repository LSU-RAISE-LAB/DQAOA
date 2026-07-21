"""Complete six-variable workflow corresponding to the manuscript example.

The CPLEX row is optional. If DOcplex or a local CPLEX runtime is unavailable,
`compare_qubo_modes` reports an error for that mode and continues with the
remaining modes.
"""

from pathlib import Path

import numpy as np

from dqaoa_qubo_multi_mode_callable_v5_batched_parallel import (
    AdamConfig,
    AnalysisConfig,
    ConvergenceConfig,
    MultiStartConfig,
    QuboObjectiveConfig,
    TrainConfig,
    compare_qubo_modes,
    plot_depth_progression,
    plot_mode_comparison,
    plot_top_k_elite_bitstrings,
    print_mode_comparison_table,
    print_result_summary,
    save_dqaoa_style_qaoa_circuit_figure,
    save_monolithic_qaoa_style_circuit_figure,
    save_qubo_mode_circuit_figure,
    save_qubo_mode_paper_schematic,
)

OUTPUT_DIR = Path("outputs/example_01")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H = np.array(
    [
        [-0.214529478, 0.620997848, -0.978818088, -1.40181904, 0.0798320096, 0.0],
        [0.0, 0.233485298, 0.841898687, 0.111014469, 0.0, 0.766889695],
        [0.0, 0.0, 0.337768348, 0.456686564, 0.0, 0.00124164979],
        [0.0, 0.0, 0.0, -0.230794135, 1.31540469, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.41730053, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.729683997],
    ],
    dtype=float,
)

f = np.array(
    [0.17668033, -0.83276013, -0.59186832, -0.85773984, -0.67432827, 0.51544642],
    dtype=float,
)
c0 = 0.5

results = compare_qubo_modes(
    H=H,
    f=f,
    c0=c0,
    name="user_nontrivial_6var",
    modes=[
        "bruteforce",
        "miqp_cplex",
        "monolithic_qaoa",
        "abstract_distributed_qaoa",
        "telegate_explicit_qaoa",
    ],
    p_max=2,
    num_qpus=2,
    capacities=[3, 4],
    train_cfg=TrainConfig(
        shots_train=256,
        avg_k=1,
        seed_trans=1234,
        seed_base=7,
        batch_evaluations=True,
    ),
    obj_cfg=QuboObjectiveConfig(objective_name="mean_cost", delta_close_cost=0.0),
    analysis_cfg=AnalysisConfig(elite_top_k=20, certificate_tol=1e-9),
    final_shots=512,
    adam_cfg=AdamConfig(iters=4, lr=0.08, seed=123),
    multistart_cfg=MultiStartConfig(
        num_random_starts=12,
        warm_start_perturbations=3,
        warm_start_sigma=0.10,
        add_plain_warm_start=True,
        keep_top_k_per_depth=1,
        select_by_exact_when_available=True,
        parallel_restarts=4,
    ),
    compute_exact_if_small=True,
    exact_threshold_n=20,
    convergence_cfg=ConvergenceConfig(),
    store_history=True,
)

print_mode_comparison_table(results)
for result in results:
    print_result_summary(result)

qaoa_modes = {
    "monolithic_qaoa",
    "abstract_distributed_qaoa",
    "telegate_explicit_qaoa",
}
for result in results:
    if (
        result.get("mode") in qaoa_modes
        and result.get("status") == "finished"
        and result.get("elite_rows_top_k") is not None
    ):
        plot_top_k_elite_bitstrings(
            result,
            top_k=10,
            save_path=str(OUTPUT_DIR / f"elite_topk_{result['mode']}.png"),
            title=f"Top-10 elite bitstrings for {result['mode']}",
        )

mono = save_qubo_mode_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="paper_example",
    mode="monolithic_qaoa",
    p=1,
    circuit_view="parametrized",
    measure=False,
    output_path=str(OUTPUT_DIR / "monolithic_qaoa_ansatz.png"),
)
print(mono["output_path"], mono["stats"])

dist = save_qubo_mode_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="paper_example",
    mode="telegate_explicit_qaoa",
    p=1,
    num_qpus=2,
    capacities=[3, 4],
    circuit_view="parametrized",
    measure=False,
    output_path=str(OUTPUT_DIR / "distributed_dqaoa_ansatz.png"),
)
print(dist["output_path"], dist["assignment"], dist["stats"])

save_qubo_mode_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="paper_example",
    mode="monolithic_qaoa",
    p=1,
    circuit_view="transpiled",
    measure=False,
    train_cfg=TrainConfig(seed_trans=1234, optimization_level=1),
    output_path=str(OUTPUT_DIR / "monolithic_qaoa_transpiled.png"),
)

save_monolithic_qaoa_style_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="example1",
    p=1,
    output_path=str(OUTPUT_DIR / "monolithic_qaoa_style.png"),
    show_numeric_coeffs=False,
    max_quadratic_terms_to_draw=5,
    include_linear_layer=True,
    include_mixer_layer=True,
)

save_qubo_mode_paper_schematic(
    H=H,
    f=f,
    c0=c0,
    name="example1",
    mode="monolithic_qaoa",
    p=1,
    output_path=str(OUTPUT_DIR / "monolithic_qaoa_schematic.png"),
)

save_qubo_mode_paper_schematic(
    H=H,
    f=f,
    c0=c0,
    name="example1",
    mode="telegate_explicit_qaoa",
    p=1,
    num_qpus=2,
    capacities=[3, 4],
    output_path=str(OUTPUT_DIR / "distributed_dqaoa_schematic.png"),
)

save_dqaoa_style_qaoa_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="example1",
    p=1,
    num_qpus=2,
    capacities=[3, 4],
    output_path=str(OUTPUT_DIR / "distributed_qaoa_style_shortened_paper.png"),
    show_numeric_coeffs=False,
    max_quadratic_terms_to_draw=4,
    include_linear_layer=True,
    include_mixer_layer=True,
)

save_dqaoa_style_qaoa_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="example1",
    p=1,
    num_qpus=2,
    capacities=[3, 4],
    output_path=str(OUTPUT_DIR / "distributed_qaoa_style_paper.png"),
    show_numeric_coeffs=False,
)

for result in results:
    if result.get("mode") in {"monolithic_qaoa", "telegate_explicit_qaoa"} and result.get("status") == "finished":
        plot_depth_progression(
            result,
            save_path=str(OUTPUT_DIR / f"depth_progression_{result['mode']}.png"),
        )

plot_mode_comparison(results, save_path=str(OUTPUT_DIR / "mode_comparison.png"))

save_dqaoa_style_qaoa_circuit_figure(
    H=H,
    f=f,
    c0=c0,
    name="your_case",
    p=1,
    num_qpus=2,
    capacities=[2, 4],
    output_path=str(OUTPUT_DIR / "distributed_dqaoa_final_fixed_qpu_boxes.png"),
    dpi=500,
    max_quadratic_terms_to_draw=3,
    include_linear_layer=True,
    include_mixer_layer=True,
    trim_right_after_mixer=True,
    align_mixer_to_last_measure=True,
    wire_label_fontsize=15,
    comm_label_fontsize=15,
    label_fontweight="bold",
    draw_qpu_group_boxes=True,
    qpu_group_box_lw=2.2,
    qpu_group_box_pad_y=0.525,
    qpu_group_box_left_x=0.04,
    qpu_group_box_right_pad=0.06,
)
