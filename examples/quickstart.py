"""Small, runnable comparison without the optional CPLEX dependency."""

import numpy as np

from dqaoa_qubo import (
    AdamConfig,
    AnalysisConfig,
    MultiStartConfig,
    TrainConfig,
    compare_qubo_modes,
    print_mode_comparison_table,
)

H = np.array(
    [
        [-0.5, 0.8, 0.0],
        [0.0, -0.2, 0.6],
        [0.0, 0.0, 0.1],
    ],
    dtype=float,
)
f = np.array([0.1, -0.4, 0.2], dtype=float)

results = compare_qubo_modes(
    H=H,
    f=f,
    c0=0.0,
    name="quickstart_3var",
    modes=["bruteforce", "monolithic_qaoa", "telegate_explicit_qaoa"],
    p_max=1,
    num_qpus=2,
    capacities=[2, 1],
    train_cfg=TrainConfig(shots_train=128, batch_evaluations=True),
    final_shots=256,
    adam_cfg=AdamConfig(iters=2, seed=123),
    multistart_cfg=MultiStartConfig(num_random_starts=2, parallel_restarts=1),
    analysis_cfg=AnalysisConfig(elite_top_k=10),
)

print_mode_comparison_table(results)
