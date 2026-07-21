"""Compare abstract and explicit TeleGate distributions at fixed angles."""

import numpy as np

from dqaoa_qubo import (
    print_validation_summary_fixed_angle,
    validate_fixed_angle_equivalence,
)

H = np.array(
    [
        [-0.5, 0.7, 0.0, 0.2],
        [0.0, -0.1, 0.8, 0.0],
        [0.0, 0.0, 0.3, -0.6],
        [0.0, 0.0, 0.0, -0.2],
    ],
    dtype=float,
)
f = np.array([0.1, -0.2, 0.0, 0.3], dtype=float)

result = validate_fixed_angle_equivalence(
    H=H,
    f=f,
    p=1,
    gammas=[0.3],
    betas=[0.7],
    num_qpus=2,
    capacities=[2, 2],
    shots=2048,
    seed_sim=123,
)

print_validation_summary_fixed_angle(result)
