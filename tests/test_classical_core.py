import numpy as np

from dqaoa_qubo import (
    allocation_to_assignment,
    canonicalize_qubo_from_dense,
    make_allocation,
    qubo_value_from_bitstring,
    solve_qubo_bruteforce,
)


def test_canonicalization_and_evaluation():
    H = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    f = np.array([0.5, -1.0])
    instance = canonicalize_qubo_from_dense(H, f, c0=2.0, name="two_var")

    assert np.allclose(instance.linear, [1.5, 3.0])
    assert instance.quadratic == {(0, 1): 5.0}
    assert qubo_value_from_bitstring("00", instance) == 2.0
    assert qubo_value_from_bitstring("10", instance) == 3.5
    assert qubo_value_from_bitstring("01", instance) == 5.0
    assert qubo_value_from_bitstring("11", instance) == 11.5


def test_bruteforce_finds_known_optimum():
    H = np.diag([-1.0, 2.0])
    instance = canonicalize_qubo_from_dense(H, np.zeros(2), c0=0.0)
    result = solve_qubo_bruteforce(instance)

    assert result["status"] == "optimal"
    assert result["best_bitstring"] == "10"
    assert result["best_cost"] == -1.0


def test_contiguous_allocation():
    allocation = make_allocation(
        n=6,
        num_qpus=2,
        capacities=[3, 3],
        min_used_qpus=2,
    )
    assert allocation_to_assignment(6, allocation) == [0, 0, 0, 1, 1, 1]
