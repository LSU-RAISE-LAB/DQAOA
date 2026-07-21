from __future__ import annotations

import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Callable

import numpy as np

# =============================================================================
# imports
# =============================================================================
HAVE_QISKIT = True
QISKIT_IMPORT_ERROR = None
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit import ParameterVector
    from qiskit_aer import AerSimulator
except Exception as e:
    HAVE_QISKIT = False
    QISKIT_IMPORT_ERROR = e

HAVE_DOCPLEX = True
DOCPLEX_IMPORT_ERROR = None
try:
    from docplex.mp.model import Model
except Exception as e:
    HAVE_DOCPLEX = False
    DOCPLEX_IMPORT_ERROR = e

HAVE_MATPLOTLIB = True
MATPLOTLIB_IMPORT_ERROR = None
try:
    import matplotlib.pyplot as plt
except Exception as e:
    HAVE_MATPLOTLIB = False
    MATPLOTLIB_IMPORT_ERROR = e


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class QUBOInstance:
    name: str
    n: int
    c0: float
    linear: np.ndarray
    quadratic: Dict[Tuple[int, int], float]   # only i<j


@dataclass
class TrainConfig:
    shots_train: int = 256
    avg_k: int = 1
    seed_trans: int = 1234
    seed_base: int = 7
    optimization_level: int = 1
    sim_method: str = "automatic"
    batch_evaluations: bool = True


@dataclass
class AdamConfig:
    iters: int = 8
    lr: float = 0.08
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    fd_eps0: float = 0.12
    seed: int = 0
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    spsa_A: float = 0.0
    eval_at_each_iter: bool = True

@dataclass
class QuboObjectiveConfig:
    """
    Training objective is intentionally pure mean_cost.
    This config is kept only for compatibility and reporting settings.
    """
    objective_name: str = "mean_cost"
    delta_close_cost: float = 0.0


@dataclass
class MultiStartConfig:
    num_random_starts: int = 4
    warm_start_perturbations: int = 1
    warm_start_sigma: float = 0.10
    add_plain_warm_start: bool = True
    keep_top_k_per_depth: int = 1
    select_by_exact_when_available: bool = True
    parallel_restarts: int = 1


@dataclass
class AnalysisConfig:
    elite_top_k: int = 10
    certificate_tol: float = 1e-9

@dataclass
class ConvergenceConfig:
    stable_window: int = 3
    obj_tol: float = 1e-3
    mean_cost_tol: float = 1e-3
    p_best_tol: float = 1e-3
    p_close_tol: float = 1e-3
    gap_tol: float = 1e-3


# =============================================================================
# Generic utilities
# =============================================================================
def clip_angles(x: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(x, dtype=float), np.pi)


def bitstring_from_int(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def invert_endianness(bs: str) -> str:
    return bs.replace(" ", "")[::-1]


def require_qiskit():
    if not HAVE_QISKIT:
        raise ImportError(
            "This mode requires qiskit and qiskit-aer, but they could not be imported.\n"
            f"Original import error: {QISKIT_IMPORT_ERROR}"
        )


def require_docplex():
    if not HAVE_DOCPLEX:
        raise ImportError(
            "This mode requires docplex (and a local CPLEX installation), but docplex "
            f"could not be imported.\nOriginal import error: {DOCPLEX_IMPORT_ERROR}"
        )


def require_matplotlib():
    if not HAVE_MATPLOTLIB:
        raise ImportError(
            "This plotting function requires matplotlib, but it could not be imported.\n"
            f"Original import error: {MATPLOTLIB_IMPORT_ERROR}"
        )


def _to_py_scalar(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _to_py_float_list(arr) -> List[float]:
    return [float(v) for v in np.asarray(arr, dtype=float).ravel().tolist()]


def summarize_result(res: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "mode": res.get("mode"),
        "status": res.get("status"),
        "best_bitstring": res.get("best_bitstring"),
        "best_cost": _to_py_scalar(res.get("best_cost")),
        "incumbent_bitstring": res.get("incumbent_bitstring", res.get("best_bitstring")),
        "incumbent_cost": _to_py_scalar(res.get("incumbent_cost", res.get("best_cost"))),
        "mean_cost": _to_py_scalar(res.get("mean_cost")),
        "runtime_to_solution_seconds": _to_py_scalar(res.get("runtime_to_solution_seconds")),
        "remote_cx_count": _to_py_scalar(res.get("remote_cx_count")),
        "cross_qpu_terms": _to_py_scalar(res.get("cross_qpu_terms")),
        "final_objective_value": _to_py_scalar(res.get("final_objective_value")),
        "final_p_best_observed": _to_py_scalar(res.get("final_p_best_observed")),
        "final_p_close_cost": _to_py_scalar(res.get("final_p_close_cost")),
        "final_p_exact_bitstring": _to_py_scalar(res.get("final_p_exact_bitstring")),
        "final_p_exact_cost": _to_py_scalar(res.get("final_p_exact_cost")),
        "final_p_incumbent_bitstring": _to_py_scalar(res.get("final_p_incumbent_bitstring", res.get("final_p_best_observed"))),
        "final_p_incumbent_cost": _to_py_scalar(res.get("final_p_incumbent_cost", res.get("final_p_best_observed"))),
        "elite_top_k": _to_py_scalar(res.get("elite_top_k")),
        "elite_mass_top_k": _to_py_scalar(res.get("elite_mass_top_k")),
        "elite_mean_cost_top_k": _to_py_scalar(res.get("elite_mean_cost_top_k")),
        "num_unique_sampled": _to_py_scalar(res.get("num_unique_sampled")),
        "distribution_gap": _to_py_scalar(res.get("distribution_gap")),
        "practical_converged": res.get("practical_converged"),
        "stable_over_window": res.get("stable_over_window"),
        "exact_optimum_cost": _to_py_scalar(res.get("exact_optimum_cost")),
        "optimality_gap_to_exact": _to_py_scalar(res.get("optimality_gap_to_exact")),
        "cost_match_exact": res.get("cost_match_exact"),
        "bitstring_match_exact": res.get("bitstring_match_exact"),
        "certificate_lower_bound": _to_py_scalar(res.get("certificate_lower_bound")),
        "certificate_gap": _to_py_scalar(res.get("certificate_gap")),
        "certified_optimal": res.get("certified_optimal"),
        "restart_consensus_cost": _to_py_scalar(res.get("restart_consensus_cost")),
        "restart_consensus_bitstring": _to_py_scalar(res.get("restart_consensus_bitstring")),
        "mode_consensus_cost": _to_py_scalar(res.get("mode_consensus_cost")),
        "mode_consensus_bitstring": _to_py_scalar(res.get("mode_consensus_bitstring")),
    }
    if "p" in res:
        out["p"] = _to_py_scalar(res.get("p"))
    if "angles" in res:
        out["angles"] = [float(v) for v in res.get("angles", [])]
    if "gammas" in res:
        out["gammas"] = [float(v) for v in res.get("gammas", [])]
    if "betas" in res:
        out["betas"] = [float(v) for v in res.get("betas", [])]
    if "details" in res and isinstance(res["details"], dict):
        out["details"] = res["details"]
    return out


def print_result_summary(res: Dict[str, Any]):
    s = summarize_result(res)
    print("\n================ QUBO RESULT SUMMARY ================")
    print(f"mode                      = {s.get('mode')}")
    print(f"status                    = {s.get('status')}")
    if 'p' in s:
        print(f"p                         = {s.get('p')}")
    print(f"best_bitstring            = {s.get('best_bitstring')}")
    print(f"best_cost                 = {s.get('best_cost')}")
    print(f"incumbent_bitstring       = {s.get('incumbent_bitstring')}")
    print(f"incumbent_cost            = {s.get('incumbent_cost')}")
    print(f"mean_cost                 = {s.get('mean_cost')}")
    print(f"final_objective_value     = {s.get('final_objective_value')}")
    print(f"final_p_best_observed     = {s.get('final_p_best_observed')}")
    print(f"final_p_close_cost        = {s.get('final_p_close_cost')}")
    print(f"final_p_exact_bitstring   = {s.get('final_p_exact_bitstring')}")
    print(f"final_p_exact_cost        = {s.get('final_p_exact_cost')}")
    print(f"final_p_inc_bitstring     = {s.get('final_p_incumbent_bitstring')}")
    print(f"final_p_inc_cost          = {s.get('final_p_incumbent_cost')}")
    print(f"elite_top_k               = {s.get('elite_top_k')}")
    print(f"elite_mass_top_k          = {s.get('elite_mass_top_k')}")
    print(f"elite_mean_cost_top_k     = {s.get('elite_mean_cost_top_k')}")
    print(f"num_unique_sampled        = {s.get('num_unique_sampled')}")
    print(f"distribution_gap          = {s.get('distribution_gap')}")
    print(f"restart_consensus_cost    = {s.get('restart_consensus_cost')}")
    print(f"restart_consensus_bs      = {s.get('restart_consensus_bitstring')}")
    print(f"mode_consensus_cost       = {s.get('mode_consensus_cost')}")
    print(f"mode_consensus_bs         = {s.get('mode_consensus_bitstring')}")
    print(f"practical_converged       = {s.get('practical_converged')}")
    print(f"stable_over_window        = {s.get('stable_over_window')}")
    print(f"exact_optimum_cost        = {s.get('exact_optimum_cost')}")
    print(f"optimality_gap_to_exact   = {s.get('optimality_gap_to_exact')}")
    print(f"cost_match_exact          = {s.get('cost_match_exact')}")
    print(f"bitstring_match_exact     = {s.get('bitstring_match_exact')}")
    print(f"certificate_lower_bound   = {s.get('certificate_lower_bound')}")
    print(f"certificate_gap           = {s.get('certificate_gap')}")
    print(f"certified_optimal         = {s.get('certified_optimal')}")
    print(f"runtime_to_solution_s     = {s.get('runtime_to_solution_seconds')}")
    print(f"remote_cx_count           = {s.get('remote_cx_count')}")
    print(f"cross_qpu_terms           = {s.get('cross_qpu_terms')}")
    if "angles" in s:
        print(f"angles                    = {s.get('angles')}")
    if "gammas" in s:
        print(f"gammas                    = {s.get('gammas')}")
    if "betas" in s:
        print(f"betas                     = {s.get('betas')}")
    if "details" in s:
        print(f"details                   = {s.get('details')}")
    print("=====================================================\n")


# =============================================================================
# QUBO canonicalization / evaluation
# =============================================================================
def canonicalize_qubo_from_dense(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    tol: float = 1e-14,
) -> QUBOInstance:
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")
    n = H.shape[0]
    if f is None:
        f = np.zeros(n, dtype=float)
    else:
        f = np.asarray(f, dtype=float).reshape(-1)
        if len(f) != n:
            raise ValueError("len(f) must equal H.shape[0].")
    linear = f.copy()
    quadratic: Dict[Tuple[int, int], float] = {}
    linear += np.diag(H)
    for i in range(n):
        for j in range(i + 1, n):
            bij = H[i, j] + H[j, i]
            if abs(bij) > tol:
                quadratic[(i, j)] = float(bij)
    return QUBOInstance(
        name=name,
        n=n,
        c0=float(c0),
        linear=linear.astype(float),
        quadratic=quadratic,
    )


def qubo_value_from_bitstring(bs_qubit_order: str, instance: QUBOInstance) -> float:
    z = np.array([int(ch) for ch in bs_qubit_order], dtype=float)
    val = instance.c0
    val += float(np.dot(instance.linear, z))
    for (i, j), bij in instance.quadratic.items():
        val += bij * z[i] * z[j]
    return float(val)


# =============================================================================
# Exact classical references
# =============================================================================
def solve_qubo_bruteforce(instance: QUBOInstance) -> Dict[str, Any]:
    t0 = time.perf_counter()
    best_cost = float("inf")
    best_bs = None
    all_costs: Dict[str, float] = {}
    for x in range(2 ** instance.n):
        bs = bitstring_from_int(x, instance.n)
        val = qubo_value_from_bitstring(bs, instance)
        all_costs[bs] = float(val)
        if val < best_cost:
            best_cost = val
            best_bs = bs
    optimal_bitstrings = sorted([bs for bs, cost in all_costs.items() if abs(cost - best_cost) <= 1e-12])
    elapsed = time.perf_counter() - t0
    return {
        "mode": "bruteforce",
        "status": "optimal",
        "best_cost": float(best_cost),
        "best_bitstring": str(best_bs),
        "optimal_bitstrings": optimal_bitstrings,
        "optimal_degeneracy": len(optimal_bitstrings),
        "mean_cost": None,
        "runtime_to_solution_seconds": float(elapsed),
        "remote_cx_count": 0,
        "cross_qpu_terms": 0,
        "details": {},
    }


def solve_qubo_miqp_cplex(instance: QUBOInstance, time_limit: Optional[float] = None) -> Dict[str, Any]:
    require_docplex()
    t0 = time.perf_counter()
    mdl = Model(name=f"qubo_{instance.name}")
    z = mdl.binary_var_list(instance.n, name="z")
    obj = instance.c0
    for i in range(instance.n):
        obj += instance.linear[i] * z[i]
    for (i, j), bij in instance.quadratic.items():
        obj += bij * z[i] * z[j]
    mdl.minimize(obj)
    if time_limit is not None:
        mdl.parameters.timelimit = float(time_limit)
    sol = mdl.solve(log_output=False)
    elapsed = time.perf_counter() - t0
    if sol is None:
        return {
            "mode": "miqp_cplex",
            "status": "no_solution",
            "best_cost": None,
            "best_bitstring": None,
            "mean_cost": None,
            "runtime_to_solution_seconds": float(elapsed),
            "remote_cx_count": 0,
            "cross_qpu_terms": 0,
            "details": {"solver_status": str(mdl.solve_details.status)},
        }
    best_z = np.array([int(round(sol.get_value(v))) for v in z], dtype=int)
    best_bs = "".join(str(int(v)) for v in best_z)
    best_cost = qubo_value_from_bitstring(best_bs, instance)
    return {
        "mode": "miqp_cplex",
        "status": "optimal_or_best_found",
        "best_cost": float(best_cost),
        "best_bitstring": best_bs,
        "mean_cost": None,
        "runtime_to_solution_seconds": float(elapsed),
        "remote_cx_count": 0,
        "cross_qpu_terms": 0,
        "details": {
            "solver_status": str(mdl.solve_details.status),
            "objective_from_docplex": float(sol.objective_value),
        },
    }


# =============================================================================
# Allocation helpers
# =============================================================================
def make_balanced_capacities(n: int, num_qpus: int) -> List[int]:
    if num_qpus < 1:
        raise ValueError("num_qpus must be >= 1")
    if num_qpus > n:
        raise ValueError("num_qpus cannot exceed n")
    base = n // num_qpus
    rem = n % num_qpus
    return [base + (1 if i < rem else 0) for i in range(num_qpus)]


def validate_allocation(
    n: int,
    num_qpus: int,
    capacities: List[int],
    allocation: Dict[int, Tuple[int, int]],
    min_used_qpus: int = 1,
):
    if set(allocation.keys()) != set(range(n)):
        raise ValueError("Allocation must contain every node exactly once.")
    seen_slots = set()
    used_qpus = set()
    for node in range(n):
        q, local = allocation[node]
        if not (0 <= q < num_qpus):
            raise ValueError(f"Node {node} assigned to invalid QPU {q}.")
        if not (0 <= local < capacities[q]):
            raise ValueError(f"Invalid local slot for node {node} on QPU {q}.")
        if (q, local) in seen_slots:
            raise ValueError(f"Duplicate slot {(q, local)}.")
        seen_slots.add((q, local))
        used_qpus.add(q)
    if len(used_qpus) < min_used_qpus:
        raise ValueError(f"Need at least {min_used_qpus} used QPUs, got {len(used_qpus)}.")


def make_allocation(
    n: int,
    num_qpus: int,
    capacities: List[int],
    min_used_qpus: int = 1,
) -> Dict[int, Tuple[int, int]]:
    if len(capacities) != num_qpus:
        raise ValueError("capacities must have length num_qpus")
    if sum(capacities) < n:
        raise ValueError("sum(capacities) must be >= n")
    alloc: Dict[int, Tuple[int, int]] = {}
    q = 0
    local = 0
    for i in range(n):
        while local >= capacities[q]:
            q += 1
            local = 0
        alloc[i] = (q, local)
        local += 1
    validate_allocation(n, num_qpus, capacities, alloc, min_used_qpus=min_used_qpus)
    return alloc


def make_allocation_from_assignment(
    n: int,
    num_qpus: int,
    capacities: List[int],
    qpu_assignment: List[int],
    min_used_qpus: int = 1,
) -> Dict[int, Tuple[int, int]]:
    if len(qpu_assignment) != n:
        raise ValueError("len(qpu_assignment) must equal n")
    next_slot = [0] * num_qpus
    alloc: Dict[int, Tuple[int, int]] = {}
    for node, q in enumerate(qpu_assignment):
        if not (0 <= q < num_qpus):
            raise ValueError(f"Node {node} assigned to invalid QPU {q}.")
        if next_slot[q] >= capacities[q]:
            raise ValueError(f"QPU {q} capacity exceeded.")
        alloc[node] = (q, next_slot[q])
        next_slot[q] += 1
    validate_allocation(n, num_qpus, capacities, alloc, min_used_qpus=min_used_qpus)
    return alloc


def allocation_to_assignment(n: int, allocation: Dict[int, Tuple[int, int]]) -> List[int]:
    return [allocation[i][0] for i in range(n)]


def count_cross_qpu_quadratic_terms(
    quadratic: Dict[Tuple[int, int], float],
    allocation: Dict[int, Tuple[int, int]],
) -> int:
    return sum(1 for (i, j) in quadratic.keys() if allocation[i][0] != allocation[j][0])


def build_weighted_qubo_adjacency(
    n: int,
    quadratic: Dict[Tuple[int, int], float]
) -> List[Dict[int, float]]:
    adj = [dict() for _ in range(n)]
    for (i, j), bij in quadratic.items():
        w = abs(float(bij))
        adj[i][j] = adj[i].get(j, 0.0) + w
        adj[j][i] = adj[j].get(i, 0.0) + w
    return adj


def graph_aware_greedy_qpu_assignment_qubo(
    instance: QUBOInstance,
    num_qpus: int,
    capacities: List[int],
    min_used_qpus: int = 2,
) -> List[int]:
    n = instance.n
    adj = build_weighted_qubo_adjacency(n, instance.quadratic)
    weighted_degree = [sum(adj[i].values()) for i in range(n)]
    node_order = sorted(range(n), key=lambda i: (weighted_degree[i], i), reverse=True)
    assignment: List[Optional[int]] = [None] * n
    remaining = capacities.copy()
    counts = [0] * num_qpus
    seed_qpus = list(range(min_used_qpus))
    unused_nodes = node_order.copy()
    for q in seed_qpus:
        if remaining[q] <= 0:
            raise ValueError(f"QPU {q} has zero capacity and cannot be seeded.")
        node = unused_nodes.pop(0)
        assignment[node] = q
        remaining[q] -= 1
        counts[q] += 1
    for node in node_order:
        if assignment[node] is not None:
            continue
        candidate_scores = []
        for q in range(num_qpus):
            if remaining[q] <= 0:
                continue
            same_qpu_weight = sum(w for nbr, w in adj[node].items() if assignment[nbr] == q)
            qpu_is_active = 1 if counts[q] > 0 else 0
            rem_cap = remaining[q]
            candidate_scores.append((same_qpu_weight, qpu_is_active, rem_cap, -q))
        if not candidate_scores:
            raise ValueError(f"No feasible QPU available for node {node}.")
        best_score = max(candidate_scores)
        chosen_q = -best_score[3]
        assignment[node] = chosen_q
        remaining[chosen_q] -= 1
        counts[chosen_q] += 1
    final_assignment = [int(q) for q in assignment]
    _ = make_allocation_from_assignment(
        n=n,
        num_qpus=num_qpus,
        capacities=capacities,
        qpu_assignment=final_assignment,
        min_used_qpus=min_used_qpus,
    )
    return final_assignment


def build_allocation_candidates_qubo(
    instance: QUBOInstance,
    num_qpus: int,
    capacities: List[int],
    p_eval: int,
    manual_assignment: Optional[List[int]] = None,
    min_used_qpus: int = 2,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    n = instance.n
    alloc_contig = make_allocation(
        n=n,
        num_qpus=num_qpus,
        capacities=capacities,
        min_used_qpus=min_used_qpus,
    )
    cross_contig = count_cross_qpu_quadratic_terms(instance.quadratic, alloc_contig)
    candidates.append({
        "mode": "contiguous",
        "allocation": alloc_contig,
        "assignment": allocation_to_assignment(n, alloc_contig),
        "cross_qpu_terms": cross_contig,
        "remote_cx_at_p": 2 * cross_contig * p_eval,
    })
    ga_assignment = graph_aware_greedy_qpu_assignment_qubo(
        instance=instance,
        num_qpus=num_qpus,
        capacities=capacities,
        min_used_qpus=min_used_qpus,
    )
    alloc_ga = make_allocation_from_assignment(
        n=n,
        num_qpus=num_qpus,
        capacities=capacities,
        qpu_assignment=ga_assignment,
        min_used_qpus=min_used_qpus,
    )
    cross_ga = count_cross_qpu_quadratic_terms(instance.quadratic, alloc_ga)
    candidates.append({
        "mode": "graph_aware",
        "allocation": alloc_ga,
        "assignment": ga_assignment,
        "cross_qpu_terms": cross_ga,
        "remote_cx_at_p": 2 * cross_ga * p_eval,
    })
    if manual_assignment is not None:
        alloc_manual = make_allocation_from_assignment(
            n=n,
            num_qpus=num_qpus,
            capacities=capacities,
            qpu_assignment=manual_assignment,
            min_used_qpus=min_used_qpus,
        )
        cross_manual = count_cross_qpu_quadratic_terms(instance.quadratic, alloc_manual)
        candidates.append({
            "mode": "manual",
            "allocation": alloc_manual,
            "assignment": manual_assignment,
            "cross_qpu_terms": cross_manual,
            "remote_cx_at_p": 2 * cross_manual * p_eval,
        })
    return candidates


def select_best_allocation_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    mode_priority = {"manual": 0, "graph_aware": 1, "contiguous": 2}
    return sorted(
        candidates,
        key=lambda c: (
            c["cross_qpu_terms"],
            c["remote_cx_at_p"],
            mode_priority.get(c["mode"], 99),
        )
    )[0]


# =============================================================================
# Qiskit execution helpers
# =============================================================================
def run_counts_shot_based(
    qc: "QuantumCircuit",
    shots: int,
    seed_sim: int = 0,
    seed_trans: int = 0,
    optimization_level: int = 1,
    sim_method: str = "automatic",
) -> Dict[str, int]:
    require_qiskit()
    backend = AerSimulator(method=sim_method, seed_simulator=seed_sim)
    tqc = transpile(
        qc,
        backend=backend,
        seed_transpiler=seed_trans,
        optimization_level=optimization_level,
    )
    job = backend.run(tqc, shots=shots)
    res = job.result()
    counts = res.get_counts()
    return {str(k): int(v) for k, v in counts.items()}


# =============================================================================
# Remote-CX primitives
# =============================================================================
def apply_remote_cx_telegate_abstract(
    qc: "QuantumCircuit",
    ctrl,
    tgt,
    stats: Dict[str, Any]
):
    stats["remote_cx"] = stats.get("remote_cx", 0) + 1
    qc.cx(ctrl, tgt)


def apply_remote_cx_telegate_explicit(
    qc: "QuantumCircuit",
    ctrl,
    tgt,
    comm_a,
    comm_b,
    creg: "ClassicalRegister",
    ff_ptr: int,
    stats: Dict[str, Any],
) -> int:
    stats["remote_cx"] = stats.get("remote_cx", 0) + 1
    c_a = creg[ff_ptr]
    c_b = creg[ff_ptr + 1]
    qc.reset(comm_a)
    qc.reset(comm_b)
    qc.h(comm_a)
    qc.cx(comm_a, comm_b)
    qc.cx(ctrl, comm_a)
    qc.measure(comm_a, c_a)
    qc.cx(comm_b, tgt)
    qc.h(comm_b)
    qc.measure(comm_b, c_b)
    with qc.if_test((c_a, 1)):
        qc.x(tgt)
    with qc.if_test((c_b, 1)):
        qc.z(ctrl)
    return ff_ptr + 2


# =============================================================================
# QAOA circuit builders
# =============================================================================
def build_qubo_qaoa_circuit(
    instance: QUBOInstance,
    p: int,
    gammas: List[float],
    betas: List[float],
    allocation: Dict[int, Tuple[int, int]],
    measure: bool = True,
    remote_mode: str = "abstract",
) -> Tuple["QuantumCircuit", Dict[str, Any]]:
    require_qiskit()
    n = instance.n
    if len(gammas) != p or len(betas) != p:
        raise ValueError("gammas and betas must have length p")
    if remote_mode not in {"abstract", "telegate_explicit"}:
        raise ValueError("remote_mode must be 'abstract' or 'telegate_explicit'")
    cross_terms = count_cross_qpu_quadratic_terms(instance.quadratic, allocation)
    remote_uses_total = 2 * cross_terms * p
    if remote_mode == "telegate_explicit":
        qr = QuantumRegister(n, "q")
        qcomm = QuantumRegister(2, "comm")
        cr = ClassicalRegister(n + 2 * remote_uses_total, "c")
        qc = QuantumCircuit(qr, qcomm, cr)
    else:
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c") if measure else None
        qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)
        qcomm = None
    stats = {
        "remote_cx": 0,
        "cross_qpu_terms": cross_terms,
        "total_quadratic_terms": len(instance.quadratic),
        "remote_mode": remote_mode,
    }
    qc.h(qr)
    ff_ptr = n
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        for i, ai in enumerate(instance.linear):
            if abs(ai) > 1e-14:
                qc.rz(-gamma * ai, qr[i])
        for (i, j), bij in instance.quadratic.items():
            if abs(bij) <= 1e-14:
                continue
            qc.rz(-gamma * bij / 2.0, qr[i])
            qc.rz(-gamma * bij / 2.0, qr[j])
            same_qpu = allocation[i][0] == allocation[j][0]
            if same_qpu:
                qc.cx(qr[i], qr[j])
                qc.rz(gamma * bij / 2.0, qr[j])
                qc.cx(qr[i], qr[j])
            else:
                if remote_mode == "telegate_explicit":
                    ff_ptr = apply_remote_cx_telegate_explicit(
                        qc=qc,
                        ctrl=qr[i],
                        tgt=qr[j],
                        comm_a=qcomm[0],
                        comm_b=qcomm[1],
                        creg=cr,
                        ff_ptr=ff_ptr,
                        stats=stats,
                    )
                    qc.rz(gamma * bij / 2.0, qr[j])
                    ff_ptr = apply_remote_cx_telegate_explicit(
                        qc=qc,
                        ctrl=qr[i],
                        tgt=qr[j],
                        comm_a=qcomm[0],
                        comm_b=qcomm[1],
                        creg=cr,
                        ff_ptr=ff_ptr,
                        stats=stats,
                    )
                else:
                    apply_remote_cx_telegate_abstract(qc, qr[i], qr[j], stats)
                    qc.rz(gamma * bij / 2.0, qr[j])
                    apply_remote_cx_telegate_abstract(qc, qr[i], qr[j], stats)
        for i in range(n):
            qc.rx(2.0 * beta, qr[i])
    if measure:
        for i in range(n):
            qc.measure(qr[i], cr[i])
    return qc, stats


def build_parametrized_qubo_qaoa_circuit(
    instance: QUBOInstance,
    p: int,
    allocation: Dict[int, Tuple[int, int]],
    measure: bool = True,
    remote_mode: str = "abstract",
) -> Tuple["QuantumCircuit", "ParameterVector", "ParameterVector", Dict[str, Any]]:
    require_qiskit()
    n = instance.n
    if remote_mode not in {"abstract", "telegate_explicit"}:
        raise ValueError("remote_mode must be 'abstract' or 'telegate_explicit'")

    gamma_params = ParameterVector("gamma", p)
    beta_params = ParameterVector("beta", p)
    cross_terms = count_cross_qpu_quadratic_terms(instance.quadratic, allocation)
    remote_uses_total = 2 * cross_terms * p

    if remote_mode == "telegate_explicit":
        qr = QuantumRegister(n, "q")
        qcomm = QuantumRegister(2, "comm")
        cr = ClassicalRegister(n + 2 * remote_uses_total, "c")
        qc = QuantumCircuit(qr, qcomm, cr)
    else:
        qr = QuantumRegister(n, "q")
        cr = ClassicalRegister(n, "c") if measure else None
        qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)
        qcomm = None

    stats = {
        "remote_cx": 0,
        "cross_qpu_terms": cross_terms,
        "total_quadratic_terms": len(instance.quadratic),
        "remote_mode": remote_mode,
    }

    qc.h(qr)
    ff_ptr = n
    for layer in range(p):
        gamma = gamma_params[layer]
        beta = beta_params[layer]
        for i, ai in enumerate(instance.linear):
            if abs(ai) > 1e-14:
                qc.rz(-gamma * ai, qr[i])
        for (i, j), bij in instance.quadratic.items():
            if abs(bij) <= 1e-14:
                continue
            qc.rz(-gamma * bij / 2.0, qr[i])
            qc.rz(-gamma * bij / 2.0, qr[j])
            same_qpu = allocation[i][0] == allocation[j][0]
            if same_qpu:
                qc.cx(qr[i], qr[j])
                qc.rz(gamma * bij / 2.0, qr[j])
                qc.cx(qr[i], qr[j])
            else:
                if remote_mode == "telegate_explicit":
                    ff_ptr = apply_remote_cx_telegate_explicit(
                        qc=qc,
                        ctrl=qr[i],
                        tgt=qr[j],
                        comm_a=qcomm[0],
                        comm_b=qcomm[1],
                        creg=cr,
                        ff_ptr=ff_ptr,
                        stats=stats,
                    )
                    qc.rz(gamma * bij / 2.0, qr[j])
                    ff_ptr = apply_remote_cx_telegate_explicit(
                        qc=qc,
                        ctrl=qr[i],
                        tgt=qr[j],
                        comm_a=qcomm[0],
                        comm_b=qcomm[1],
                        creg=cr,
                        ff_ptr=ff_ptr,
                        stats=stats,
                    )
                else:
                    apply_remote_cx_telegate_abstract(qc, qr[i], qr[j], stats)
                    qc.rz(gamma * bij / 2.0, qr[j])
                    apply_remote_cx_telegate_abstract(qc, qr[i], qr[j], stats)
        for i in range(n):
            qc.rx(2.0 * beta, qr[i])

    if measure:
        for i in range(n):
            qc.measure(qr[i], cr[i])

    return qc, gamma_params, beta_params, stats


# =============================================================================
# Parametrized runner cache (FIRST FIX ONLY)
# =============================================================================
@dataclass
class CachedQuboCircuitRunner:
    instance: QUBOInstance
    p: int
    allocation: Dict[int, Tuple[int, int]]
    remote_mode: str
    optimization_level: int
    sim_method: str
    seed_trans: int
    measure: bool = True

    def __post_init__(self):
        require_qiskit()
        self.backend = AerSimulator(method=self.sim_method)
        (
            self.param_qc,
            self.gamma_params,
            self.beta_params,
            self.static_stats,
        ) = build_parametrized_qubo_qaoa_circuit(
            instance=self.instance,
            p=self.p,
            allocation=self.allocation,
            measure=self.measure,
            remote_mode=self.remote_mode,
        )
        self.transpiled_qc = transpile(
            self.param_qc,
            backend=self.backend,
            seed_transpiler=self.seed_trans,
            optimization_level=self.optimization_level,
        )

    def bind_parameters(self, gammas: List[float], betas: List[float]) -> "QuantumCircuit":
        if len(gammas) != self.p or len(betas) != self.p:
            raise ValueError("gammas and betas must have length p")
        bind_map = {self.gamma_params[i]: float(gammas[i]) for i in range(self.p)}
        bind_map.update({self.beta_params[i]: float(betas[i]) for i in range(self.p)})
        return self.transpiled_qc.assign_parameters(bind_map, inplace=False)

    def run_counts(self, gammas: List[float], betas: List[float], shots: int, seed_sim: int = 0) -> Dict[str, int]:
        bound_qc = self.bind_parameters(gammas=gammas, betas=betas)
        job = self.backend.run(bound_qc, shots=shots, seed_simulator=seed_sim)
        res = job.result()
        counts = res.get_counts()
        return {str(k): int(v) for k, v in counts.items()}

    def run_counts_batch(
        self,
        gammas_list: List[List[float]],
        betas_list: List[List[float]],
        shots: int,
        seed_sim: int = 0,
    ) -> List[Dict[str, int]]:
        if len(gammas_list) != len(betas_list):
            raise ValueError("gammas_list and betas_list must have the same length")
        if len(gammas_list) == 0:
            return []
        if len(gammas_list) == 1:
            return [self.run_counts(gammas_list[0], betas_list[0], shots=shots, seed_sim=seed_sim)]
        bound_qcs = [
            self.bind_parameters(gammas=gammas, betas=betas)
            for gammas, betas in zip(gammas_list, betas_list)
        ]
        job = self.backend.run(bound_qcs, shots=shots, seed_simulator=seed_sim)
        res = job.result()
        all_counts = res.get_counts()
        if isinstance(all_counts, list):
            return [{str(k): int(v) for k, v in counts.items()} for counts in all_counts]
        return [{str(k): int(v) for k, v in all_counts.items()}]


_RUNNER_CACHE: Dict[Tuple[Any, ...], CachedQuboCircuitRunner] = {}


def clear_circuit_runner_cache():
    _RUNNER_CACHE.clear()


def _instance_signature(instance: QUBOInstance, ndigits: int = 14) -> Tuple[Any, ...]:
    linear_sig = tuple(float(np.round(v, ndigits)) for v in np.asarray(instance.linear, dtype=float).tolist())
    quad_sig = tuple(
        (int(i), int(j), float(np.round(bij, ndigits)))
        for (i, j), bij in sorted(instance.quadratic.items())
    )
    return (
        int(instance.n),
        float(np.round(instance.c0, ndigits)),
        linear_sig,
        quad_sig,
    )


def get_cached_qubo_runner(
    instance: QUBOInstance,
    p: int,
    allocation: Dict[int, Tuple[int, int]],
    remote_mode: str,
    train_cfg: TrainConfig,
    measure: bool = True,
) -> CachedQuboCircuitRunner:
    key = (
        _instance_signature(instance),
        int(p),
        tuple(allocation_to_assignment(instance.n, allocation)),
        remote_mode,
        bool(measure),
        int(train_cfg.optimization_level),
        str(train_cfg.sim_method),
        int(train_cfg.seed_trans),
        int(threading.get_ident()),
    )
    runner = _RUNNER_CACHE.get(key)
    if runner is None:
        runner = CachedQuboCircuitRunner(
            instance=instance,
            p=p,
            allocation=allocation,
            remote_mode=remote_mode,
            optimization_level=train_cfg.optimization_level,
            sim_method=train_cfg.sim_method,
            seed_trans=train_cfg.seed_trans,
            measure=measure,
        )
        _RUNNER_CACHE[key] = runner
    return runner






# =============================================================================
# Paper-style distributed QAOA circuit figure (DVQE-like visual style)
# =============================================================================
def _resolve_plot_setup_for_dqaoa(
    instance: QUBOInstance,
    p_eval: int,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Resolve the allocation exactly the way the distributed solver does for
    telegate_explicit_qaoa, but without running optimization.
    """
    n = instance.n
    if num_qpus < 2:
        raise ValueError("Distributed DQAOA figure requires num_qpus >= 2.")
    if capacities is None:
        capacities = make_balanced_capacities(n, num_qpus)

    candidates = build_allocation_candidates_qubo(
        instance=instance,
        num_qpus=num_qpus,
        capacities=capacities,
        p_eval=p_eval,
        manual_assignment=manual_assignment,
        min_used_qpus=2,
    )
    best_candidate = select_best_allocation_candidate(candidates)
    allocation = best_candidate["allocation"]

    return {
        "allocation": allocation,
        "assignment": allocation_to_assignment(instance.n, allocation),
        "selected_allocation_mode": best_candidate["mode"],
        "num_qpus": int(num_qpus),
        "capacities": [int(v) for v in capacities],
        "remote_mode": "telegate_explicit",
    }


def _draw_box(ax, x, y, w, h, text, fc="#c9a3ff", ec="#333333", fontsize=7, lw=1.0, z=5):
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x - w/2, y - h/2), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=z
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=z+1)


def _draw_measure(ax, x, y, size=0.34, z=5):
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x - size/2, y - size/2), size, size,
        facecolor="black", edgecolor="black", linewidth=1.0,
        zorder=z
    )
    ax.add_patch(rect)
    ax.text(x, y, "↗", ha="center", va="center", fontsize=9, color="white", zorder=z+1)


def _draw_reset(ax, x, y, size=0.34, z=5):
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (x - size/2, y - size/2), size, size,
        facecolor="black", edgecolor="black", linewidth=1.0,
        zorder=z
    )
    ax.add_patch(rect)
    ax.text(x, y, r"$|0\rangle$", ha="center", va="center", fontsize=7, color="white", zorder=z+1)


def _draw_cnot(ax, x, y_ctrl, y_tgt, color="#78aefc", lw=1.3, z=2):
    from matplotlib.patches import Circle
    ax.plot([x, x], [y_ctrl, y_tgt], color=color, linewidth=lw, zorder=z)
    ax.add_patch(Circle((x, y_ctrl), 0.050, facecolor=color, edgecolor=color, linewidth=lw, zorder=z+1))
    ax.add_patch(Circle((x, y_tgt), 0.105, facecolor="white", edgecolor=color, linewidth=lw, zorder=z+1))
    ax.plot([x - 0.07, x + 0.07], [y_tgt, y_tgt], color=color, linewidth=lw, zorder=z+1)
    ax.plot([x, x], [y_tgt - 0.07, y_tgt + 0.07], color=color, linewidth=lw, zorder=z+1)


def _draw_classical_arrow(ax, x0, y0, x1, y1, color="#7a7a7a", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
        linestyle="solid",
        color=color,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(patch)


def _fmt_num(v: float, ndigits: int = 3) -> str:
    return f"{float(v):.{ndigits}g}"


def _draw_vertical_edge_connector(
    ax,
    x,
    y_src,
    h_src,
    y_dst,
    h_dst=0.0,
    color="#7a7a7a",
    lw=1.2,
    z=1,
):
    """
    Vertical connector from source-box edge center to destination-box edge center.
    If h_dst=0, the destination is treated like a line/bus.
    """
    if y_src > y_dst:
        # source above destination
        y0 = y_src - h_src / 2.0
        y1 = y_dst + h_dst / 2.0
    else:
        # source below destination
        y0 = y_src + h_src / 2.0
        y1 = y_dst - h_dst / 2.0

    ax.plot([x, x], [y0, y1], color=color, linewidth=lw, zorder=z)


def _draw_horizontal_edge_connector(
    ax,
    x_src,
    w_src,
    x_dst,
    w_dst,
    y,
    color="#7a7a7a",
    lw=1.2,
    z=1,
):
    """
    Horizontal connector from source-box edge center to destination-box edge center.
    """
    if x_src < x_dst:
        x0 = x_src + w_src / 2.0
        x1 = x_dst - w_dst / 2.0
    else:
        x0 = x_src - w_src / 2.0
        x1 = x_dst + w_dst / 2.0

    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=z)



def _draw_bus_dot(ax, x, y, r=0.05, color="black", z=4):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color, zorder=z))



def _draw_box_dqaoa(
    ax,
    x,
    y,
    w,
    h,
    text,
    fc="#c9a3ff",
    ec="#333333",
    fontsize=18,
    lw=1.2,
    z=5,
    fontweight="bold",
):
    """Dedicated box helper for the DQAOA paper-style renderer."""
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x - w / 2, y - h / 2),
        w,
        h,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        zorder=z + 1,
    )


def _draw_measure_dqaoa(ax, x, y, size=0.42, z=5):
    """White measurement box with clearer text."""
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x - size / 2, y - size / 2),
        size,
        size,
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        zorder=z,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        "M",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="black",
        zorder=z + 1,
    )


def _draw_reset_dqaoa(ax, x, y, size=0.46, z=5):
    """Black reset box with clearer |0> text."""
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x - size / 2, y - size / 2),
        size,
        size,
        facecolor="black",
        edgecolor="black",
        linewidth=1.0,
        zorder=z,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        r"$|0\rangle$",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="white",
        zorder=z + 1,
    )


def _draw_cnot_dqaoa(ax, x, y_ctrl, y_tgt, color="#78aefc", lw=1.3, z=2):
    """Dedicated CNOT helper for the DQAOA paper-style renderer."""
    from matplotlib.patches import Circle

    ax.plot([x, x], [y_ctrl, y_tgt], color=color, linewidth=lw, zorder=z)
    ax.add_patch(
        Circle(
            (x, y_ctrl),
            0.058,
            facecolor=color,
            edgecolor=color,
            linewidth=lw,
            zorder=z + 1,
        )
    )
    ax.add_patch(
        Circle(
            (x, y_tgt),
            0.125,
            facecolor="white",
            edgecolor=color,
            linewidth=lw,
            zorder=z + 1,
        )
    )
    ax.plot([x - 0.085, x + 0.085], [y_tgt, y_tgt], color=color, linewidth=lw, zorder=z + 1)
    ax.plot([x, x], [y_tgt - 0.085, y_tgt + 0.085], color=color, linewidth=lw, zorder=z + 1)


def _draw_bus_dot_dqaoa(ax, x, y, r=0.05, color="black", z=4):
    """Dedicated bus-dot helper to avoid conflicts with later helper definitions."""
    from matplotlib.patches import Circle

    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=color, zorder=z))


def _draw_qpu_group_box_dqaoa(
    ax,
    x_left,
    x_right,
    y_top,
    y_bottom,
    color,
    fill_color=None,
    lw=2.2,
    dash=(8, 5),
    z=20,
    fill_alpha=0.45,
):
    """
    Draw a translucent QPU-group background plus dashed outline.
    The background stays behind the circuit, while the dashed outline stays on top.
    """
    from matplotlib.patches import Rectangle

    if fill_color is None:
        fill_color = color

    # Background fill
    bg = Rectangle(
        (x_left, y_bottom),
        x_right - x_left,
        y_top - y_bottom,
        fill=True,
        facecolor=fill_color,
        edgecolor="none",
        alpha=fill_alpha,
        zorder=-1,
        clip_on=False,
    )
    ax.add_patch(bg)

    # Dashed outline
    rect = Rectangle(
        (x_left, y_bottom),
        x_right - x_left,
        y_top - y_bottom,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        linestyle=(0, dash),
        zorder=z,
        clip_on=False,
    )
    ax.add_patch(rect)

def _draw_telegate_remote_cx_clean(
    ax,
    x,
    yi,
    yj,
    comm_a_y,
    comm_b_y,
    cat_y,
):
    """
    Clean paper-style TeleGate remote-CX drawing.

    The spacing is intentionally larger than the earlier version:
      - the X correction is separated from the CNOT target circle,
      - the H, M, and |0> boxes on communication wires do not touch,
      - the vertical classical lines start/end at box edges.
    """

    # Main positions. These are the spacing controls for the yellow-marked areas.
    x_h_a     = x - 0.38
    x_ent     = x
    x_ctrl_a  = x + 0.62

    x_m_a     = x + 1.22
    x_rst_a   = x + 1.82

    x_xcorr   = x + 1.48
    x_cnot_bt = x + 2.18
    x_zcorr   = x + 2.58

    x_h_b     = x + 2.88
    x_m_b     = x + 3.50
    x_rst_b   = x + 4.12

    # Bell-pair preparation
    _draw_box_dqaoa(ax, x_h_a, comm_a_y, 0.52, 0.48, "H", fc="#98c7ff", fontsize=18)
    _draw_cnot_dqaoa(ax, x_ent, comm_a_y, comm_b_y)

    # Control-to-comm_a CNOT
    _draw_cnot_dqaoa(ax, x_ctrl_a, yi, comm_a_y)

    # Measure/reset comm_a
    _draw_measure_dqaoa(ax, x_m_a, comm_a_y, size=0.42)
    _draw_reset_dqaoa(ax, x_rst_a, comm_a_y, size=0.46)

    # Conditional X and remote target CNOT.
    # The center spacing x_cnot_bt - x_xcorr is deliberately large.
    _draw_box_dqaoa(ax, x_xcorr, yj, 0.52, 0.48, "X", fc="#56e0d0", fontsize=18)
    _draw_cnot_dqaoa(ax, x_cnot_bt, comm_b_y, yj)

    # Conditional Z
    _draw_box_dqaoa(ax, x_zcorr, yi, 0.52, 0.48, "Z", fc="#56e0d0", fontsize=18)

    # H, M, |0> on comm_b with clearer spacing.
    _draw_box_dqaoa(ax, x_h_b, comm_b_y, 0.52, 0.48, "H", fc="#98c7ff", fontsize=18)
    _draw_measure_dqaoa(ax, x_m_b, comm_b_y, size=0.42)
    _draw_reset_dqaoa(ax, x_rst_b, comm_b_y, size=0.46)

    # Measurement/correction boxes -> cat_measure bus.
    # These lines connect to the gate/box edges, not to the horizontal wires.
    for xx, yy, hh in [
        (x_m_a, comm_a_y, 0.42),
        (x_m_b, comm_b_y, 0.42),
        (x_xcorr, yj, 0.48),
        (x_zcorr, yi, 0.48),
    ]:
        _draw_vertical_edge_connector(ax, xx, yy, hh, cat_y, 0.0)
        _draw_bus_dot_dqaoa(ax, xx, cat_y)

    return {
        "end_x": x + 4.45,
        "last_m_x": x_m_b,
        "rightmost_x": x_rst_b + 0.23,
    }


def save_dqaoa_style_qaoa_circuit_figure(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    p: int = 1,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    output_path: str = "distributed_qaoa_style.png",
    dpi: int = 300,
    show_numeric_coeffs: bool = False,
    max_quadratic_terms_to_draw: Optional[int] = None,
    include_linear_layer: bool = True,
    include_mixer_layer: bool = True,
    trim_right_after_mixer: bool = True,
    align_mixer_to_last_measure: bool = True,
    wire_label_fontsize: int = 15,
    comm_label_fontsize: int = 15,
    label_fontweight: str = "bold",
    draw_qpu_group_boxes: bool = True,
    qpu_group_box_lw: float = 2.2,
    qpu_group_box_pad_y: float = 0.58,
    qpu_group_box_left_x: float = 0.04,
    qpu_group_box_right_pad: float = 0.06,
):
    """
    Draw a custom distributed QAOA circuit figure in a DVQE-like style,
    but based on the actual QUBO instance, allocation, and TeleGate logic.

    For the shortened paper figure like
    distributed_dqaoa_shortened_rx_end_readable_edited.png, call this with
    max_quadratic_terms_to_draw=3 and include_mixer_layer=True.

    The shortened version does not crop a half-drawn TeleGate block. Instead,
    it stops drawing after the selected number of quadratic terms, then draws
    the final Rx mixer gates immediately after that point.

    If align_mixer_to_last_measure=True, the final Rx column is shifted left so
    that it is vertically aligned with the last M box of the final TeleGate
    block. This is only a visual-layout option for the paper figure.

    This version uses larger paper-readable gate blocks and bolder/larger
    labels inside each block.
    """
    require_matplotlib()

    if p < 1:
        raise ValueError("p must be >= 1")

    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    setup = _resolve_plot_setup_for_dqaoa(
        instance=instance,
        p_eval=p,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
    )
    allocation = setup["allocation"]

    # Order wires by QPU then local slot to mimic distributed layout.
    ordered_vars = sorted(range(instance.n), key=lambda i: (allocation[i][0], allocation[i][1]))

    # y positions
    y_gap = 1.05
    y_positions = {}
    y_top = 0.0
    for idx, var in enumerate(ordered_vars):
        y_positions[var] = y_top - idx * y_gap

    comm_a_y = y_top - len(ordered_vars) * y_gap - 0.75
    comm_b_y = comm_a_y - y_gap
    cat_y = comm_b_y - 0.90

    # Wire labels like the DVQE-style example.
    wire_labels = {}
    for var in ordered_vars:
        q, slot = allocation[var]
        wire_labels[var] = f"qpu{q}_{slot}"

    # Collect quadratic terms in draw order.
    # This is the key point for deleting the red-marked partial right-side pieces:
    # do not draw those later terms in the first place.
    quad_items = sorted(instance.quadratic.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if max_quadratic_terms_to_draw is not None:
        quad_items = quad_items[:int(max_quadratic_terms_to_draw)]

    # Width estimate. The cross-QPU term is wider now because spacing was increased.
    n_lin = sum(1 for ai in instance.linear if abs(ai) > 1e-14) if include_linear_layer else 0
    n_quad = sum(1 for (_, bij) in quad_items if abs(bij) > 1e-14)
    n_cross = sum(
        1
        for (i, j), bij in quad_items
        if abs(bij) > 1e-14 and allocation[i][0] != allocation[j][0]
    )
    n_local = n_quad - n_cross

    width_est = 2.2 + 0.55 * len(ordered_vars) + p * (
        (1.1 if include_linear_layer else 0.0)
        + 2.10 * n_local
        + 10.80 * n_cross
        + (1.30 if include_mixer_layer else 0.0)
    )

    fig_w = max(12.5, min(34.0, width_est))
    fig_h = max(4.5, 0.72 * (len(ordered_vars) + 3.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_start = 0.6
    x_end_wire = width_est + 1.20

    # Draw wires.
    # The visible right edge will be trimmed after the mixer gates by xlim.
    for var in ordered_vars:
        y = y_positions[var]
        ax.plot([x_start, x_end_wire], [y, y], color="black", linewidth=1.1, zorder=0)
        ax.text(
            0.30,
            y,
            wire_labels[var],
            ha="right",
            va="center",
            fontsize=wire_label_fontsize,
            fontweight=label_fontweight,
            style="italic",
        )

    ax.plot([x_start, x_end_wire], [comm_a_y, comm_a_y], color="black", linewidth=1.1, zorder=0)
    ax.plot([x_start, x_end_wire], [comm_b_y, comm_b_y], color="black", linewidth=1.1, zorder=0)
    ax.text(
        0.30,
        comm_a_y,
        "comm_qpu0",
        ha="right",
        va="center",
        fontsize=comm_label_fontsize,
        fontweight=label_fontweight,
    )
    ax.text(
        0.30,
        comm_b_y,
        "comm_qpu1",
        ha="right",
        va="center",
        fontsize=comm_label_fontsize,
        fontweight=label_fontweight,
    )

    ax.plot([x_start, x_end_wire], [cat_y, cat_y], color="#7f8fa6", linewidth=1.0, zorder=0)
    ax.text(
        0.30,
        cat_y,
        "cat_measure",
        ha="right",
        va="center",
        fontsize=comm_label_fontsize,
        fontweight=label_fontweight,
    )

    # Left-side initialization.
    x = 0.95
    for var in ordered_vars:
        _draw_box_dqaoa(ax, x, y_positions[var], 0.52, 0.48, "H", fc="#98c7ff", fontsize=18)
    x += 1.00

    # Track the last measurement column so the final Rx mixer column can be
    # aligned above it when requested.
    last_measure_x_for_mixer = None
    rightmost_drawn_x = x_end_wire

    # Build one layer and repeat p times.
    for layer in range(p):
        # Linear layer.
        if include_linear_layer:
            for var in ordered_vars:
                ai = instance.linear[var]
                if abs(ai) > 1e-14:
                    txt = r"$\mathbf{R}_{z}$" if not show_numeric_coeffs else f"$R_z$\n-γ·{_fmt_num(ai)}"
                    _draw_box_dqaoa(ax, x, y_positions[var], 0.92, 0.48, txt, fc="#c9a3ff", fontsize=18)
            x += 1.18

        # Quadratic terms.
        for (i, j), bij in quad_items:
            if abs(bij) <= 1e-14:
                continue

            yi = y_positions[i]
            yj = y_positions[j]
            same_qpu = allocation[i][0] == allocation[j][0]

            if same_qpu:
                # Local ZZ via CX-Rz-CX.
                _draw_cnot_dqaoa(ax, x, yi, yj)
                txt = r"$\mathbf{R}_{z}$" if not show_numeric_coeffs else f"$R_z$\nγ·{_fmt_num(bij/2.0)}"
                _draw_box_dqaoa(ax, x + 0.68, yj, 0.82, 0.46, txt, fc="#c9a3ff", fontsize=18)
                _draw_cnot_dqaoa(ax, x + 1.38, yi, yj)
                x += 1.95

            else:
                # TeleGate-based remote CX, then Rz, then TeleGate-based remote CX.
                remote1 = _draw_telegate_remote_cx_clean(
                    ax=ax,
                    x=x,
                    yi=yi,
                    yj=yj,
                    comm_a_y=comm_a_y,
                    comm_b_y=comm_b_y,
                    cat_y=cat_y,
                )
                end1 = remote1["end_x"]
                last_measure_x_for_mixer = remote1["last_m_x"]
                rightmost_drawn_x = max(rightmost_drawn_x, remote1["rightmost_x"])

                # Central Rz.
                rz_x = end1 + 0.70
                txt = r"$\mathbf{R}_{z}$" if not show_numeric_coeffs else f"$R_z$\nγ·{_fmt_num(bij/2.0)}"
                _draw_box_dqaoa(ax, rz_x, yj, 0.82, 0.46, txt, fc="#c9a3ff", fontsize=18)

                # Second remote CX.
                x2 = rz_x + 1.25
                remote2 = _draw_telegate_remote_cx_clean(
                    ax=ax,
                    x=x2,
                    yi=yi,
                    yj=yj,
                    comm_a_y=comm_a_y,
                    comm_b_y=comm_b_y,
                    cat_y=cat_y,
                )
                end2 = remote2["end_x"]
                last_measure_x_for_mixer = remote2["last_m_x"]
                rightmost_drawn_x = max(rightmost_drawn_x, remote2["rightmost_x"])

                x = end2 + 0.65

        # Mixer layer.
        if include_mixer_layer:
            mixer_x = x
            if align_mixer_to_last_measure and last_measure_x_for_mixer is not None:
                # Shift only the Rx column left so it is vertically above the
                # final M gate of the final TeleGate block.
                mixer_x = float(last_measure_x_for_mixer)

            for var in ordered_vars:
                _draw_box_dqaoa(
                    ax,
                    mixer_x,
                    y_positions[var],
                    0.86,
                    0.48,
                    r"$\mathbf{R}_{x}$",
                    fc="#c9a3ff",
                    fontsize=18,
                )

            rightmost_drawn_x = max(rightmost_drawn_x, mixer_x + 0.43)
            x = max(x, mixer_x + 1.10)

    # Trim the visible right edge without cutting any TeleGate reset boxes.
    if trim_right_after_mixer:
        x_right = max(rightmost_drawn_x + 0.18, 1.0)
    else:
        x_right = x_end_wire

    # Optional dashed QPU-group outlines for the paper figure.
    # QPU 0 is yellow and QPU 1 is green by default, matching your edited example.
    if draw_qpu_group_boxes:
        qpu_box_colors = {
            0: "#f2c300",  # yellow outline
            1: "#00b050",  # green outline
        }
        
        qpu_box_fill_colors = {
            0: "#fff2cc",  # light yellow background
            1: "#d9ead3",  # light green background
        }
        qpus_present = sorted({allocation[var][0] for var in ordered_vars})
        for q in qpus_present:
            vars_on_qpu = [var for var in ordered_vars if allocation[var][0] == q]
            if not vars_on_qpu:
                continue
            ys = [y_positions[var] for var in vars_on_qpu]
            y_box_top = max(ys) + qpu_group_box_pad_y
            y_box_bottom = min(ys) - qpu_group_box_pad_y
            _draw_qpu_group_box_dqaoa(
                ax=ax,
                x_left=qpu_group_box_left_x,
                x_right=x_right - qpu_group_box_right_pad,
                y_top=y_box_top,
                y_bottom=y_box_bottom,
                color=qpu_box_colors.get(q, "#777777"),
                fill_color=qpu_box_fill_colors.get(q, "#eeeeee"),
                lw=qpu_group_box_lw,
                fill_alpha=0.45,
            )
    # Put title at the true middle top of the whole visible circuit
    title_y = 0.72
    ax.text(
        0.5 * x_right,
        title_y,
        f"Distributed QAOA circuit with (p={p})",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="normal",
        zorder=30,
    )

    ax.set_xlim(0.0, x_right)
    ax.set_ylim(cat_y - 0.45, 0.85)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": output_path,
        "mode": "telegate_explicit_qaoa",
        "p": int(p),
        "assignment": setup["assignment"],
        "allocation": allocation,
        "selected_allocation_mode": setup["selected_allocation_mode"],
        "remote_mode": setup["remote_mode"],
        "cross_qpu_terms": count_cross_qpu_quadratic_terms(instance.quadratic, allocation),
        "num_qpus": setup["num_qpus"],
        "capacities": setup["capacities"],
        "quadratic_terms_drawn": int(n_quad),
        "align_mixer_to_last_measure": bool(align_mixer_to_last_measure),
        "wire_label_fontsize": int(wire_label_fontsize),
        "comm_label_fontsize": int(comm_label_fontsize),
        "draw_qpu_group_boxes": bool(draw_qpu_group_boxes),
    }

def _pick_representative_qubo_terms(
    instance: QUBOInstance,
    allocation: Dict[int, Tuple[int, int]],
) -> Dict[str, Optional[Tuple[int, int, float]]]:
    """
    Pick one representative local quadratic term and one representative
    cross-QPU quadratic term for a compact paper-style schematic.
    """
    local_term = None
    cross_term = None

    for (i, j), bij in sorted(instance.quadratic.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if abs(bij) <= 1e-14:
            continue
        if allocation[i][0] == allocation[j][0]:
            if local_term is None:
                local_term = (i, j, float(bij))
        else:
            if cross_term is None:
                cross_term = (i, j, float(bij))
        if local_term is not None and cross_term is not None:
            break

    return {
        "local_term": local_term,
        "cross_term": cross_term,
    }


# =============================================================================
# Paper-style circuit schematic helpers
# =============================================================================
def resolve_qubo_visualization_setup(
    instance: QUBOInstance,
    mode: str,
    p_eval: int,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Resolve allocation and remote mode exactly as the solver does,
    but without running optimization.
    """
    if mode not in {
        "monolithic_qaoa",
        "abstract_distributed_qaoa",
        "telegate_explicit_qaoa",
    }:
        raise ValueError(
            "mode must be one of "
            "{'monolithic_qaoa', 'abstract_distributed_qaoa', 'telegate_explicit_qaoa'}"
        )

    n = instance.n

    if mode == "monolithic_qaoa":
        num_qpus_eff = 1
        capacities_eff = [n]
        allocation = make_allocation(
            n=n,
            num_qpus=1,
            capacities=capacities_eff,
            min_used_qpus=1,
        )
        selected_mode = "monolithic"
        remote_mode = "abstract"
    else:
        if num_qpus < 2:
            raise ValueError("Distributed modes require num_qpus >= 2.")
        if capacities is None:
            capacities = make_balanced_capacities(n, num_qpus)

        candidates = build_allocation_candidates_qubo(
            instance=instance,
            num_qpus=num_qpus,
            capacities=capacities,
            p_eval=p_eval,
            manual_assignment=manual_assignment,
            min_used_qpus=2,
        )
        best_candidate = select_best_allocation_candidate(candidates)

        allocation = best_candidate["allocation"]
        selected_mode = best_candidate["mode"]
        num_qpus_eff = num_qpus
        capacities_eff = capacities
        remote_mode = "abstract" if mode == "abstract_distributed_qaoa" else "telegate_explicit"

    return {
        "allocation": allocation,
        "selected_allocation_mode": selected_mode,
        "num_qpus_eff": int(num_qpus_eff),
        "capacities_eff": [int(v) for v in capacities_eff],
        "remote_mode": remote_mode,
        "assignment": allocation_to_assignment(instance.n, allocation),
    }


def _fmt_coeff(x: float, ndigits: int = 3) -> str:
    return f"{float(x):.{ndigits}g}"


def _draw_rect(ax, x, y, w, h, text, fontsize=7, fc="#caa7ff", ec="black", lw=1.0, rotation=0):
    from matplotlib.patches import Rectangle
    rect = Rectangle((x - w / 2.0, y - h / 2.0), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, rotation=rotation)


def _draw_measure(ax, x, y, size=0.26, fontsize=7):
    from matplotlib.patches import Rectangle
    rect = Rectangle((x - size / 2.0, y - size / 2.0), size, size, facecolor="white", edgecolor="black", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, y, "M", ha="center", va="center", fontsize=fontsize)


def _draw_cnot(ax, x, y_ctrl, y_tgt, color="#6aa6ff", lw=1.2):
    from matplotlib.patches import Circle
    ax.plot([x, x], [y_ctrl, y_tgt], color=color, linewidth=lw)
    ax.add_patch(Circle((x, y_ctrl), 0.045, facecolor=color, edgecolor=color, linewidth=lw))
    ax.add_patch(Circle((x, y_tgt), 0.085, facecolor="white", edgecolor=color, linewidth=lw))
    ax.plot([x - 0.06, x + 0.06], [y_tgt, y_tgt], color=color, linewidth=lw)
    ax.plot([x, x], [y_tgt - 0.06, y_tgt + 0.06], color=color, linewidth=lw)

def _draw_bus_dot(ax, x, y_bus, color="black", r=0.06):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y_bus), r, facecolor=color, edgecolor=color, linewidth=1.0))


def _draw_vertical_classical_to_bus(
    ax,
    x,
    y_from,
    y_bus,
    color="#7a7a7a",
    lw=1.2,
    arrow_at_bus=True,
    dot_on_bus=True,
):
    """
    Draw a straight vertical classical connection from a measurement or
    classically controlled gate to the cat_measure bus.
    """
    ax.plot([x, x], [y_from, y_bus], color=color, linewidth=lw)
    if arrow_at_bus:
        # tiny arrowhead near the bus
        dy = -0.16 if y_from > y_bus else 0.16
        ax.annotate(
            "",
            xy=(x, y_bus),
            xytext=(x, y_bus + dy),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        )
    if dot_on_bus:
        _draw_bus_dot(ax, x, y_bus, color="black", r=0.05)


def _draw_classical_bus_label(ax, x, y_bus, text, fontsize=7, color="#444444"):
    ax.text(x, y_bus - 0.18, text, ha="center", va="top", fontsize=fontsize, color=color)


def _draw_classical_arrow(ax, x0, y0, x1, y1, color="#666666", rad=0.12):
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->",
        mutation_scale=10,
        linewidth=1.0,
        linestyle="dashed",
        color=color,
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(arr)


def _wire_y_positions(n_data: int, explicit_telegate: bool) -> Dict[str, Any]:
    """
    Return y positions for data and communication wires.
    Top wire has highest y.
    """
    data_y = [-(i) for i in range(n_data)]
    if explicit_telegate:
        comm_a_y = -n_data - 0.8
        comm_b_y = -n_data - 1.8
    else:
        comm_a_y = None
        comm_b_y = None
    return {
        "data_y": data_y,
        "comm_a_y": comm_a_y,
        "comm_b_y": comm_b_y,
    }


def save_qubo_mode_paper_schematic(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    mode: str = "monolithic_qaoa",
    p: int = 1,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    output_path: str = "qubo_mode_schematic.png",
    dpi: int = 300,
    show_coeff_values: bool = True,
    include_final_measurements: bool = False,
    title: Optional[str] = None,
):
    """
    Draw a compact paper-style schematic for monolithic QAOA or distributed DQAOA.

    This function does NOT draw the full exact dense circuit.
    Instead, it uses the actual QUBO instance and allocation, then shows:
      - initialization
      - one compressed linear cost layer
      - one representative local ZZ term
      - one representative cross-QPU TeleGate ZZ term (if present)
      - mixer layer

    This is intended for figures in the paper, not debugging the full circuit.
    """
    require_matplotlib()

    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    setup = resolve_qubo_visualization_setup(
        instance=instance,
        mode=mode,
        p_eval=p,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
    )

    allocation = setup["allocation"]
    assignment = setup["assignment"]
    remote_mode = setup["remote_mode"]
    explicit_telegate = (remote_mode == "telegate_explicit")
    n = instance.n

    reps = _pick_representative_qubo_terms(instance, allocation)
    local_term = reps["local_term"]
    cross_term = reps["cross_term"]

    yinfo = _wire_y_positions(n_data=n, explicit_telegate=explicit_telegate)
    data_y = yinfo["data_y"]
    comm_a_y = yinfo["comm_a_y"]
    comm_b_y = yinfo["comm_b_y"]

    # Compact figure size
    fig_w = 13.5 if explicit_telegate else 10.5
    fig_h = max(3.0, 0.72 * (n + (2 if explicit_telegate else 0)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x0 = 0.6
    x_end = 12.8 if explicit_telegate else 9.6

    # Draw wires
    for i in range(n):
        ax.plot([x0, x_end], [data_y[i], data_y[i]], color="black", linewidth=1.1)
        ax.text(0.25, data_y[i], f"$q_{{{i}}}$", ha="right", va="center", fontsize=10)
        if mode != "monolithic_qaoa":
            ax.text(
                0.28, data_y[i] - 0.26,
                f"QPU {assignment[i]}",
                ha="right", va="center",
                fontsize=7, color="#555555"
            )

    if explicit_telegate:
        ax.plot([x0, x_end], [comm_a_y, comm_a_y], color="black", linewidth=1.1)
        ax.plot([x0, x_end], [comm_b_y, comm_b_y], color="black", linewidth=1.1)
        ax.text(0.25, comm_a_y, "comm$_a$", ha="right", va="center", fontsize=10)
        ax.text(0.25, comm_b_y, "comm$_b$", ha="right", va="center", fontsize=10)

    # -------------------------
    # 1) Initial Hadamards
    # -------------------------
    x = 1.0
    for i in range(n):
        _draw_rect(ax, x, data_y[i], 0.34, 0.34, "H", fontsize=8, fc="#9cc7ff")

    # -------------------------
    # 2) Compressed linear layer
    # -------------------------
    x = 2.0
    y_top = data_y[0]
    y_bot = data_y[-1]
    yc = 0.5 * (y_top + y_bot)
    h = abs(y_top - y_bot) + 0.7
    label = "Linear cost layer\n$R_Z(-\\gamma \\ell_i)$"
    _draw_rect(ax, x, yc, 1.45, h, label, fontsize=8, fc="#d8b8ff")

    # -------------------------
    # 3) Representative local ZZ
    # -------------------------
    x = 4.0
    if local_term is not None:
        i, j, bij = local_term
        y_top = min(data_y[i], data_y[j])
        y_bot = max(data_y[i], data_y[j])
        yc = 0.5 * (y_top + y_bot)
        h = abs(y_top - y_bot) + 0.55
        if show_coeff_values:
            txt = f"Local $ZZ$\n$(q_{i},q_{j})$\n$b_{{{i}{j}}}={_fmt_coeff(bij)}$"
        else:
            txt = f"Local $ZZ$\n$(q_{i},q_{j})$"
        _draw_rect(ax, x, yc, 1.1, h, txt, fontsize=7, fc="#caa7ff")
    else:
        _draw_rect(ax, x, yc, 1.1, 0.55, "No local\nquadratic term", fontsize=7, fc="#f1f1f1")

    # -------------------------
    # 4) Representative remote TeleGate ZZ
    # -------------------------
    if explicit_telegate:
        x = 7.1
        if cross_term is not None:
            i, j, bij = cross_term

            # First TeleGate CX
            yc = 0.5 * (data_y[i] + comm_b_y)
            h = abs(data_y[i] - comm_b_y) + 0.65
            _draw_rect(ax, x, yc, 0.95, h, "TeleGate\nCX", fontsize=8, fc="#b9f2e6")

            # Measurements
            _draw_measure(ax, x + 0.55, comm_a_y, size=0.22, fontsize=6)
            _draw_measure(ax, x + 0.80, comm_b_y, size=0.22, fontsize=6)

            # Conditional corrections
            _draw_rect(ax, x + 0.58, data_y[j], 0.25, 0.25, "X", fontsize=7, fc="#9cf0d1")
            _draw_rect(ax, x + 0.58, data_y[i], 0.25, 0.25, "Z", fontsize=7, fc="#9cf0d1")

            _draw_classical_arrow(ax, x + 0.55, comm_a_y - 0.14, x + 0.58, data_y[j] - 0.16)
            _draw_classical_arrow(ax, x + 0.80, comm_b_y + 0.14, x + 0.58, data_y[i] + 0.16)

            # Middle phase
            x_mid = x + 1.25
            if show_coeff_values:
                txt_mid = f"$R_Z$\n$\\frac{{\\gamma b_{{{i}{j}}}}}{{2}}$\n({_fmt_coeff(bij)})"
            else:
                txt_mid = "$R_Z$"
            _draw_rect(ax, x_mid, data_y[j], 0.85, 0.42, txt_mid, fontsize=7, fc="#d8b8ff")

            # Second TeleGate CX
            x2 = x + 2.05
            yc = 0.5 * (data_y[i] + comm_b_y)
            h = abs(data_y[i] - comm_b_y) + 0.65
            _draw_rect(ax, x2, yc, 0.95, h, "TeleGate\nCX", fontsize=8, fc="#b9f2e6")

            _draw_measure(ax, x2 + 0.55, comm_a_y, size=0.22, fontsize=6)
            _draw_measure(ax, x2 + 0.80, comm_b_y, size=0.22, fontsize=6)

            _draw_rect(ax, x2 + 0.58, data_y[j], 0.25, 0.25, "X", fontsize=7, fc="#9cf0d1")
            _draw_rect(ax, x2 + 0.58, data_y[i], 0.25, 0.25, "Z", fontsize=7, fc="#9cf0d1")

            _draw_classical_arrow(ax, x2 + 0.55, comm_a_y - 0.14, x2 + 0.58, data_y[j] - 0.16)
            _draw_classical_arrow(ax, x2 + 0.80, comm_b_y + 0.14, x2 + 0.58, data_y[i] + 0.16)

            # Annotation
            ann_x = x + 1.1
            ann_y = comm_b_y - 0.75
            ax.text(
                ann_x, ann_y,
                f"Representative cross-QPU term: $(q_{i},q_{j})$",
                ha="center", va="center", fontsize=8
            )
        else:
            _draw_rect(ax, x, 0.5 * (data_y[0] + comm_b_y), 1.4, 1.0,
                       "No cross-QPU\nquadratic term", fontsize=8, fc="#f1f1f1")

    # -------------------------
    # 5) Mixer layer
    # -------------------------
    x = 11.4 if explicit_telegate else 6.0
    for i in range(n):
        _draw_rect(ax, x, data_y[i], 0.6, 0.38, "$R_X$\n$2\\beta$", fontsize=7, fc="#d8b8ff")

    # Optional final measurements
    if include_final_measurements:
        x_m = x + 0.8
        for i in range(n):
            _draw_measure(ax, x_m, data_y[i], size=0.24, fontsize=7)

    # Title
    if title is None:
        if mode == "monolithic_qaoa":
            title = f"Monolithic QAOA schematic (p={p})"
        elif mode == "abstract_distributed_qaoa":
            title = f"Abstract distributed QAOA schematic (p={p})"
        else:
            title = f"Distributed DQAOA schematic (p={p})"

    ax.set_title(title, fontsize=12)

    # Limits
    y_bottom = (comm_b_y - 0.9) if explicit_telegate else (data_y[-1] - 0.7)
    ax.set_xlim(0.0, x_end + 0.2)
    ax.set_ylim(y_bottom, 0.8)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": output_path,
        "mode": mode,
        "p": int(p),
        "assignment": assignment,
        "allocation": allocation,
        "selected_allocation_mode": setup["selected_allocation_mode"],
        "remote_mode": remote_mode,
        "cross_qpu_terms": count_cross_qpu_quadratic_terms(instance.quadratic, allocation),
        "num_qpus": setup["num_qpus_eff"],
        "capacities": setup["capacities_eff"],
        "representative_local_term": local_term,
        "representative_cross_term": cross_term,
    }





def save_monolithic_qaoa_style_circuit_figure(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    p: int = 1,
    output_path: str = "monolithic_qaoa_style_big_blocks.png",
    dpi: int = 500,
    show_numeric_coeffs: bool = False,
    max_quadratic_terms_to_draw: Optional[int] = None,
    include_linear_layer: bool = True,
    include_mixer_layer: bool = True,

    # New readability controls
    wire_label_fontsize: int = 18,
    gate_fontsize: int = 18,
    title_fontsize: int = 16,
    label_fontweight: str = "bold",
):
    """
    Draw a custom monolithic QAOA ansatz figure with larger paper-readable
    gate blocks, larger labels inside the blocks, and larger q0--q5 labels.
    """
    require_matplotlib()

    if p < 1:
        raise ValueError("p must be >= 1")

    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    n = instance.n

    # Monolithic ordering: q0, q1, ..., q(n-1)
    ordered_vars = list(range(n))

    y_gap = 1.05
    y_positions = {var: -(idx * y_gap) for idx, var in enumerate(ordered_vars)}

    quad_items = sorted(instance.quadratic.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    if max_quadratic_terms_to_draw is not None:
        quad_items = quad_items[:int(max_quadratic_terms_to_draw)]

    n_lin = sum(1 for ai in instance.linear if abs(ai) > 1e-14) if include_linear_layer else 0
    n_quad = sum(1 for (_, bij) in quad_items if abs(bij) > 1e-14)

    # Slightly larger spacing because the blocks are larger now.
    width_est = 2.6 + 0.65 * n + p * (
        (1.30 if include_linear_layer else 0.0)
        + 2.05 * n_quad
        + (1.25 if include_mixer_layer else 0.0)
    )

    fig_w = max(10.5, min(30.0, width_est))
    fig_h = max(4.2, 0.78 * (n + 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_start = 0.65
    x_end = width_est + 0.85

    # Background wires and larger q labels
    for var in ordered_vars:
        y = y_positions[var]
        ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.2, zorder=0)
        ax.text(
            0.32,
            y,
            f"$q_{{{var}}}$",
            ha="right",
            va="center",
            fontsize=wire_label_fontsize,
            fontweight=label_fontweight,
            style="italic",
        )

    # Left-side initialization
    x = 1.00
    for var in ordered_vars:
        _draw_box(
            ax,
            x,
            y_positions[var],
            0.52,
            0.48,
            "H",
            fc="#98c7ff",
            fontsize=gate_fontsize,
            lw=1.2,
        )

    x += 1.05

    for layer in range(p):
        # Linear Rz layer
        if include_linear_layer:
            for var in ordered_vars:
                ai = instance.linear[var]
                if abs(ai) > 1e-14:
                    txt = r"$\mathbf{R}_{z}$" if not show_numeric_coeffs else f"$R_z$\n-γ·{_fmt_num(ai)}"
                    _draw_box(
                        ax,
                        x,
                        y_positions[var],
                        0.92,
                        0.48,
                        txt,
                        fc="#c9a3ff",
                        fontsize=gate_fontsize,
                        lw=1.2,
                    )
            x += 1.25

        # Quadratic local ZZ terms
        for (i, j), bij in quad_items:
            if abs(bij) <= 1e-14:
                continue

            yi = y_positions[i]
            yj = y_positions[j]

            _draw_cnot(ax, x, yi, yj)

            txt = r"$\mathbf{R}_{z}$" if not show_numeric_coeffs else f"$R_z$\nγ·{_fmt_num(bij/2.0)}"
            _draw_box(
                ax,
                x + 0.70,
                yj,
                0.82,
                0.46,
                txt,
                fc="#c9a3ff",
                fontsize=gate_fontsize,
                lw=1.2,
            )

            _draw_cnot(ax, x + 1.40, yi, yj)

            x += 1.95

        # Mixer Rx layer
        if include_mixer_layer:
            for var in ordered_vars:
                _draw_box(
                    ax,
                    x,
                    y_positions[var],
                    0.86,
                    0.48,
                    r"$\mathbf{R}_{x}$",
                    fc="#c9a3ff",
                    fontsize=gate_fontsize,
                    lw=1.2,
                )
            x += 1.10

    # Centered title
    ax.text(
        0.5 * x_end,
        0.72,
        f"Monolithic QAOA circuit with (p={p})",
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="normal",
        zorder=30,
    )

    ax.set_xlim(0.0, x_end)
    ax.set_ylim(y_positions[ordered_vars[-1]] - 0.60, 0.90)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": output_path,
        "mode": "monolithic_qaoa",
        "p": int(p),
        "num_qubits": n,
        "quadratic_terms_drawn": int(sum(1 for (_, bij) in quad_items if abs(bij) > 1e-14)),
        "linear_terms_present": int(n_lin),
        "wire_label_fontsize": int(wire_label_fontsize),
        "gate_fontsize": int(gate_fontsize),
    }



# =============================================================================
# Circuit-visualization helpers
# =============================================================================
def resolve_qubo_mode_circuit_setup(
    instance: QUBOInstance,
    mode: str,
    p_eval: int,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Resolve the allocation and remote mode exactly the way the solver would
    for a given QAOA/DQAOA mode, but without running optimization.

    Returns a dictionary containing:
      - allocation
      - selected_allocation_mode
      - num_qpus_eff
      - capacities_eff
      - remote_mode
      - assignment
    """
    if mode not in {
        "monolithic_qaoa",
        "abstract_distributed_qaoa",
        "telegate_explicit_qaoa",
    }:
        raise ValueError(
            "mode must be one of "
            "{'monolithic_qaoa', 'abstract_distributed_qaoa', 'telegate_explicit_qaoa'}"
        )

    n = instance.n

    if mode == "monolithic_qaoa":
        num_qpus_eff = 1
        capacities_eff = [n]
        allocation = make_allocation(
            n=n,
            num_qpus=1,
            capacities=capacities_eff,
            min_used_qpus=1,
        )
        selected_mode = "monolithic"
        remote_mode = "abstract"
    else:
        if num_qpus < 2:
            raise ValueError("Distributed modes require num_qpus >= 2.")
        if capacities is None:
            capacities = make_balanced_capacities(n, num_qpus)

        candidates = build_allocation_candidates_qubo(
            instance=instance,
            num_qpus=num_qpus,
            capacities=capacities,
            p_eval=p_eval,
            manual_assignment=manual_assignment,
            min_used_qpus=2,
        )
        best_candidate = select_best_allocation_candidate(candidates)

        allocation = best_candidate["allocation"]
        selected_mode = best_candidate["mode"]
        num_qpus_eff = num_qpus
        capacities_eff = capacities
        remote_mode = "abstract" if mode == "abstract_distributed_qaoa" else "telegate_explicit"

    return {
        "allocation": allocation,
        "selected_allocation_mode": selected_mode,
        "num_qpus_eff": int(num_qpus_eff),
        "capacities_eff": [int(v) for v in capacities_eff],
        "remote_mode": remote_mode,
        "assignment": allocation_to_assignment(instance.n, allocation),
    }


def build_qubo_mode_circuit(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    mode: str = "monolithic_qaoa",
    p: int = 1,
    gammas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    circuit_view: str = "parametrized",
    measure: bool = False,
    train_cfg: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """
    Build a circuit for visualization for an arbitrary QUBO instance.

    Parameters
    ----------
    mode:
        'monolithic_qaoa', 'abstract_distributed_qaoa', or 'telegate_explicit_qaoa'

    circuit_view:
        'parametrized' -> symbolic gamma/beta circuit from build_parametrized_qubo_qaoa_circuit
        'bound'        -> numeric-angle circuit from build_qubo_qaoa_circuit
        'transpiled'   -> transpiled parameterized circuit from CachedQuboCircuitRunner

    measure:
        Whether to include final measurements on the data qubits.
        For paper-style ansatz figures, measure=False is usually cleaner.

    Notes
    -----
    - For circuit_view='bound', if gammas/betas are omitted, default illustrative
      values are used.
    - For distributed explicit mode with measure=False, intermediate TeleGate
      measurements are still present because they are required by the remote routine.
    """
    require_qiskit()

    if p < 1:
        raise ValueError("p must be >= 1")

    if circuit_view not in {"parametrized", "bound", "transpiled"}:
        raise ValueError("circuit_view must be one of {'parametrized', 'bound', 'transpiled'}")

    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)

    setup = resolve_qubo_mode_circuit_setup(
        instance=instance,
        mode=mode,
        p_eval=p,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
    )

    allocation = setup["allocation"]
    remote_mode = setup["remote_mode"]

    if train_cfg is None:
        train_cfg = TrainConfig()

    result: Dict[str, Any] = {
        "mode": mode,
        "instance": instance,
        "p": int(p),
        "allocation": allocation,
        "assignment": setup["assignment"],
        "selected_allocation_mode": setup["selected_allocation_mode"],
        "num_qpus": setup["num_qpus_eff"],
        "capacities": setup["capacities_eff"],
        "remote_mode": remote_mode,
        "measure": bool(measure),
        "circuit_view": circuit_view,
    }

    if circuit_view == "parametrized":
        qc, gamma_params, beta_params, stats = build_parametrized_qubo_qaoa_circuit(
            instance=instance,
            p=p,
            allocation=allocation,
            measure=measure,
            remote_mode=remote_mode,
        )
        result.update({
            "circuit": qc,
            "gamma_params": gamma_params,
            "beta_params": beta_params,
            "stats": stats,
        })
        return result

    if circuit_view == "transpiled":
        runner = get_cached_qubo_runner(
            instance=instance,
            p=p,
            allocation=allocation,
            remote_mode=remote_mode,
            train_cfg=train_cfg,
            measure=measure,
        )
        result.update({
            "circuit": runner.transpiled_qc,
            "stats": dict(runner.static_stats),
        })
        return result

    # circuit_view == "bound"
    if gammas is None:
        gammas = [0.3] * p
    if betas is None:
        betas = [0.7] * p
    if len(gammas) != p or len(betas) != p:
        raise ValueError("gammas and betas must both have length p")

    qc, stats = build_qubo_qaoa_circuit(
        instance=instance,
        p=p,
        gammas=list(gammas),
        betas=list(betas),
        allocation=allocation,
        measure=measure,
        remote_mode=remote_mode,
    )
    result.update({
        "circuit": qc,
        "gammas": [float(v) for v in gammas],
        "betas": [float(v) for v in betas],
        "stats": stats,
    })
    return result


def save_qubo_mode_circuit_figure(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    mode: str = "monolithic_qaoa",
    p: int = 1,
    gammas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    circuit_view: str = "parametrized",
    measure: bool = False,
    train_cfg: Optional[TrainConfig] = None,
    output_path: str = "qubo_circuit.png",
    fold: int = -1,
    idle_wires: bool = False,
    dpi: int = 300,
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a monolithic/distributed QAOA circuit for an arbitrary QUBO instance
    and save it as a figure.

    Returns the same metadata as build_qubo_mode_circuit, plus output_path.
    """
    require_matplotlib()

    built = build_qubo_mode_circuit(
        H=H,
        f=f,
        c0=c0,
        name=name,
        mode=mode,
        p=p,
        gammas=gammas,
        betas=betas,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
        circuit_view=circuit_view,
        measure=measure,
        train_cfg=train_cfg,
    )

    qc = built["circuit"]
    fig = qc.draw(
        output="mpl",
        fold=fold,
        idle_wires=idle_wires,
        style=style if style is not None else {"fontsize": 9},
    )
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    built["output_path"] = output_path
    return built

























# =============================================================================
# Count parsing and metrics
# =============================================================================
def extract_data_bits_from_full_count_key(bs_raw: str, n: int) -> str:
    compact = bs_raw.replace(" ", "")
    logical = compact[::-1]
    return logical[:n]


def bitstring_from_counts_key(bs_raw: str, n: int, explicit_telegate: bool) -> str:
    if explicit_telegate:
        return extract_data_bits_from_full_count_key(bs_raw, n)
    return invert_endianness(bs_raw)


def decoded_distribution_from_counts(
    counts: Dict[str, int],
    n: int,
    explicit_telegate: bool = False,
) -> Dict[str, float]:
    shots = sum(counts.values())
    out: Dict[str, float] = {}
    if shots <= 0:
        return out
    for bs_raw, ct in counts.items():
        bs = bitstring_from_counts_key(bs_raw, n, explicit_telegate)
        out[bs] = out.get(bs, 0.0) + ct / shots
    return out


def decoded_count_map_from_counts(
    counts: Dict[str, int],
    n: int,
    explicit_telegate: bool = False,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for bs_raw, ct in counts.items():
        bs = bitstring_from_counts_key(bs_raw, n, explicit_telegate)
        out[bs] = out.get(bs, 0) + int(ct)
    return out


def counts_qubo_stats(
    counts: Dict[str, int],
    instance: QUBOInstance,
    explicit_telegate: bool = False,
) -> Dict[str, Any]:
    shots = sum(counts.values())
    if shots <= 0:
        return {
            "shots": 0,
            "mean_cost": float("inf"),
            "best_cost_observed": float("inf"),
            "best_bs_qubit": None,
            "best_count": 0,
            "p_best_observed": 0.0,
            "distribution": {},
            "decoded_counts": {},
        }
    distribution = decoded_distribution_from_counts(counts, instance.n, explicit_telegate=explicit_telegate)
    decoded_counts = decoded_count_map_from_counts(counts, instance.n, explicit_telegate=explicit_telegate)
    best_cost_observed = float("inf")
    best_bs_qubit = None
    best_count = 0
    mean_cost = 0.0
    for bs_qubit, p in distribution.items():
        cost = qubo_value_from_bitstring(bs_qubit, instance)
        mean_cost += p * cost
        ct = int(decoded_counts.get(bs_qubit, 0))
        if cost < best_cost_observed - 1e-15:
            best_cost_observed = cost
            best_bs_qubit = bs_qubit
            best_count = ct
        elif abs(cost - best_cost_observed) <= 1e-15 and ct > best_count:
            best_bs_qubit = bs_qubit
            best_count = ct
    return {
        "shots": int(shots),
        "mean_cost": float(mean_cost),
        "best_cost_observed": float(best_cost_observed),
        "best_bs_qubit": best_bs_qubit,
        "best_count": int(best_count),
        "p_best_observed": float(best_count / shots),
        "distribution": distribution,
        "decoded_counts": decoded_counts,
    }


def low_cost_mass_from_distribution(
    distribution: Dict[str, float],
    instance: QUBOInstance,
    base_cost: float,
    delta: float,
) -> float:
    thr = float(base_cost + delta)
    mass = 0.0
    for bs_qubit, p in distribution.items():
        cost = qubo_value_from_bitstring(bs_qubit, instance)
        if cost <= thr:
            mass += p
    return float(mass)


def exact_cost_mass_from_distribution(
    distribution: Dict[str, float],
    instance: QUBOInstance,
    exact_optimum_cost: Optional[float],
) -> Optional[float]:
    if exact_optimum_cost is None:
        return None
    mass = 0.0
    for bs_qubit, p in distribution.items():
        cost = qubo_value_from_bitstring(bs_qubit, instance)
        if abs(cost - exact_optimum_cost) <= 1e-12:
            mass += p
    return float(mass)


def exact_bitstring_mass_from_distribution(
    distribution: Dict[str, float],
    exact_optimum_bitstring: Optional[str],
) -> Optional[float]:
    if exact_optimum_bitstring is None:
        return None
    return float(distribution.get(exact_optimum_bitstring, 0.0))


def build_elite_metrics_from_counts(
    counts: Dict[str, int],
    instance: QUBOInstance,
    explicit_telegate: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:
    decoded_counts = decoded_count_map_from_counts(counts, instance.n, explicit_telegate=explicit_telegate)
    shots = int(sum(decoded_counts.values()))
    rows: List[Dict[str, Any]] = []
    if shots <= 0:
        return {
            "elite_top_k": int(max(0, top_k)),
            "elite_rows_top_k": [],
            "elite_mass_top_k": 0.0,
            "elite_mean_cost_top_k": None,
            "num_unique_sampled": 0,
            "incumbent_cost": None,
            "incumbent_bitstring": None,
            "incumbent_count": 0,
            "p_incumbent_bitstring": 0.0,
            "p_incumbent_cost": 0.0,
        }
    for bs, ct in decoded_counts.items():
        freq = float(ct / shots)
        rows.append({
            "bitstring": bs,
            "cost": float(qubo_value_from_bitstring(bs, instance)),
            "count": int(ct),
            "frequency": freq,
        })
    rows.sort(key=lambda r: (r["cost"], -r["count"], r["bitstring"]))
    elite_top_k = int(max(0, top_k))
    elite_rows = rows[:elite_top_k] if elite_top_k > 0 else []
    if elite_rows:
        incumbent = elite_rows[0]
        incumbent_cost = float(incumbent["cost"])
        incumbent_bitstring = str(incumbent["bitstring"])
        incumbent_count = int(incumbent["count"])
        p_incumbent_bitstring = float(incumbent["frequency"])
        p_incumbent_cost = float(sum(r["frequency"] for r in rows if abs(r["cost"] - incumbent_cost) <= 1e-12))
    else:
        incumbent_cost = None
        incumbent_bitstring = None
        incumbent_count = 0
        p_incumbent_bitstring = 0.0
        p_incumbent_cost = 0.0
    elite_mass_top_k = float(sum(r["frequency"] for r in elite_rows))
    elite_mean_cost_top_k = None
    if elite_mass_top_k > 0.0:
        elite_mean_cost_top_k = float(sum(r["frequency"] * r["cost"] for r in elite_rows) / elite_mass_top_k)
    return {
        "elite_top_k": elite_top_k,
        "elite_rows_top_k": elite_rows,
        "elite_mass_top_k": elite_mass_top_k,
        "elite_mean_cost_top_k": elite_mean_cost_top_k,
        "num_unique_sampled": int(len(rows)),
        "incumbent_cost": incumbent_cost,
        "incumbent_bitstring": incumbent_bitstring,
        "incumbent_count": incumbent_count,
        "p_incumbent_bitstring": p_incumbent_bitstring,
        "p_incumbent_cost": p_incumbent_cost,
    }


def compute_final_distribution_metrics(
    counts: Dict[str, int],
    instance: QUBOInstance,
    obj_cfg: QuboObjectiveConfig,
    analysis_cfg: AnalysisConfig,
    explicit_telegate: bool = False,
    exact_optimum_cost: Optional[float] = None,
    exact_optimum_bitstring: Optional[str] = None,
) -> Dict[str, Any]:
    st = counts_qubo_stats(counts, instance, explicit_telegate=explicit_telegate)
    elite = build_elite_metrics_from_counts(
        counts=counts,
        instance=instance,
        explicit_telegate=explicit_telegate,
        top_k=analysis_cfg.elite_top_k,
    )
    p_close_cost = low_cost_mass_from_distribution(
        distribution=st["distribution"],
        instance=instance,
        base_cost=st["best_cost_observed"],
        delta=obj_cfg.delta_close_cost,
    )
    p_exact_cost = exact_cost_mass_from_distribution(
        distribution=st["distribution"],
        instance=instance,
        exact_optimum_cost=exact_optimum_cost,
    )
    p_exact_bitstring = exact_bitstring_mass_from_distribution(
        distribution=st["distribution"],
        exact_optimum_bitstring=exact_optimum_bitstring,
    )
    return {
        "best_cost_observed": float(st["best_cost_observed"]),
        "best_bitstring": st["best_bs_qubit"],
        "incumbent_cost": elite["incumbent_cost"],
        "incumbent_bitstring": elite["incumbent_bitstring"],
        "incumbent_count": elite["incumbent_count"],
        "p_incumbent_bitstring": elite["p_incumbent_bitstring"],
        "p_incumbent_cost": elite["p_incumbent_cost"],
        "mean_cost": float(st["mean_cost"]),
        "p_best_observed": float(st["p_best_observed"]),
        "p_close_cost": float(p_close_cost),
        "p_exact_cost": p_exact_cost,
        "p_exact_bitstring": p_exact_bitstring,
        "objective_value": float(st["mean_cost"]),
        "distribution_gap": float(st["mean_cost"] - st["best_cost_observed"]),
        "distribution": st["distribution"],
        "elite_top_k": elite["elite_top_k"],
        "elite_rows_top_k": elite["elite_rows_top_k"],
        "elite_mass_top_k": elite["elite_mass_top_k"],
        "elite_mean_cost_top_k": elite["elite_mean_cost_top_k"],
        "num_unique_sampled": elite["num_unique_sampled"],
    }


def evaluate_training_stability(
    history: List[Dict[str, Any]],
    cfg: ConvergenceConfig,
) -> Dict[str, Any]:
    if not history:
        return {
            "stable_window": 0,
            "last_obj_delta": None,
            "last_mean_cost_delta": None,
            "last_p_best_delta": None,
            "last_p_close_delta": None,
            "stable_over_window": False,
        }
    if len(history) == 1:
        return {
            "stable_window": 1,
            "last_obj_delta": None,
            "last_mean_cost_delta": None,
            "last_p_best_delta": None,
            "last_p_close_delta": None,
            "stable_over_window": False,
        }
    lookback = min(cfg.stable_window, len(history) - 1)
    last = history[-1]
    ref = history[-1 - lookback]
    last_obj_delta = abs(float(last["obj"]) - float(ref["obj"]))
    last_mean_cost_delta = abs(
        float(last["info"].get("mean_cost", float("nan")))
        - float(ref["info"].get("mean_cost", float("nan")))
    )
    last_p_best_delta = abs(
        float(last["info"].get("p_best_observed", float("nan")))
        - float(ref["info"].get("p_best_observed", float("nan")))
    )
    last_p_close_delta = abs(
        float(last["info"].get("p_close_cost", float("nan")))
        - float(ref["info"].get("p_close_cost", float("nan")))
    )
    stable_over_window = (
        last_obj_delta <= cfg.obj_tol
        and last_mean_cost_delta <= cfg.mean_cost_tol
        and last_p_best_delta <= cfg.p_best_tol
        and last_p_close_delta <= cfg.p_close_tol
    )
    return {
        "stable_window": int(lookback),
        "last_obj_delta": float(last_obj_delta),
        "last_mean_cost_delta": float(last_mean_cost_delta),
        "last_p_best_delta": float(last_p_best_delta),
        "last_p_close_delta": float(last_p_close_delta),
        "stable_over_window": bool(stable_over_window),
    }


def build_convergence_report(
    final_metrics: Dict[str, Any],
    stability: Dict[str, Any],
    cfg: ConvergenceConfig,
    certificate_lower_bound: Optional[float] = None,
    certificate_tol: float = 1e-9,
    exact_optimum_cost: Optional[float] = None,
    exact_optimum_bitstring: Optional[str] = None,
) -> Dict[str, Any]:
    best_cost = final_metrics.get("best_cost_observed")
    best_bitstring = final_metrics.get("best_bitstring")
    distribution_gap = final_metrics.get("distribution_gap")
    cost_match_exact = None
    optimality_gap_to_exact = None
    if exact_optimum_cost is not None and best_cost is not None:
        optimality_gap_to_exact = float(best_cost - exact_optimum_cost)
        cost_match_exact = bool(abs(optimality_gap_to_exact) <= 1e-9)
    bitstring_match_exact = None
    if exact_optimum_bitstring is not None and best_bitstring is not None:
        bitstring_match_exact = bool(best_bitstring == exact_optimum_bitstring)
    certificate_gap = None
    if certificate_lower_bound is not None and best_cost is not None:
        certificate_gap = float(best_cost - certificate_lower_bound)
    practical_converged = bool(
        stability.get("stable_over_window", False)
        and distribution_gap is not None
        and float(distribution_gap) <= cfg.gap_tol
    )
    certified_optimal = None
    if certificate_gap is not None:
        certified_optimal = bool(certificate_gap <= certificate_tol)
    return {
        "exact_optimum_cost": exact_optimum_cost,
        "exact_optimum_bitstring": exact_optimum_bitstring,
        "optimality_gap_to_exact": optimality_gap_to_exact,
        "cost_match_exact": cost_match_exact,
        "bitstring_match_exact": bitstring_match_exact,
        "certificate_lower_bound": certificate_lower_bound,
        "certificate_gap": certificate_gap,
        "distribution_gap": distribution_gap,
        "stable_over_window": stability.get("stable_over_window"),
        "practical_converged": practical_converged,
        "certified_optimal": certified_optimal,
    }


# =============================================================================
# Objective and optimizer
# =============================================================================
class QuboObjectiveEvaluator:
    def __init__(
        self,
        instance: QUBOInstance,
        p: int,
        allocation: Dict[int, Tuple[int, int]],
        train_cfg: TrainConfig,
        obj_cfg: QuboObjectiveConfig,
        remote_mode: str = "abstract",
    ):
        self.instance = instance
        self.p = p
        self.allocation = allocation
        self.train_cfg = train_cfg
        self.obj_cfg = obj_cfg
        self.remote_mode = remote_mode
        self.explicit_telegate = (remote_mode == "telegate_explicit")

    def _get_runner(self) -> CachedQuboCircuitRunner:
        return get_cached_qubo_runner(
            instance=self.instance,
            p=self.p,
            allocation=self.allocation,
            remote_mode=self.remote_mode,
            train_cfg=self.train_cfg,
            measure=True,
        )

    def _stats_from_counts(self, counts: Dict[str, int], runner: CachedQuboCircuitRunner) -> Dict[str, Any]:
        st = counts_qubo_stats(counts, self.instance, explicit_telegate=self.explicit_telegate)
        st["p_close_cost"] = low_cost_mass_from_distribution(
            distribution=st["distribution"],
            instance=self.instance,
            base_cost=st["best_cost_observed"],
            delta=self.obj_cfg.delta_close_cost,
        )
        st["dist_stats"] = dict(runner.static_stats)
        return st

    def evaluate_many(self, xs: List[np.ndarray]) -> List[Tuple[float, Dict[str, Any]]]:
        if len(xs) == 0:
            return []
        xs_clip = [clip_angles(np.asarray(x, dtype=float)) for x in xs]
        per_point_mean_costs = [[] for _ in xs_clip]
        per_point_p_best = [[] for _ in xs_clip]
        per_point_p_close = [[] for _ in xs_clip]
        last_stats = [None for _ in xs_clip]

        runner = self._get_runner()
        gammas_list = [list(x[0::2]) for x in xs_clip]
        betas_list = [list(x[1::2]) for x in xs_clip]

        for k in range(self.train_cfg.avg_k):
            seed_sim = int(self.train_cfg.seed_base + 1000 * k)
            if self.train_cfg.batch_evaluations and len(xs_clip) > 1:
                batch_counts = runner.run_counts_batch(
                    gammas_list=gammas_list,
                    betas_list=betas_list,
                    shots=self.train_cfg.shots_train,
                    seed_sim=seed_sim,
                )
            else:
                batch_counts = [
                    runner.run_counts(
                        gammas=gammas,
                        betas=betas,
                        shots=self.train_cfg.shots_train,
                        seed_sim=seed_sim,
                    )
                    for gammas, betas in zip(gammas_list, betas_list)
                ]

            for idx, counts in enumerate(batch_counts):
                st = self._stats_from_counts(counts, runner)
                per_point_mean_costs[idx].append(st["mean_cost"])
                per_point_p_best[idx].append(st["p_best_observed"])
                per_point_p_close[idx].append(st["p_close_cost"])
                last_stats[idx] = st["dist_stats"]

        out: List[Tuple[float, Dict[str, Any]]] = []
        for idx in range(len(xs_clip)):
            mean_cost = float(np.mean(per_point_mean_costs[idx]))
            p_best = float(np.mean(per_point_p_best[idx]))
            p_close = float(np.mean(per_point_p_close[idx]))
            info = {
                "mean_cost": mean_cost,
                "p_close_cost": p_close,
                "p_best_observed": p_best,
                "dist_stats": last_stats[idx],
            }
            out.append((float(mean_cost), info))
        return out

    def __call__(self, x: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        return self.evaluate_many([x])[0]


def build_objective_fn_qubo(
    instance: QUBOInstance,
    p: int,
    allocation: Dict[int, Tuple[int, int]],
    train_cfg: TrainConfig,
    obj_cfg: QuboObjectiveConfig,
    remote_mode: str = "abstract",
) -> QuboObjectiveEvaluator:
    return QuboObjectiveEvaluator(
        instance=instance,
        p=p,
        allocation=allocation,
        train_cfg=train_cfg,
        obj_cfg=obj_cfg,
        remote_mode=remote_mode,
    )

def adam_optimize(
    x0: np.ndarray,
    f_obj: Callable[[np.ndarray], Tuple[float, Dict[str, Any]]],
    cfg: AdamConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
    """
    Adam-style optimizer using an SPSA gradient estimator.
    This version supports batched objective evaluation when the objective
    exposes an `evaluate_many` method.
    """
    rng = np.random.default_rng(cfg.seed)
    x = clip_angles(np.array(x0, dtype=float))
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    best_x = x.copy()
    best_val = float("inf")
    history: List[Dict[str, Any]] = []

    alpha = float(cfg.spsa_alpha)
    gamma = float(cfg.spsa_gamma)
    A = float(cfg.spsa_A)
    batch_eval = getattr(f_obj, "evaluate_many", None)

    for it in range(cfg.iters):
        ak = float(cfg.lr / ((it + 1.0 + A) ** alpha))
        ck = float(cfg.fd_eps0 / ((it + 1.0) ** gamma))
        delta = rng.choice(np.array([-1.0, 1.0]), size=len(x))

        x_plus = clip_angles(x + ck * delta)
        x_minus = clip_angles(x - ck * delta)
        if cfg.eval_at_each_iter:
            if callable(batch_eval):
                (obj_p, info_p), (obj_m, info_m), (obj, info) = batch_eval([x_plus, x_minus, x])
            else:
                obj_p, info_p = f_obj(x_plus)
                obj_m, info_m = f_obj(x_minus)
                obj, info = f_obj(x)
        else:
            if callable(batch_eval):
                (obj_p, info_p), (obj_m, info_m) = batch_eval([x_plus, x_minus])
            else:
                obj_p, info_p = f_obj(x_plus)
                obj_m, info_m = f_obj(x_minus)
            obj = float(0.5 * (obj_p + obj_m))
            info = {
                "mean_cost": float(0.5 * (info_p.get("mean_cost", obj_p) + info_m.get("mean_cost", obj_m))),
                "p_close_cost": float(0.5 * (info_p.get("p_close_cost", 0.0) + info_m.get("p_close_cost", 0.0))),
                "p_best_observed": float(0.5 * (info_p.get("p_best_observed", 0.0) + info_m.get("p_best_observed", 0.0))),
                "dist_stats": info_p.get("dist_stats", info_m.get("dist_stats")),
            }

        grad = ((obj_p - obj_m) / (2.0 * ck)) * delta

        if obj < best_val:
            best_val = float(obj)
            best_x = x.copy()

        history.append({
            "iter": int(it),
            "obj": float(obj),
            "x": x.copy(),
            "info": info,
            "obj_plus": float(obj_p),
            "obj_minus": float(obj_m),
            "ak": ak,
            "ck": ck,
        })
        if verbose:
            print(
                f"iter={it:03d} | obj={obj:.8f} | "
                f"mean_cost={info.get('mean_cost', float('nan')):.8f} | "
                f"p_best={info.get('p_best_observed', float('nan')):.6f}",
                flush=True,
            )

        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * (grad ** 2)
        mhat = m / (1.0 - cfg.beta1 ** (it + 1))
        vhat = v / (1.0 - cfg.beta2 ** (it + 1))
        x = clip_angles(x - ak * mhat / (np.sqrt(vhat) + cfg.eps))

        if it < cfg.iters - 1 and np.allclose(grad, 0.0):
            x = clip_angles(x + 1e-4 * rng.standard_normal(size=len(x)))
    return best_x, float(best_val), history



# =============================================================================
# Multi-start helpers
# =============================================================================
def lift_angles_to_next_depth(prev_best_x: np.ndarray, new_p: int) -> np.ndarray:
    prev_best_x = np.asarray(prev_best_x, dtype=float).ravel()
    prev_p = len(prev_best_x) // 2
    if new_p < prev_p:
        raise ValueError("new_p must be >= previous depth")
    if new_p == prev_p:
        return clip_angles(prev_best_x.copy())
    gammas_prev = prev_best_x[0::2]
    betas_prev = prev_best_x[1::2]
    gammas_new = np.zeros(new_p, dtype=float)
    betas_new = np.zeros(new_p, dtype=float)
    gammas_new[:prev_p] = gammas_prev
    betas_new[:prev_p] = betas_prev
    fill_gamma = gammas_prev[-1] if prev_p > 0 else 0.25 * np.pi
    fill_beta = betas_prev[-1] if prev_p > 0 else 0.25 * np.pi
    gammas_new[prev_p:] = fill_gamma
    betas_new[prev_p:] = fill_beta
    x = np.zeros(2 * new_p, dtype=float)
    x[0::2] = gammas_new
    x[1::2] = betas_new
    return clip_angles(x)


def generate_initial_points_for_depth(
    p: int,
    base_seed: int,
    ms_cfg: MultiStartConfig,
    prev_best_x: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    rng = np.random.default_rng(base_seed)
    initials: List[np.ndarray] = []
    if prev_best_x is not None:
        warm = lift_angles_to_next_depth(prev_best_x, p)
        if ms_cfg.add_plain_warm_start:
            initials.append(warm.copy())
        for _ in range(ms_cfg.warm_start_perturbations):
            initials.append(clip_angles(warm + ms_cfg.warm_start_sigma * rng.standard_normal(size=len(warm))))
    for _ in range(ms_cfg.num_random_starts):
        initials.append(rng.random(2 * p) * np.pi)
    return initials


def candidate_rank_tuple(
    result: Dict[str, Any],
    exact_available: bool,
    prefer_exact: bool,
) -> Tuple:
    if exact_available and prefer_exact:
        return (
            0 if result.get("cost_match_exact") else 1,
            0 if result.get("bitstring_match_exact") else 1,
            -float(result.get("final_p_exact_cost") or 0.0),
            -float(result.get("final_p_exact_bitstring") or 0.0),
            float(result.get("best_cost") if result.get("best_cost") is not None else float("inf")),
            -float(result.get("final_p_incumbent_cost") or 0.0),
            -float(result.get("final_p_incumbent_bitstring") or 0.0),
            float(result.get("mean_cost") if result.get("mean_cost") is not None else float("inf")),
            float(result.get("final_objective_value") if result.get("final_objective_value") is not None else float("inf")),
        )
    return (
        float(result.get("best_cost") if result.get("best_cost") is not None else float("inf")),
        -float(result.get("final_p_incumbent_cost") or 0.0),
        -float(result.get("final_p_incumbent_bitstring") or 0.0),
        float(result.get("mean_cost") if result.get("mean_cost") is not None else float("inf")),
        float(result.get("distribution_gap") if result.get("distribution_gap") is not None else float("inf")),
        float(result.get("final_objective_value") if result.get("final_objective_value") is not None else float("inf")),
    )


def choose_best_result(
    results: List[Dict[str, Any]],
    exact_available: bool,
    prefer_exact: bool,
) -> Dict[str, Any]:
    if not results:
        raise ValueError("No results were provided to choose_best_result.")
    return sorted(
        results,
        key=lambda r: candidate_rank_tuple(r, exact_available=exact_available, prefer_exact=prefer_exact)
    )[0]


# =============================================================================
# Consensus helpers
# =============================================================================
def compute_restart_consensus(
    candidates: List[Dict[str, Any]],
    selected_cost: Optional[float],
    selected_bitstring: Optional[str],
    cost_tol: float = 1e-12,
) -> Dict[str, Optional[float]]:
    if not candidates:
        return {
            "restart_consensus_cost": None,
            "restart_consensus_bitstring": None,
        }
    denom = float(len(candidates))
    cost_hits = 0
    bs_hits = 0
    for cand in candidates:
        cand_cost = cand.get("best_cost")
        cand_bs = cand.get("best_bitstring")
        if selected_cost is not None and cand_cost is not None and abs(float(cand_cost) - float(selected_cost)) <= cost_tol:
            cost_hits += 1
        if selected_bitstring is not None and cand_bs == selected_bitstring:
            bs_hits += 1
    return {
        "restart_consensus_cost": float(cost_hits / denom),
        "restart_consensus_bitstring": float(bs_hits / denom),
    }


def annotate_mode_consensus(
    results: List[Dict[str, Any]],
    cost_tol: float = 1e-12,
) -> List[Dict[str, Any]]:
    qaoa_modes = {"monolithic_qaoa", "abstract_distributed_qaoa", "telegate_explicit_qaoa"}
    qaoa_results = [r for r in results if r.get("mode") in qaoa_modes and r.get("best_cost") is not None]
    denom = float(len(qaoa_results)) if qaoa_results else 0.0
    for r in results:
        if denom <= 0.0 or r.get("best_cost") is None:
            r["mode_consensus_cost"] = None
            r["mode_consensus_bitstring"] = None
            continue
        same_cost = 0
        same_bs = 0
        for q in qaoa_results:
            if abs(float(q.get("best_cost")) - float(r.get("best_cost"))) <= cost_tol:
                same_cost += 1
            if q.get("best_bitstring") == r.get("best_bitstring"):
                same_bs += 1
        r["mode_consensus_cost"] = float(same_cost / denom)
        r["mode_consensus_bitstring"] = float(same_bs / denom)
    return results


# =============================================================================
# QAOA solve helpers
# =============================================================================
def evaluate_qaoa_candidate(
    instance: QUBOInstance,
    p: int,
    x_angles: np.ndarray,
    allocation: Dict[int, Tuple[int, int]],
    remote_mode: str,
    train_cfg: TrainConfig,
    obj_cfg: QuboObjectiveConfig,
    analysis_cfg: AnalysisConfig,
    final_shots: int,
    certificate_lower_bound: Optional[float],
    exact_optimum_cost: Optional[float],
    exact_optimum_bitstring: Optional[str],
    convergence_cfg: ConvergenceConfig,
    history: List[Dict[str, Any]],
    train_objective_best: float,
    selected_allocation_mode: str,
    num_qpus_eff: int,
    capacities_eff: List[int],
    return_full_output: bool,
    store_history: bool,
    store_final_counts: bool,
    restart_index: int,
    start_type: str,
) -> Dict[str, Any]:
    explicit_telegate = (remote_mode == "telegate_explicit")
    gammas = list(clip_angles(x_angles)[0::2])
    betas = list(clip_angles(x_angles)[1::2])
    runner = get_cached_qubo_runner(
        instance=instance,
        p=p,
        allocation=allocation,
        remote_mode=remote_mode,
        train_cfg=train_cfg,
        measure=True,
    )
    counts_fin = runner.run_counts(
        gammas=gammas,
        betas=betas,
        shots=final_shots,
        seed_sim=9999 + 101 * p + 17 * restart_index,
    )
    final_metrics = compute_final_distribution_metrics(
        counts=counts_fin,
        instance=instance,
        obj_cfg=obj_cfg,
        analysis_cfg=analysis_cfg,
        explicit_telegate=explicit_telegate,
        exact_optimum_cost=exact_optimum_cost,
        exact_optimum_bitstring=exact_optimum_bitstring,
    )
    stability = evaluate_training_stability(history, convergence_cfg)
    convergence_report = build_convergence_report(
        final_metrics=final_metrics,
        stability=stability,
        cfg=convergence_cfg,
        certificate_lower_bound=certificate_lower_bound,
        certificate_tol=analysis_cfg.certificate_tol,
        exact_optimum_cost=exact_optimum_cost,
        exact_optimum_bitstring=exact_optimum_bitstring,
    )
    details = {
        "selected_allocation_mode": selected_allocation_mode,
        "assignment": allocation_to_assignment(instance.n, allocation),
        "num_qpus": int(num_qpus_eff),
        "capacities": [int(v) for v in capacities_eff],
        "remote_mode": remote_mode,
        "train_objective_best": float(train_objective_best),
        "objective_formula": "mean_cost",
        "batch_evaluations": bool(train_cfg.batch_evaluations),
        "delta_close_cost": float(obj_cfg.delta_close_cost),
        "elite_top_k": int(analysis_cfg.elite_top_k),
        "certificate_tol": float(analysis_cfg.certificate_tol),
        "stable_window": int(stability["stable_window"]),
        "last_obj_delta": stability["last_obj_delta"],
        "last_mean_cost_delta": stability["last_mean_cost_delta"],
        "last_p_best_delta": stability["last_p_best_delta"],
        "last_p_close_delta": stability["last_p_close_delta"],
        "restart_index": int(restart_index),
        "start_type": start_type,
    }
    if return_full_output:
        details["allocation"] = allocation
    if store_history:
        details["history"] = history
    if store_final_counts:
        details["final_counts"] = counts_fin
    return {
        "status": "finished",
        "p": int(p),
        "angles": _to_py_float_list(x_angles),
        "gammas": [float(v) for v in gammas],
        "betas": [float(v) for v in betas],
        "best_cost": float(final_metrics["best_cost_observed"]),
        "best_bitstring": final_metrics["best_bitstring"],
        "incumbent_cost": final_metrics["incumbent_cost"],
        "incumbent_bitstring": final_metrics["incumbent_bitstring"],
        "incumbent_count": int(final_metrics["incumbent_count"]),
        "mean_cost": float(final_metrics["mean_cost"]),
        "final_objective_value": float(final_metrics["objective_value"]),
        "final_p_best_observed": float(final_metrics["p_best_observed"]),
        "final_p_close_cost": float(final_metrics["p_close_cost"]),
        "final_p_exact_cost": final_metrics["p_exact_cost"],
        "final_p_exact_bitstring": final_metrics["p_exact_bitstring"],
        "final_p_incumbent_bitstring": float(final_metrics["p_incumbent_bitstring"]),
        "final_p_incumbent_cost": float(final_metrics["p_incumbent_cost"]),
        "elite_top_k": int(final_metrics["elite_top_k"]),
        "elite_rows_top_k": final_metrics["elite_rows_top_k"],
        "elite_mass_top_k": float(final_metrics["elite_mass_top_k"]),
        "elite_mean_cost_top_k": final_metrics["elite_mean_cost_top_k"],
        "num_unique_sampled": int(final_metrics["num_unique_sampled"]),
        "distribution_gap": float(final_metrics["distribution_gap"]),
        "exact_optimum_cost": convergence_report["exact_optimum_cost"],
        "optimality_gap_to_exact": convergence_report["optimality_gap_to_exact"],
        "cost_match_exact": convergence_report["cost_match_exact"],
        "bitstring_match_exact": convergence_report["bitstring_match_exact"],
        "certificate_lower_bound": convergence_report["certificate_lower_bound"],
        "certificate_gap": convergence_report["certificate_gap"],
        "stable_over_window": convergence_report["stable_over_window"],
        "practical_converged": convergence_report["practical_converged"],
        "certified_optimal": convergence_report["certified_optimal"],
        "runtime_to_solution_seconds": None,
        "remote_cx_count": int(runner.static_stats["remote_cx"]),
        "cross_qpu_terms": int(runner.static_stats["cross_qpu_terms"]),
        "details": details,
    }


def run_single_qaoa_mode(
    instance: QUBOInstance,
    mode: str,
    p_max: int = 1,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    train_cfg: Optional[TrainConfig] = None,
    obj_cfg: Optional[QuboObjectiveConfig] = None,
    analysis_cfg: Optional[AnalysisConfig] = None,
    final_shots: int = 512,
    adam_cfg: Optional[AdamConfig] = None,
    multistart_cfg: Optional[MultiStartConfig] = None,
    verbose: bool = False,
    return_full_output: bool = False,
    store_history: bool = False,
    store_final_counts: bool = False,
    certificate_lower_bound: Optional[float] = None,
    exact_optimum_cost: Optional[float] = None,
    exact_optimum_bitstring: Optional[str] = None,
    convergence_cfg: Optional[ConvergenceConfig] = None,
) -> Dict[str, Any]:
    require_qiskit()
    if mode not in {
        "monolithic_qaoa",
        "abstract_distributed_qaoa",
        "telegate_explicit_qaoa",
    }:
        raise ValueError(f"Unsupported QAOA mode: {mode}")
    if train_cfg is None:
        train_cfg = TrainConfig()
    if obj_cfg is None:
        obj_cfg = QuboObjectiveConfig()
    if analysis_cfg is None:
        analysis_cfg = AnalysisConfig()
    if adam_cfg is None:
        adam_cfg = AdamConfig()
    if multistart_cfg is None:
        multistart_cfg = MultiStartConfig()
    if convergence_cfg is None:
        convergence_cfg = ConvergenceConfig()
    n = instance.n
    if mode == "monolithic_qaoa":
        num_qpus_eff = 1
        capacities_eff = [n]
        allocation = make_allocation(n, 1, capacities_eff, min_used_qpus=1)
        selected_mode = "monolithic"
        remote_mode = "abstract"
    else:
        if num_qpus < 2:
            raise ValueError("Distributed QAOA modes require num_qpus >= 2.")
        if capacities is None:
            capacities = make_balanced_capacities(n, num_qpus)
        candidates = build_allocation_candidates_qubo(
            instance=instance,
            num_qpus=num_qpus,
            capacities=capacities,
            p_eval=p_max,
            manual_assignment=manual_assignment,
            min_used_qpus=2,
        )
        best_candidate = select_best_allocation_candidate(candidates)
        allocation = best_candidate["allocation"]
        selected_mode = best_candidate["mode"]
        num_qpus_eff = num_qpus
        capacities_eff = capacities
        remote_mode = "abstract" if mode == "abstract_distributed_qaoa" else "telegate_explicit"
    t0 = time.perf_counter()
    exact_available = (exact_optimum_cost is not None)
    prefer_exact = bool(multistart_cfg.select_by_exact_when_available)
    prev_depth_best_x: Optional[np.ndarray] = None
    best_overall: Optional[Dict[str, Any]] = None
    depth_summaries: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    selected_depth_candidates: List[Dict[str, Any]] = []
    for p in range(1, p_max + 1):
        initials = generate_initial_points_for_depth(
            p=p,
            base_seed=adam_cfg.seed + 10_000 * p + 97,
            ms_cfg=multistart_cfg,
            prev_best_x=prev_depth_best_x,
        )
        f_obj = build_objective_fn_qubo(
            instance=instance,
            p=p,
            allocation=allocation,
            train_cfg=train_cfg,
            obj_cfg=obj_cfg,
            remote_mode=remote_mode,
        )
        depth_candidates: List[Dict[str, Any]] = []

        def _run_restart(restart_index: int, x0: np.ndarray) -> Dict[str, Any]:
            adam_local = AdamConfig(
                iters=adam_cfg.iters,
                lr=adam_cfg.lr,
                beta1=adam_cfg.beta1,
                beta2=adam_cfg.beta2,
                eps=adam_cfg.eps,
                fd_eps0=adam_cfg.fd_eps0,
                seed=adam_cfg.seed + 1000 * p + restart_index,
                spsa_alpha=adam_cfg.spsa_alpha,
                spsa_gamma=adam_cfg.spsa_gamma,
                spsa_A=adam_cfg.spsa_A,
                eval_at_each_iter=adam_cfg.eval_at_each_iter,
            )
            best_x, best_val, history = adam_optimize(
                x0=x0,
                f_obj=f_obj,
                cfg=adam_local,
                verbose=verbose,
            )
            start_type = "warm" if (
                prev_depth_best_x is not None and restart_index < (multistart_cfg.warm_start_perturbations + int(multistart_cfg.add_plain_warm_start))
            ) else "random"
            candidate = evaluate_qaoa_candidate(
                instance=instance,
                p=p,
                x_angles=best_x,
                allocation=allocation,
                remote_mode=remote_mode,
                train_cfg=train_cfg,
                obj_cfg=obj_cfg,
                analysis_cfg=analysis_cfg,
                final_shots=final_shots,
                certificate_lower_bound=certificate_lower_bound,
                exact_optimum_cost=exact_optimum_cost,
                exact_optimum_bitstring=exact_optimum_bitstring,
                convergence_cfg=convergence_cfg,
                history=history,
                train_objective_best=best_val,
                selected_allocation_mode=selected_mode,
                num_qpus_eff=num_qpus_eff,
                capacities_eff=capacities_eff,
                return_full_output=return_full_output,
                store_history=store_history,
                store_final_counts=store_final_counts,
                restart_index=restart_index,
                start_type=start_type,
            )
            candidate["mode"] = mode
            return candidate

        max_workers = max(1, int(getattr(multistart_cfg, "parallel_restarts", 1)))
        if max_workers > 1 and len(initials) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(initials))) as executor:
                futures = {executor.submit(_run_restart, restart_index, x0): restart_index for restart_index, x0 in enumerate(initials)}
                ordered: Dict[int, Dict[str, Any]] = {}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    ordered[idx] = fut.result()
                for idx in sorted(ordered):
                    candidate = ordered[idx]
                    depth_candidates.append(candidate)
                    all_candidates.append(candidate)
        else:
            for restart_index, x0 in enumerate(initials):
                candidate = _run_restart(restart_index, x0)
                depth_candidates.append(candidate)
                all_candidates.append(candidate)
        depth_best = choose_best_result(
            depth_candidates,
            exact_available=exact_available,
            prefer_exact=prefer_exact,
        )
        prev_depth_best_x = np.asarray(depth_best["angles"], dtype=float)
        depth_summary = {
            "p": int(p),
            "num_restarts": len(depth_candidates),
            "best_cost": depth_best.get("best_cost"),
            "best_bitstring": depth_best.get("best_bitstring"),
            "incumbent_cost": depth_best.get("incumbent_cost"),
            "incumbent_bitstring": depth_best.get("incumbent_bitstring"),
            "mean_cost": depth_best.get("mean_cost"),
            "p_best": depth_best.get("final_p_best_observed"),
            "p_incumbent_cost": depth_best.get("final_p_incumbent_cost"),
            "elite_mass_top_k": depth_best.get("elite_mass_top_k"),
            "cost_match_exact": depth_best.get("cost_match_exact"),
            "bitstring_match_exact": depth_best.get("bitstring_match_exact"),
        }
        if multistart_cfg.keep_top_k_per_depth > 0:
            ranked = sorted(
                depth_candidates,
                key=lambda r: candidate_rank_tuple(r, exact_available=exact_available, prefer_exact=prefer_exact)
            )
            depth_summary["top_candidates"] = [
                {
                    "restart_index": r["details"].get("restart_index"),
                    "start_type": r["details"].get("start_type"),
                    "best_cost": r.get("best_cost"),
                    "best_bitstring": r.get("best_bitstring"),
                    "incumbent_cost": r.get("incumbent_cost"),
                    "incumbent_bitstring": r.get("incumbent_bitstring"),
                    "mean_cost": r.get("mean_cost"),
                    "p_best": r.get("final_p_best_observed"),
                    "p_incumbent_cost": r.get("final_p_incumbent_cost"),
                    "elite_mass_top_k": r.get("elite_mass_top_k"),
                }
                for r in ranked[:multistart_cfg.keep_top_k_per_depth]
            ]
        depth_summaries.append(depth_summary)
        if best_overall is None:
            best_overall = depth_best
            selected_depth_candidates = depth_candidates
        else:
            new_best = choose_best_result(
                [best_overall, depth_best],
                exact_available=exact_available,
                prefer_exact=prefer_exact,
            )
            if new_best is depth_best:
                selected_depth_candidates = depth_candidates
            best_overall = new_best
    elapsed = time.perf_counter() - t0
    assert best_overall is not None
    best_overall["runtime_to_solution_seconds"] = float(elapsed)
    selected_consensus = compute_restart_consensus(
        selected_depth_candidates,
        selected_cost=best_overall.get("best_cost"),
        selected_bitstring=best_overall.get("best_bitstring"),
    )
    overall_consensus = compute_restart_consensus(
        all_candidates,
        selected_cost=best_overall.get("best_cost"),
        selected_bitstring=best_overall.get("best_bitstring"),
    )
    best_overall["restart_consensus_cost"] = selected_consensus["restart_consensus_cost"]
    best_overall["restart_consensus_bitstring"] = selected_consensus["restart_consensus_bitstring"]
    best_overall.setdefault("details", {})
    best_overall["details"]["depth_summaries"] = depth_summaries
    best_overall["details"]["multistart"] = {
        "num_random_starts": multistart_cfg.num_random_starts,
        "warm_start_perturbations": multistart_cfg.warm_start_perturbations,
        "warm_start_sigma": multistart_cfg.warm_start_sigma,
        "add_plain_warm_start": multistart_cfg.add_plain_warm_start,
        "selection_uses_exact_when_available": multistart_cfg.select_by_exact_when_available,
        "parallel_restarts": multistart_cfg.parallel_restarts,
    }
    best_overall["details"]["analysis"] = {
        "elite_top_k": analysis_cfg.elite_top_k,
        "certificate_tol": analysis_cfg.certificate_tol,
    }
    best_overall["details"]["restart_consensus_overall_cost"] = overall_consensus["restart_consensus_cost"]
    best_overall["details"]["restart_consensus_overall_bitstring"] = overall_consensus["restart_consensus_bitstring"]
    return best_overall



# =============================================================================
# Public mode-dispatch API
# =============================================================================
VALID_MODES = {
    "bruteforce",
    "miqp_cplex",
    "monolithic_qaoa",
    "abstract_distributed_qaoa",
    "telegate_explicit_qaoa",
}


def solve_qubo_mode(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    mode: str = "abstract_distributed_qaoa",
    p_max: int = 1,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    train_cfg: Optional[TrainConfig] = None,
    obj_cfg: Optional[QuboObjectiveConfig] = None,
    analysis_cfg: Optional[AnalysisConfig] = None,
    final_shots: int = 512,
    adam_cfg: Optional[AdamConfig] = None,
    multistart_cfg: Optional[MultiStartConfig] = None,
    cplex_time_limit: Optional[float] = None,
    verbose: bool = False,
    return_full_output: bool = False,
    store_history: bool = False,
    store_final_counts: bool = False,
    compute_exact_if_small: bool = True,
    exact_threshold_n: int = 16,
    certificate_lower_bound: Optional[float] = None,
    convergence_cfg: Optional[ConvergenceConfig] = None,
    _instance: Optional[QUBOInstance] = None,
    _precomputed_exact_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Valid modes: {sorted(VALID_MODES)}")
    if analysis_cfg is None:
        analysis_cfg = AnalysisConfig()
    instance = _instance if _instance is not None else canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    exact_optimum_cost = None
    exact_optimum_bitstring = None
    exact_res = _precomputed_exact_result
    if exact_res is None and compute_exact_if_small and instance.n <= exact_threshold_n:
        exact_res = solve_qubo_bruteforce(instance)
    if exact_res is not None:
        exact_optimum_cost = exact_res["best_cost"]
        exact_optimum_bitstring = exact_res["best_bitstring"]
    certificate_lb_eff = exact_optimum_cost if exact_optimum_cost is not None else certificate_lower_bound
    if mode == "bruteforce":
        res = solve_qubo_bruteforce(instance)
        res.update({
            "incumbent_cost": float(res["best_cost"]),
            "incumbent_bitstring": res["best_bitstring"],
            "incumbent_count": 1,
            "final_objective_value": float(res["best_cost"]),
            "final_p_best_observed": 1.0,
            "final_p_close_cost": 1.0,
            "final_p_exact_cost": 1.0,
            "final_p_exact_bitstring": 1.0,
            "final_p_incumbent_bitstring": 1.0,
            "final_p_incumbent_cost": 1.0,
            "elite_top_k": 1,
            "elite_rows_top_k": [{"bitstring": res["best_bitstring"], "cost": float(res["best_cost"]), "count": 1, "frequency": 1.0}],
            "elite_mass_top_k": 1.0,
            "elite_mean_cost_top_k": float(res["best_cost"]),
            "num_unique_sampled": 1,
            "distribution_gap": 0.0,
            "exact_optimum_cost": exact_optimum_cost,
            "optimality_gap_to_exact": 0.0 if exact_optimum_cost is not None else None,
            "cost_match_exact": True if exact_optimum_cost is not None else None,
            "bitstring_match_exact": (res["best_bitstring"] == exact_optimum_bitstring) if exact_optimum_bitstring is not None else None,
            "certificate_lower_bound": certificate_lb_eff,
            "certificate_gap": (float(res["best_cost"] - certificate_lb_eff) if certificate_lb_eff is not None else None),
            "stable_over_window": None,
            "practical_converged": True,
            "certified_optimal": (True if certificate_lb_eff is not None and abs(float(res["best_cost"] - certificate_lb_eff)) <= analysis_cfg.certificate_tol else None),
            "restart_consensus_cost": None,
            "restart_consensus_bitstring": None,
        })
        return res
    if mode == "miqp_cplex":
        res = solve_qubo_miqp_cplex(instance, time_limit=cplex_time_limit)
        if res.get("best_cost") is not None:
            res["incumbent_cost"] = float(res["best_cost"])
            res["incumbent_bitstring"] = res.get("best_bitstring")
            res["final_p_incumbent_bitstring"] = None
            res["final_p_incumbent_cost"] = None
            res["elite_top_k"] = None
            res["elite_rows_top_k"] = None
            res["elite_mass_top_k"] = None
            res["elite_mean_cost_top_k"] = None
            res["num_unique_sampled"] = None
        if exact_optimum_cost is not None and res.get("best_cost") is not None:
            res["exact_optimum_cost"] = exact_optimum_cost
            res["optimality_gap_to_exact"] = float(res["best_cost"] - exact_optimum_cost)
            res["cost_match_exact"] = abs(res["optimality_gap_to_exact"]) <= 1e-9
            res["bitstring_match_exact"] = (res.get("best_bitstring") == exact_optimum_bitstring) if exact_optimum_bitstring is not None else None
        res["certificate_lower_bound"] = certificate_lb_eff
        res["certificate_gap"] = (
            float(res["best_cost"] - certificate_lb_eff)
            if certificate_lb_eff is not None and res.get("best_cost") is not None else None
        )
        res["certified_optimal"] = (
            bool(res["certificate_gap"] <= analysis_cfg.certificate_tol)
            if res.get("certificate_gap") is not None else None
        )
        return res
    return run_single_qaoa_mode(
        instance=instance,
        mode=mode,
        p_max=p_max,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
        train_cfg=train_cfg,
        obj_cfg=obj_cfg,
        analysis_cfg=analysis_cfg,
        final_shots=final_shots,
        adam_cfg=adam_cfg,
        multistart_cfg=multistart_cfg,
        verbose=verbose,
        return_full_output=return_full_output,
        store_history=store_history,
        store_final_counts=store_final_counts,
        certificate_lower_bound=certificate_lb_eff,
        exact_optimum_cost=exact_optimum_cost,
        exact_optimum_bitstring=exact_optimum_bitstring,
        convergence_cfg=convergence_cfg,
    )


def compare_qubo_modes(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    modes: Optional[List[str]] = None,
    p_max: int = 1,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    train_cfg: Optional[TrainConfig] = None,
    obj_cfg: Optional[QuboObjectiveConfig] = None,
    analysis_cfg: Optional[AnalysisConfig] = None,
    final_shots: int = 512,
    adam_cfg: Optional[AdamConfig] = None,
    multistart_cfg: Optional[MultiStartConfig] = None,
    cplex_time_limit: Optional[float] = None,
    verbose: bool = False,
    return_full_output: bool = False,
    store_history: bool = False,
    store_final_counts: bool = False,
    compute_exact_if_small: bool = True,
    exact_threshold_n: int = 16,
    certificate_lower_bound: Optional[float] = None,
    convergence_cfg: Optional[ConvergenceConfig] = None,
) -> List[Dict[str, Any]]:
    if modes is None:
        modes = [
            "bruteforce",
            "miqp_cplex",
            "monolithic_qaoa",
            "abstract_distributed_qaoa",
            "telegate_explicit_qaoa",
        ]
    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    exact_res = None
    if compute_exact_if_small and instance.n <= exact_threshold_n:
        exact_res = solve_qubo_bruteforce(instance)
    results = []
    for mode in modes:
        try:
            res = solve_qubo_mode(
                H=H,
                f=f,
                c0=c0,
                name=name,
                mode=mode,
                p_max=p_max,
                num_qpus=num_qpus,
                capacities=capacities,
                manual_assignment=manual_assignment,
                train_cfg=train_cfg,
                obj_cfg=obj_cfg,
                analysis_cfg=analysis_cfg,
                final_shots=final_shots,
                adam_cfg=adam_cfg,
                multistart_cfg=multistart_cfg,
                cplex_time_limit=cplex_time_limit,
                verbose=verbose,
                return_full_output=return_full_output,
                store_history=store_history,
                store_final_counts=store_final_counts,
                compute_exact_if_small=compute_exact_if_small,
                exact_threshold_n=exact_threshold_n,
                certificate_lower_bound=certificate_lower_bound,
                convergence_cfg=convergence_cfg,
                _instance=instance,
                _precomputed_exact_result=exact_res,
            )
        except Exception as e:
            res = {
                "mode": mode,
                "status": f"error: {type(e).__name__}",
                "best_cost": None,
                "best_bitstring": None,
                "incumbent_cost": None,
                "incumbent_bitstring": None,
                "mean_cost": None,
                "runtime_to_solution_seconds": None,
                "remote_cx_count": None,
                "cross_qpu_terms": None,
                "final_objective_value": None,
                "final_p_best_observed": None,
                "final_p_close_cost": None,
                "final_p_exact_cost": None,
                "final_p_exact_bitstring": None,
                "final_p_incumbent_bitstring": None,
                "final_p_incumbent_cost": None,
                "elite_top_k": None,
                "elite_rows_top_k": None,
                "elite_mass_top_k": None,
                "elite_mean_cost_top_k": None,
                "num_unique_sampled": None,
                "distribution_gap": None,
                "practical_converged": None,
                "stable_over_window": None,
                "exact_optimum_cost": None,
                "optimality_gap_to_exact": None,
                "cost_match_exact": None,
                "bitstring_match_exact": None,
                "certificate_lower_bound": certificate_lower_bound,
                "certificate_gap": None,
                "certified_optimal": None,
                "restart_consensus_cost": None,
                "restart_consensus_bitstring": None,
                "details": {"error_message": str(e)},
            }
        results.append(res)
    return annotate_mode_consensus(results)


def print_mode_comparison_table(results: List[Dict[str, Any]]):
    print("\n================ QUBO MODE COMPARISON ================")
    header = (
        f"{'mode':28s} | {'status':20s} | {'best_bitstring':14s} | "
        f"{'best_cost':12s} | {'mean_cost':12s} | {'gap':10s} | {'p_best':10s} | {'runtime_s':12s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{str(r.get('mode')):28s} | "
            f"{str(r.get('status')):20s} | "
            f"{str(r.get('best_bitstring')):14s} | "
            f"{str(r.get('best_cost')):12s} | "
            f"{str(r.get('mean_cost')):12s} | "
            f"{str(r.get('distribution_gap')):10s} | "
            f"{str(r.get('final_p_best_observed')):10s} | "
            f"{str(r.get('runtime_to_solution_seconds')):12s}"
        )
    print("======================================================\n")


def plot_top_k_elite_bitstrings(
    result: Dict[str, Any],
    top_k: Optional[int] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    use_frequency: bool = True,
    annotate_costs: bool = True,
    dpi: int = 150,
):
    require_matplotlib()
    elite_rows = result.get("elite_rows_top_k")
    if elite_rows is None:
        raise ValueError("This result does not contain elite_rows_top_k. Run a QAOA mode first.")
    if top_k is not None:
        elite_rows = elite_rows[:int(max(0, top_k))]
    if not elite_rows:
        raise ValueError("No elite bitstrings are available to plot.")
    labels = [str(r["bitstring"]) for r in elite_rows]
    values = [float(r["frequency"] if use_frequency else r["count"]) for r in elite_rows]
    costs = [float(r["cost"]) for r in elite_rows]
    width = max(8.0, 0.7 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5.5))
    bars = ax.bar(labels, values)
    ax.set_xlabel("Elite bitstrings")
    ax.set_ylabel("Frequency" if use_frequency else "Count")
    if title is None:
        k = len(elite_rows)
        title = f"Top-{k} elite bitstrings: {result.get('mode', 'qubo_mode')}"
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    if annotate_costs:
        ymax = max(values) if values else 1.0
        offset = 0.01 * ymax if ymax > 0 else 0.01
        for bar, cost in zip(bars, costs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                f"{cost:.6g}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig, ax



def plot_depth_progression(result, save_path=None):
    depth_summaries = result.get("details", {}).get("depth_summaries")
    if not depth_summaries:
        print(f"No depth summaries for mode={result.get('mode')}")
        return

    ps = [d["p"] for d in depth_summaries]
    best_costs = [d.get("best_cost") for d in depth_summaries]
    mean_costs = [d.get("mean_cost") for d in depth_summaries]
    p_best = [d.get("p_best") for d in depth_summaries]
    elite_mass = [d.get("elite_mass_top_k") for d in depth_summaries]

    plt.figure(figsize=(7, 4.5))
    plt.plot(ps, best_costs, marker="o", label="Best sampled cost")
    plt.plot(ps, mean_costs, marker="o", label="Mean sampled cost")
    plt.plot(ps, p_best, marker="o", label="P(best sampled cost)")
    plt.plot(ps, elite_mass, marker="o", label="Elite mass top-k")
    plt.xlabel("Depth p")
    plt.ylabel("Value")
    plt.title(f"Depth progression: {result.get('mode')}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()




def plot_mode_comparison(results, save_path=None):
    rows = [
        r for r in results
        if r.get("status") != None and r.get("best_cost") is not None
    ]
    modes = [r["mode"] for r in rows]
    best_costs = [r.get("best_cost") for r in rows]
    runtimes = [r.get("runtime_to_solution_seconds") for r in rows]
    p_best = [r.get("final_p_best_observed") if r.get("final_p_best_observed") is not None else 0.0 for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].bar(modes, best_costs)
    axes[0].set_title("Best sampled cost by mode")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(modes, runtimes)
    axes[1].set_title("Runtime to solution by mode")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(modes, p_best)
    axes[2].set_title("Probability of best sampled cost by mode")
    axes[2].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



import matplotlib.pyplot as plt

def plot_training_trace(result, save_path=None):
    history = result.get("details", {}).get("history")
    if not history:
        print(f"No stored history for mode={result.get('mode')}")
        return

    iters = [h["iter"] for h in history]
    obj = [h["obj"] for h in history]
    mean_cost = [h["info"].get("mean_cost") for h in history]
    p_best = [h["info"].get("p_best_observed") for h in history]
    p_close = [h["info"].get("p_close_cost") for h in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(iters, obj, label="Objective")
    plt.plot(iters, mean_cost, label="Mean cost")
    plt.plot(iters, p_best, label="P(best sampled cost)")
    plt.plot(iters, p_close, label="P(close cost)")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(f"Training trace: {result.get('mode')}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()      


# =============================================================================
# Validation helpers
# =============================================================================
def total_variation_distance(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def allocation_for_validation(
    instance: QUBOInstance,
    num_qpus: int,
    capacities: Optional[List[int]],
    manual_assignment: Optional[List[int]],
    p_eval: int,
) -> Dict[int, Tuple[int, int]]:
    if capacities is None:
        capacities = make_balanced_capacities(instance.n, num_qpus)
    candidates = build_allocation_candidates_qubo(
        instance=instance,
        num_qpus=num_qpus,
        capacities=capacities,
        p_eval=p_eval,
        manual_assignment=manual_assignment,
        min_used_qpus=2,
    )
    best_candidate = select_best_allocation_candidate(candidates)
    return best_candidate["allocation"]


def validate_fixed_angle_equivalence(
    H: np.ndarray,
    f: Optional[np.ndarray] = None,
    c0: float = 0.0,
    name: str = "generic_qubo",
    p: int = 1,
    gammas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    num_qpus: int = 2,
    capacities: Optional[List[int]] = None,
    manual_assignment: Optional[List[int]] = None,
    shots: int = 1024,
    seed_sim: int = 123,
    seed_trans: int = 321,
) -> Dict[str, Any]:
    require_qiskit()
    instance = canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=name)
    if gammas is None:
        gammas = [0.3] * p
    if betas is None:
        betas = [0.7] * p
    allocation = allocation_for_validation(
        instance=instance,
        num_qpus=num_qpus,
        capacities=capacities,
        manual_assignment=manual_assignment,
        p_eval=p,
    )

    train_abs = TrainConfig(seed_trans=seed_trans)
    train_tel = TrainConfig(seed_trans=seed_trans)
    runner_abs = get_cached_qubo_runner(
        instance=instance,
        p=p,
        allocation=allocation,
        remote_mode="abstract",
        train_cfg=train_abs,
        measure=True,
    )
    runner_tel = get_cached_qubo_runner(
        instance=instance,
        p=p,
        allocation=allocation,
        remote_mode="telegate_explicit",
        train_cfg=train_tel,
        measure=True,
    )

    counts_abs = runner_abs.run_counts(gammas=gammas, betas=betas, shots=shots, seed_sim=seed_sim)
    counts_tel = runner_tel.run_counts(gammas=gammas, betas=betas, shots=shots, seed_sim=seed_sim)

    dist_abs = decoded_distribution_from_counts(counts_abs, instance.n, explicit_telegate=False)
    dist_tel = decoded_distribution_from_counts(counts_tel, instance.n, explicit_telegate=True)
    tvd = total_variation_distance(dist_abs, dist_tel)
    st_abs = counts_qubo_stats(counts_abs, instance, explicit_telegate=False)
    st_tel = counts_qubo_stats(counts_tel, instance, explicit_telegate=True)
    out = {
        "allocation": allocation,
        "gammas": gammas,
        "betas": betas,
        "shots": int(shots),
        "tvd_decoded_distribution": float(tvd),
        "abstract": {
            "best_bitstring": st_abs["best_bs_qubit"],
            "best_cost": st_abs["best_cost_observed"],
            "mean_cost": st_abs["mean_cost"],
            "remote_cx_count": runner_abs.static_stats["remote_cx"],
        },
        "telegate_explicit": {
            "best_bitstring": st_tel["best_bs_qubit"],
            "best_cost": st_tel["best_cost_observed"],
            "mean_cost": st_tel["mean_cost"],
            "remote_cx_count": runner_tel.static_stats["remote_cx"],
        },
    }
    return out


def print_validation_summary_fixed_angle(res: Dict[str, Any]):
    print("\n========== FIXED-ANGLE EQUIVALENCE CHECK ==========")
    print(f"shots                     = {res['shots']}")
    print(f"gammas                    = {res['gammas']}")
    print(f"betas                     = {res['betas']}")
    print(f"TVD(decoded distribution) = {res['tvd_decoded_distribution']}")
    print("-- abstract --")
    print(f"best_bitstring            = {res['abstract']['best_bitstring']}")
    print(f"best_cost                 = {res['abstract']['best_cost']}")
    print(f"mean_cost                 = {res['abstract']['mean_cost']}")
    print(f"remote_cx_count           = {res['abstract']['remote_cx_count']}")
    print("-- telegate_explicit --")
    print(f"best_bitstring            = {res['telegate_explicit']['best_bitstring']}")
    print(f"best_cost                 = {res['telegate_explicit']['best_cost']}")
    print(f"mean_cost                 = {res['telegate_explicit']['mean_cost']}")
    print(f"remote_cx_count           = {res['telegate_explicit']['remote_cx_count']}")
    print("===================================================\n")
