from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from dqaoa_streamlit_demo import solver_backend as dq


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Built-in research examples
# -----------------------------------------------------------------------------
def _upper_triangular_example_6() -> tuple[np.ndarray, np.ndarray, float, str]:
    """Previously used 6-variable non-trivial QUBO example."""
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
    desc = "Exact 6-variable non-trivial QUBO case study used in the Streamlit demo. Recommended distributed setting: 2 QPUs with capacities [3, 3]."
    return H, f, c0, desc


def _jupyter_example_12() -> tuple[np.ndarray, np.ndarray, float, str]:
    """Exact 12-variable QUBO example from the Jupyter experiment."""
    H = np.array(
        [
            [-0.36,  0.82,  0.00,  0.00,  0.00,  0.61,  0.00,  0.00,  0.00,  0.00,  0.00, -0.48],
            [ 0.00,  0.24, -0.57,  0.00,  0.00,  0.00,  0.43,  0.00,  0.00,  0.00,  0.00,  0.00],
            [ 0.00,  0.00, -0.19,  0.74,  0.00,  0.00,  0.00, -0.51,  0.00,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.28, -0.66,  0.00,  0.00,  0.00,  0.46,  0.00,  0.00,  0.00],
            [ 0.00, -0.21,  0.00,  0.00, -0.33,  0.79,  0.00,  0.00,  0.00, -0.42,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.18, -0.53,  0.00,  0.00,  0.00,  0.58,  0.00],
            [ 0.00,  0.00,  0.32,  0.00,  0.00,  0.00, -0.14,  0.69,  0.00,  0.00,  0.00, -0.36],
            [ 0.00,  0.00,  0.00, -0.27,  0.00,  0.00,  0.00,  0.31, -0.71,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.27,  0.64,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.24,  0.00,  0.00,  0.00,  0.00,  0.22, -0.59,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00, -0.19,  0.00,  0.00,  0.00,  0.00, -0.17,  0.73],
            [ 0.35,  0.00,  0.00, -0.31,  0.00,  0.00,  0.00,  0.22,  0.00,  0.00, -0.17,  0.18],
        ],
        dtype=float,
    )
    f = np.array(
        [0.21, -0.68, 0.33, -0.57, 0.46, -0.62, 0.18, -0.41, 0.52, -0.36, 0.27, -0.24],
        dtype=float,
    )
    c0 = 0.25
    desc = (
        "Exact 12-variable non-trivial QUBO example from the Jupyter experiment. "
        "Recommended distributed setting used in the experiment: 2 QPUs with capacities [6, 6]."
    )
    return H, f, c0, desc


def _jupyter_example_15() -> tuple[np.ndarray, np.ndarray, float, str]:
    """Exact 15-variable QUBO example from the Jupyter experiment."""
    H = np.array(
        [
            [-0.31,  0.76,  0.00,  0.00,  0.00,  0.48,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.41,  0.00,  0.00],
            [ 0.00,  0.26, -0.63,  0.00,  0.00,  0.00,  0.39,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.28,  0.00],
            [ 0.00,  0.00, -0.22,  0.71,  0.00,  0.00,  0.00, -0.46,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.34],
            [ 0.00,  0.00,  0.00,  0.19, -0.67,  0.00,  0.00,  0.00,  0.58,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00, -0.29,  0.81,  0.00,  0.00,  0.00, -0.37,  0.00,  0.00,  0.00,  0.00,  0.00],
            [ 0.00, -0.25,  0.00,  0.00,  0.00,  0.17, -0.54,  0.00,  0.00,  0.00,  0.62,  0.00,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.12,  0.73,  0.00,  0.00,  0.00, -0.44,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.33,  0.00,  0.00,  0.00,  0.00,  0.28, -0.69,  0.00,  0.00,  0.00,  0.51,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.25,  0.66,  0.00,  0.00,  0.00, -0.32,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.18,  0.00,  0.00,  0.00,  0.00,  0.21, -0.61,  0.00,  0.00,  0.00,  0.47],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.18,  0.77,  0.00,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.21,  0.00,  0.00,  0.00,  0.00,  0.24, -0.57,  0.00,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.16,  0.68,  0.00],
            [ 0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.29,  0.00,  0.00,  0.00,  0.00,  0.20, -0.53],
            [ 0.27,  0.00,  0.00,  0.00, -0.34,  0.00,  0.00, -0.26,  0.00,  0.41,  0.00,  0.00, -0.22,  0.00,  0.18],
        ],
        dtype=float,
    )
    f = np.array(
        [0.19, -0.71, 0.29, -0.62, 0.43, -0.55, 0.16, -0.39, 0.58, -0.34, 0.24, -0.49, 0.37, -0.27, 0.31],
        dtype=float,
    )
    c0 = 0.35
    desc = (
        "Exact 15-variable non-trivial QUBO example from the Jupyter experiment. "
        "Recommended distributed setting used in the experiment: 3 QPUs with capacities [7, 6, 4]."
    )
    return H, f, c0, desc


def sample_qubo_instances() -> dict[str, dict[str, Any]]:
    H6, f6, c06, d6 = _upper_triangular_example_6()
    H12, f12, c012, d12 = _jupyter_example_12()
    H15, f15, c015, d15 = _jupyter_example_15()
    return {
        "6-variable example": {
            "H": H6,
            "f": f6,
            "c0": c06,
            "description": d6,
            "recommended_num_qpus": 2,
            "recommended_capacities": [3, 3],
        },
        "12-variable example": {
            "H": H12,
            "f": f12,
            "c0": c012,
            "description": d12,
            "recommended_num_qpus": 2,
            "recommended_capacities": [6, 6],
        },
        "15-variable example": {
            "H": H15,
            "f": f15,
            "c0": c015,
            "description": d15,
            "recommended_num_qpus": 3,
            "recommended_capacities": [7, 6, 4],
        },
    }


# -----------------------------------------------------------------------------
# Matrix/vector parsing helpers
# -----------------------------------------------------------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _strip_np_wrappers(text: str) -> str:
    text = (text or "").strip()
    # Accept np.array([...]), numpy.array([...]), or array([...]) paste forms.
    text = re.sub(r"^(?:np\.|numpy\.)?array\s*\(", "", text.strip())
    if text.endswith(")"):
        text = text[:-1]
    return text.strip()


def _rows_from_bracketed_text(text: str) -> list[list[float]]:
    """Parse rows from NumPy-style or loose bracketed text."""
    text = _strip_np_wrappers(text)
    text = text.replace(";", "\n")
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Remove brackets and trailing commas that appear in array2string output.
        line = line.strip().strip(",")
        line = line.replace("[", " ").replace("]", " ").strip().strip(",")
        nums = [float(x) for x in _FLOAT_RE.findall(line)]
        if nums:
            rows.append(nums)
    return rows


def parse_array_text(text: str, expected_ndim: Optional[int] = None) -> np.ndarray:
    """
    Parse matrix/vector text robustly.

    Accepted formats include:
      - Python/JSON lists: [[1, 2], [3, 4]]
      - NumPy-style arrays without commas: [[1. 2.]\n [3. 4.]]
      - CSV rows: 1,2\n3,4
      - whitespace rows: 1 2\n3 4
      - semicolon rows: 1 2; 3 4
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Input text is empty.")

    literal_text = _strip_np_wrappers(text)
    try:
        value = ast.literal_eval(literal_text)
        arr = np.asarray(value, dtype=float)
    except Exception:
        rows = _rows_from_bracketed_text(text)
        if not rows:
            raise ValueError("No numeric values were found.")
        if expected_ndim == 1:
            arr = np.asarray([v for row in rows for v in row], dtype=float)
        else:
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                # One-line matrix with all numbers and no row boundaries.
                flat = [v for row in rows for v in row]
                side = int(round(np.sqrt(len(flat))))
                if expected_ndim == 2 and side * side == len(flat):
                    arr = np.asarray(flat, dtype=float).reshape(side, side)
                else:
                    raise ValueError(f"Rows have different lengths: {sorted(widths)}")
            else:
                arr = np.asarray(rows, dtype=float)

    if expected_ndim is not None:
        if expected_ndim == 1:
            arr = np.asarray(arr, dtype=float).reshape(-1)
        elif arr.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D input, got shape {arr.shape}.")
    return np.asarray(arr, dtype=float)


def _read_csv_flexible(uploaded_file) -> pd.DataFrame:
    """Read CSV/TXT with or without headers/index columns and coerce to numeric."""
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        text = raw.decode("latin-1")

    # First try pandas CSV. Then fall back to the same text parser used for paste.
    try:
        df = pd.read_csv(io.StringIO(text), header=None)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if numeric.empty:
            raise ValueError("No numeric CSV cells were found.")
        return numeric.astype(float)
    except Exception:
        arr = parse_array_text(text)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return pd.DataFrame(arr.astype(float))


def read_uploaded_csv(uploaded_file, expected_ndim: Optional[int] = None) -> np.ndarray:
    df = _read_csv_flexible(uploaded_file)
    arr = df.to_numpy(dtype=float)
    if expected_ndim == 1:
        arr = arr.reshape(-1)
    if expected_ndim == 2 and arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {arr.shape}.")
    return arr.astype(float)


def parse_optional_vector(text: str, n: int) -> np.ndarray:
    if not text.strip():
        return np.zeros(n, dtype=float)
    f = parse_array_text(text, expected_ndim=1).reshape(-1)
    if len(f) != n:
        raise ValueError(f"f must have length n={n}; got len(f)={len(f)}.")
    return f


def parse_optional_list_int(text: str) -> Optional[list[int]]:
    text = (text or "").strip()
    if not text:
        return None
    value = ast.literal_eval(text)
    return [int(v) for v in value]


def dataframe_to_float_array(df: pd.DataFrame, shape: tuple[int, ...]) -> np.ndarray:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = numeric.to_numpy(dtype=float)
    if arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}.")
    return arr


def make_blank_H_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(np.zeros((n, n), dtype=float), columns=[f"x{j}" for j in range(n)])


def make_blank_f_df(n: int) -> pd.DataFrame:
    return pd.DataFrame([np.zeros(n, dtype=float)], columns=[f"x{j}" for j in range(n)])


def matrix_to_csv_bytes(arr: np.ndarray) -> bytes:
    return pd.DataFrame(arr).to_csv(index=False, header=False).encode("utf-8")


def vector_to_csv_bytes(arr: np.ndarray) -> bytes:
    return pd.DataFrame(np.asarray(arr, dtype=float).reshape(1, -1)).to_csv(index=False, header=False).encode("utf-8")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def visible_result_fields(value):
    """Hide requested metrics in presentation/export copies, not solver data."""
    if isinstance(value, dict):
        return {
            key: visible_result_fields(item)
            for key, item in value.items()
            if "mean_cost" not in str(key).lower()
            and "p_best" not in str(key).lower()
        }
    if isinstance(value, (list, tuple)):
        return [visible_result_fields(item) for item in value]
    return value


def summarize_results_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Compact user-facing result table.

    Deliberately omits low-level/internal columns requested to be hidden from
    the main table: incumbent_cost, p_incumbent_cost, runtime_s, remote_cx,
    and practical_converged. These values remain available in raw JSON export.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "mode": mode_label(r.get("mode")),
                "status": r.get("status"),
                "best_bitstring": r.get("best_bitstring"),
                "best_cost": r.get("best_cost"),
                "optimality_gap_to_exact": r.get("optimality_gap_to_exact"),
                "cross_qpu_terms": r.get("cross_qpu_terms"),
                "cost_match_exact": r.get("cost_match_exact"),
                "bitstring_match_exact": r.get("bitstring_match_exact"),
                "certified_optimal": r.get("certified_optimal"),
            }
        )
    return pd.DataFrame(rows)


def build_plot_metrics_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Internal plotting table that keeps metrics hidden from the visible results table."""
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "mode": mode_label(r.get("mode")),
                "backend_mode": r.get("mode"),
                "best_cost": r.get("best_cost"),
                "optimality_gap_to_exact": r.get("optimality_gap_to_exact"),
                "runtime_s": r.get("runtime_to_solution_seconds"),
                "remote_cx": r.get("remote_cx_count"),
                "cross_qpu_terms": r.get("cross_qpu_terms"),
            }
        )
    df = pd.DataFrame(rows)
    for col in [
        "best_cost",
        "optimality_gap_to_exact",
        "runtime_s",
        "remote_cx",
        "cross_qpu_terms",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _exact_reference_cost(results: list[dict[str, Any]]) -> Optional[float]:
    """Return the brute-force exact cost if available, otherwise any exact_optimum_cost."""
    for r in results:
        if r.get("mode") == "bruteforce" and r.get("best_cost") is not None:
            return float(r["best_cost"])
    for r in results:
        if r.get("exact_optimum_cost") is not None:
            return float(r["exact_optimum_cost"])
    return None


def build_solution_accuracy_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build solution-accuracy rows against exact brute force.

    Cost accuracy is a stable percent score where 100% means the selected mode
    reached the exact brute-force cost. It is computed from the relative cost
    gap using max(1, |exact_cost|) in the denominator to avoid instability when
    the exact cost is close to zero.
    """
    exact_cost = _exact_reference_cost(results)
    if exact_cost is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    denom = max(1.0, abs(float(exact_cost)))
    for r in results:
        if r.get("mode") == "bruteforce" or r.get("best_cost") is None:
            continue
        best_cost = float(r["best_cost"])
        abs_gap = abs(best_cost - float(exact_cost))
        rel_gap_percent = 100.0 * abs_gap / denom
        cost_accuracy_percent = max(0.0, 100.0 - rel_gap_percent)
        rows.append(
            {
                "mode": mode_label(r.get("mode")),
                "best_cost": best_cost,
                "exact_cost": float(exact_cost),
                "absolute_gap": abs_gap,
                "relative_gap_percent": rel_gap_percent,
                "cost_accuracy_percent": cost_accuracy_percent,
                "cost_match_exact": bool(r.get("cost_match_exact")) if r.get("cost_match_exact") is not None else False,
                "bitstring_match_exact": bool(r.get("bitstring_match_exact")) if r.get("bitstring_match_exact") is not None else False,
            }
        )
    return pd.DataFrame(rows)


def build_variational_parameters_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return final optimized QAOA angles as a clean layer-by-layer table."""
    rows: list[dict[str, Any]] = []
    qaoa_modes = {"monolithic_qaoa", "telegate_explicit_qaoa"}
    for r in results:
        if r.get("mode") not in qaoa_modes:
            continue
        gammas = r.get("gammas") or []
        betas = r.get("betas") or []
        if not gammas and not betas:
            continue
        p_eff = max(len(gammas), len(betas))
        for layer_idx in range(p_eff):
            rows.append(
                {
                    "mode": mode_label(r.get("mode")),
                    "p": r.get("p"),
                    "layer": layer_idx + 1,
                    "gamma": float(gammas[layer_idx]) if layer_idx < len(gammas) else np.nan,
                    "beta": float(betas[layer_idx]) if layer_idx < len(betas) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def display_environment_status() -> None:
    qiskit_msg = "available" if dq.HAVE_QISKIT else f"missing: {dq.QISKIT_IMPORT_ERROR}"
    docplex_msg = "available" if dq.HAVE_DOCPLEX else f"missing: {dq.DOCPLEX_IMPORT_ERROR}"
    mpl_msg = "available" if dq.HAVE_MATPLOTLIB else f"missing: {dq.MATPLOTLIB_IMPORT_ERROR}"
    with st.expander("Environment status", expanded=False):
        st.write(
            {
                "Qiskit / Qiskit Aer": qiskit_msg,
                "DOcplex / CPLEX interface": docplex_msg,
                "Matplotlib": mpl_msg,
            }
        )
        if not dq.HAVE_QISKIT:
            st.warning("QAOA modes require qiskit and qiskit-aer. Brute force can still run.")
        if not dq.HAVE_DOCPLEX:
            st.info("MIQP/CPLEX mode is optional and also requires a local CPLEX runtime.")


MODE_LABEL_TO_BACKEND = {
    "Brute force exact": "bruteforce",
    "MIQP CPLEX (optional)": "miqp_cplex",
    "Monolithic QAOA": "monolithic_qaoa",
    "Distributed QAOA": "telegate_explicit_qaoa",
}
BACKEND_TO_MODE_LABEL = {v: k for k, v in MODE_LABEL_TO_BACKEND.items()}


def mode_label(mode: Any) -> str:
    return BACKEND_TO_MODE_LABEL.get(str(mode), str(mode))


def default_mode_labels_for_n(n: int) -> list[str]:
    # Default to exact + both paper-facing quantum modes. Abstract distributed
    # mode is intentionally hidden from the UI.
    return ["Brute force exact", "Monolithic QAOA", "Distributed QAOA"]


def suggest_capacities(n: int, num_qpus: int) -> list[int]:
    if num_qpus <= 1:
        return [int(n)]
    return [int(v) for v in dq.make_balanced_capacities(int(n), int(num_qpus))]


def parse_capacity_text_or_default(text: str, n: int, num_qpus: int) -> list[int]:
    parsed = parse_optional_list_int(text)
    if parsed is None:
        return suggest_capacities(n, num_qpus)
    return [int(v) for v in parsed]


def resolve_capacities_for_problem(
    text: str,
    n: int,
    num_qpus: int,
    recommended: Optional[list[int]] = None,
) -> list[int]:
    """Use typed capacities first; otherwise use a matching built-in recommendation, then balanced defaults."""
    parsed = parse_optional_list_int(text)
    if parsed is not None:
        return [int(v) for v in parsed]
    if recommended is not None and len(recommended) == int(num_qpus) and sum(int(v) for v in recommended) >= int(n):
        return [int(v) for v in recommended]
    return suggest_capacities(n, num_qpus)


def validate_qpu_settings(
    *,
    n: int,
    selected_mode_labels: list[str],
    num_qpus: int,
    capacities: list[int],
    manual_assignment: Optional[list[int]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    has_mono = "Monolithic QAOA" in selected_mode_labels
    has_dist = "Distributed QAOA" in selected_mode_labels

    if has_dist and int(num_qpus) <= 1:
        errors.append(
            "Distributed QAOA requires more than one QPU because cross-QPU TeleGate operations need at least two partitions. "
            f"Set Number of QPUs to 2 or more. Suggested capacities for n={n}: {suggest_capacities(n, 2)}."
        )
    if has_mono and not has_dist and int(num_qpus) != 1:
        errors.append(
            "Monolithic QAOA should use exactly one QPU in this demo. Set Number of QPUs to 1 and capacities to [n]. "
            f"For this problem, use capacities = [{n}]."
        )
    if has_mono and has_dist:
        warnings.append(
            "Both Monolithic QAOA and Distributed QAOA are selected. The QPU settings are used for Distributed QAOA; "
            "Monolithic QAOA still runs internally as a single-QPU baseline."
        )

    if len(capacities) != int(num_qpus):
        errors.append(
            f"The capacities list must have exactly one entry per QPU. You selected {num_qpus} QPU(s), "
            f"but capacities has length {len(capacities)}. Suggested capacities: {suggest_capacities(n, num_qpus)}."
        )
    if any(int(v) < 0 for v in capacities):
        errors.append("QPU capacities must be nonnegative integers.")
    if sum(int(v) for v in capacities) < int(n):
        errors.append(
            f"The sum of QPU capacities is {sum(capacities)}, but the QUBO has n={n} variables/qubits. "
            f"Every variable must fit on a QPU. Increase the capacities so their sum is at least {n}; "
            f"for example, use {suggest_capacities(n, num_qpus)}."
        )

    if manual_assignment is not None:
        if len(manual_assignment) != int(n):
            errors.append(
                f"Manual QPU assignment must contain exactly n={n} entries. Got {len(manual_assignment)} entries."
            )
        else:
            bad = [q for q in manual_assignment if q < 0 or q >= int(num_qpus)]
            if bad:
                errors.append(
                    f"Manual QPU assignment contains invalid QPU indices {sorted(set(bad))}. "
                    f"Valid indices are 0 through {int(num_qpus) - 1}."
                )
            counts = [manual_assignment.count(q) for q in range(int(num_qpus))]
            overloaded = [q for q, ct in enumerate(counts) if q < len(capacities) and ct > capacities[q]]
            if overloaded:
                errors.append(
                    f"Manual QPU assignment exceeds capacity on QPU(s) {overloaded}. Counts are {counts}, capacities are {capacities}."
                )
            if has_dist and len({q for q in manual_assignment if 0 <= q < int(num_qpus)}) < 2:
                errors.append(
                    "Distributed QAOA needs at least two used QPUs. Change the manual assignment so variables are split across two or more QPUs."
                )

    return errors, warnings


def inspect_background_job() -> dict[str, Any]:
    """Return current background-job status without mutating results.

    This intentionally does not load solver outputs. The UI uses it to enable
    the Show results button only after the result file exists.
    """
    job = st.session_state.get("active_job")
    if not job:
        return {"state": "idle", "message": "No active job."}

    proc = job.get("process")
    result_path = Path(job.get("result_path", ""))
    log_path = Path(job.get("log_path", ""))
    elapsed = time.time() - float(job.get("started_at", time.time()))
    pid = proc.pid if proc is not None else None

    if result_path.exists():
        return {
            "state": "ready",
            "message": "Solver run is complete. Results are ready to show.",
            "elapsed": elapsed,
            "pid": pid,
            "result_path": str(result_path),
            "log_path": str(log_path),
        }

    if proc is not None and proc.poll() is not None:
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(errors="replace")[-4000:]
        return {
            "state": "failed",
            "message": f"Solver process exited with code {proc.returncode}, but no result file was produced.",
            "elapsed": elapsed,
            "pid": pid,
            "log_path": str(log_path),
            "log_tail": log_tail,
        }

    return {
        "state": "running",
        "message": "Solver is still running.",
        "elapsed": elapsed,
        "pid": pid,
        "result_path": str(result_path),
        "log_path": str(log_path),
    }


def load_ready_results_from_active_job() -> bool:
    """Load completed solver output after the user clicks Show results."""
    job = st.session_state.get("active_job")
    if not job:
        st.session_state["job_error"] = "There is no active run to show."
        return False

    result_path = Path(job.get("result_path", ""))
    proc = job.get("process")

    if not result_path.exists():
        st.session_state["job_error"] = "Results are not ready yet. The solver is still running."
        return False

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.session_state["job_error"] = f"Could not read result file: {exc}"
        return False

    if proc is not None and proc.poll() is None:
        # The result file is the completion signal, but terminate defensively if
        # the child process has not exited yet.
        try:
            proc.terminate()
        except Exception:
            pass

    st.session_state.pop("active_job", None)
    if payload.get("status") == "done":
        st.session_state["last_results"] = payload.get("results", [])
        st.session_state["last_problem"] = job.get("problem", {})
        st.session_state["job_success"] = "Solver run finished. Results loaded."
        return True

    st.session_state["job_error"] = f"Run failed: {payload.get('error_type')}: {payload.get('error_message')}"
    st.session_state["job_traceback"] = payload.get("traceback")
    return False


def clear_failed_active_job() -> None:
    st.session_state.pop("active_job", None)


def start_background_job(job_payload: dict[str, Any], problem_payload: dict[str, Any]) -> None:
    jobs_dir = OUTPUT_DIR / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    job_path = jobs_dir / f"job_{job_id}.json"
    result_path = jobs_dir / f"result_{job_id}.json"
    log_path = jobs_dir / f"log_{job_id}.txt"

    job_path.write_text(json.dumps(to_jsonable(job_payload), indent=2), encoding="utf-8")
    log_fh = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "dqaoa_streamlit_demo.job_runner", str(job_path), str(result_path)],
            cwd=str(APP_DIR),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_fh.close()

    st.session_state["active_job"] = {
        "id": job_id,
        "process": proc,
        "job_path": str(job_path),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "started_at": time.time(),
        "problem": problem_payload,
    }
    st.session_state.pop("job_error", None)
    st.session_state.pop("job_success", None)
    st.session_state.pop("job_traceback", None)
    st.session_state["job_started"] = "Solver run started in the background. The Stop button is now available."


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DQAOA-QUBO Solver Studio",
    page_icon="🧩",
    layout="wide",
)

st.title("DQAOA-QUBO Solver Studio")
st.caption(
    "Research demo UI for exact brute force, Monolithic QAOA, and Distributed QAOA with TeleGate-explicit remote operations."
)

st.markdown(
    """
    <style>
    /* Graphic research-style background: subtle circuit/grid texture over a soft gradient. */
    [data-testid="stAppViewContainer"] {
        background-image:
            radial-gradient(circle at 12% 18%, rgba(79, 70, 229, 0.18), transparent 28%),
            radial-gradient(circle at 86% 8%, rgba(14, 165, 233, 0.16), transparent 26%),
            radial-gradient(circle at 75% 88%, rgba(16, 185, 129, 0.12), transparent 30%),
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='220' height='220' viewBox='0 0 220 220'%3E%3Cg fill='none' stroke='%235b6ee1' stroke-opacity='0.13' stroke-width='1.2'%3E%3Cpath d='M20 35h58v42h46v48h76'/%3E%3Cpath d='M24 155h54v-38h72V72h46'/%3E%3Ccircle cx='20' cy='35' r='4' fill='%235b6ee1' fill-opacity='0.16'/%3E%3Ccircle cx='78' cy='77' r='4' fill='%235b6ee1' fill-opacity='0.16'/%3E%3Ccircle cx='150' cy='117' r='4' fill='%235b6ee1' fill-opacity='0.16'/%3E%3Ccircle cx='196' cy='72' r='4' fill='%235b6ee1' fill-opacity='0.16'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #eef2ff 0%, #f7fbff 46%, #eaf7ff 100%);
        background-size: auto, auto, auto, 260px 260px, cover;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.0);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(241,245,255,0.94), rgba(232,243,255,0.94));
        border-right: 1px solid rgba(99,102,241,0.16);
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetric"], div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
        border-radius: 12px;
    }
    .running-card {
        border: 1px solid rgba(49, 130, 206, 0.32);
        border-left: 6px solid rgba(49, 130, 206, 0.95);
        border-radius: 14px;
        padding: 1.0rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        background: rgba(255,255,255,0.72);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.10);
        backdrop-filter: blur(10px);
    }
    .running-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .running-subtitle {
        color: rgba(49, 49, 49, 0.80);
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .soft-dot {
        display: inline-block;
        width: 0.58rem;
        height: 0.58rem;
        margin-right: 0.45rem;
        border-radius: 50%;
        background: rgba(49,130,206,0.95);
        box-shadow: 0 0 0 0 rgba(49,130,206,0.40);
        animation: dqaoaSoftPulse 1.8s ease-out infinite;
    }
    @keyframes dqaoaSoftPulse {
        0% { box-shadow: 0 0 0 0 rgba(49,130,206,0.35); }
        70% { box-shadow: 0 0 0 9px rgba(49,130,206,0.0); }
        100% { box-shadow: 0 0 0 0 rgba(49,130,206,0.0); }
    }

    .ready-dot {
        display: inline-block;
        width: 0.58rem;
        height: 0.58rem;
        margin-right: 0.45rem;
        border-radius: 50%;
        background: rgba(22,163,74,0.95);
        box-shadow: 0 0 0 4px rgba(22,163,74,0.12);
    }
    .error-dot {
        display: inline-block;
        width: 0.58rem;
        height: 0.58rem;
        margin-right: 0.45rem;
        border-radius: 50%;
        background: rgba(220,38,38,0.95);
        box-shadow: 0 0 0 4px rgba(220,38,38,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

display_environment_status()
samples = sample_qubo_instances()

with st.sidebar:
    st.header("Problem size")
    problem_size_n = st.number_input(
        "n variables",
        min_value=1,
        max_value=25,
        value=6,
        step=1,
        help="For Manual editor mode, this creates an editable n × n H matrix and a 1 × n f vector.",
    )
    problem_size_n = int(problem_size_n)

    st.header("Solver settings")
    st.caption(
        "Use Fast demo while exploring the UI, then switch to Balanced/Accuracy when you want stronger QAOA runs."
    )

    preset_defaults = {
        "Fast demo (recommended)": {
            "p_max": 1,
            "shots_train": 256,
            "final_shots": 128,
            "avg_k": 1,
            "adam_iters": 4,
            "num_random_starts": 2,
            "warm_start_perturbations": 0,
            "parallel_restarts": 4,
            "eval_at_each_iter": False,
        },
        "Balanced research": {
            "p_max": 2,
            "shots_train": 512,
            "final_shots": 256,
            "avg_k": 1,
            "adam_iters": 20,
            "num_random_starts": 4,
            "warm_start_perturbations": 1,
            "parallel_restarts": 4,
            "eval_at_each_iter": True,
        },
        "Accuracy check (slower)": {
            "p_max": 2,
            "shots_train": 512,
            "final_shots": 512,
            "avg_k": 1,
            "adam_iters": 8,
            "num_random_starts": 12,
            "warm_start_perturbations": 3,
            "parallel_restarts": 4,
            "eval_at_each_iter": False,
        },
    }
    speed_preset = st.selectbox(
        "Run preset",
        options=list(preset_defaults.keys()),
        index=0,
        help=(
            "Fast demo is designed for quick UI testing. Balanced research keeps the heavier defaults you requested earlier. "
            "Accuracy check uses more starts but fewer optimizer iterations, similar to the Jupyter test style."
        ),
    )
    preset = preset_defaults[speed_preset]

    selected_mode_labels = st.multiselect(
        "Modes to run",
        options=[
            "Brute force exact",
            "MIQP CPLEX (optional)",
            "Monolithic QAOA",
            "Distributed QAOA",
        ],
        default=default_mode_labels_for_n(problem_size_n),
        help=(
            "The abstract distributed mode has been removed from the UI. "
            "Distributed QAOA maps to the TeleGate-explicit backend."
        ),
    )
    modes = [MODE_LABEL_TO_BACKEND[label] for label in selected_mode_labels]

    st.subheader("QAOA depth and shots")
    preset_key = speed_preset.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    p_max = st.slider("Maximum QAOA depth p_max", min_value=1, max_value=3, value=int(preset["p_max"]), step=1, key=f"p_max_{preset_key}")
    shots_train = st.number_input("Training shots", min_value=64, max_value=8192, value=int(preset["shots_train"]), step=64, key=f"shots_train_{preset_key}")
    final_shots = st.number_input("Final shots", min_value=64, max_value=32768, value=int(preset["final_shots"]), step=64, key=f"final_shots_{preset_key}")
    avg_k = st.number_input("Shot-average repetitions avg_k", min_value=1, max_value=5, value=int(preset["avg_k"]), step=1, key=f"avg_k_{preset_key}")

    st.subheader("Optimizer")
    adam_iters = st.number_input("SPSA-Adam iterations", min_value=1, max_value=200, value=int(preset["adam_iters"]), step=1, key=f"adam_iters_{preset_key}")
    adam_lr = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.08, step=0.01, format="%.3f")
    fd_eps0 = st.number_input("SPSA perturbation fd_eps0", min_value=0.001, max_value=1.0, value=0.12, step=0.01, format="%.3f")
    eval_at_each_iter = st.checkbox(
        "Evaluate current angles every optimizer iteration",
        value=bool(preset["eval_at_each_iter"]),
        key=f"eval_at_each_iter_{preset_key}",
        help="When off, SPSA uses only the plus/minus perturbation evaluations during training. This is faster and the final circuit is still evaluated after training.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1)

    st.subheader("Multi-start")
    num_random_starts = st.number_input("Random starts", min_value=0, max_value=20, value=int(preset["num_random_starts"]), step=1, key=f"num_random_starts_{preset_key}")
    warm_start_perturbations = st.number_input("Warm-start perturbations", min_value=0, max_value=10, value=int(preset["warm_start_perturbations"]), step=1, key=f"warm_start_perturbations_{preset_key}")
    parallel_restarts = st.number_input("Parallel restarts", min_value=1, max_value=8, value=int(preset["parallel_restarts"]), step=1, key=f"parallel_restarts_{preset_key}", help="Higher values can reduce wall-clock time on multi-core machines by running restarts concurrently. Use 1 if your computer becomes overloaded.")
    quantum_mode_count_est = sum(1 for lbl in selected_mode_labels if lbl in {"Monolithic QAOA", "Distributed QAOA"})
    evals_per_iter_est = 3 if eval_at_each_iter else 2
    starts_depth1_est = int(num_random_starts)
    starts_later_est = int(num_random_starts) + int(warm_start_perturbations) + 1
    total_restart_depths_est = starts_depth1_est + max(0, int(p_max) - 1) * starts_later_est
    objective_evals_est = int(quantum_mode_count_est * total_restart_depths_est * int(adam_iters) * evals_per_iter_est * int(avg_k))
    circuit_shots_est = int(objective_evals_est * int(shots_train) + quantum_mode_count_est * total_restart_depths_est * int(final_shots))
    st.caption(f"Estimated QAOA training workload: ~{objective_evals_est:,} circuit evaluations and ~{circuit_shots_est:,} simulated shots before exact/CPLEX overhead.")
    if speed_preset != "Fast demo (recommended)" and quantum_mode_count_est >= 2:
        st.warning("This preset can be noticeably slower when both Monolithic and Distributed QAOA are selected. For quick testing, use Fast demo or run one QAOA mode at a time.")


    st.subheader("QPU settings")
    distributed_selected = "Distributed QAOA" in selected_mode_labels
    monolithic_only = ("Monolithic QAOA" in selected_mode_labels) and not distributed_selected
    if distributed_selected:
        num_qpus = int(st.number_input("Number of QPUs", min_value=2, max_value=8, value=2, step=1))
        st.caption(
            f"Distributed QAOA requires at least 2 QPUs. If capacities is blank, the app uses a balanced suggestion based on n: "
            f"{suggest_capacities(problem_size_n, num_qpus)}."
        )
    else:
        num_qpus = 1
        st.number_input(
            "Number of QPUs", min_value=1, max_value=8, value=1, step=1, disabled=True,
            help="Monolithic QAOA uses exactly one QPU in this demo."
        )
        if monolithic_only:
            st.caption(f"Monolithic QAOA default capacity: [{problem_size_n}].")

    capacities_text = st.text_input(
        "QPU capacities",
        value="",
        placeholder=f"Blank = auto suggestion, e.g. {suggest_capacities(problem_size_n, num_qpus)}",
        help="The list length must equal the number of QPUs, and the sum must be at least n.",
    )
    manual_assignment_text = st.text_input(
        "Manual QPU assignment, optional",
        value="",
        placeholder="Example for n=6 and 2 QPUs: [0,0,0,1,1,1]",
    )

    st.subheader("Simulation and exact baseline")
    sim_method = st.selectbox("Aer simulation method", options=["automatic", "matrix_product_state", "statevector"], index=1)
    compute_exact_if_small = st.checkbox("Compute brute-force exact baseline for small n", value=True)
    exact_threshold_n = st.number_input("Exact threshold n", min_value=1, max_value=30, value=16, step=1)
    cplex_time_limit = st.number_input("CPLEX time limit seconds", min_value=0.0, max_value=3600.0, value=30.0, step=5.0)
    store_history = st.checkbox("Store optimizer history", value=False)
    store_final_counts = st.checkbox("Store final counts", value=False)

    st.subheader("Analysis output")
    elite_top_k = st.number_input(
        "Elite top-k bitstrings",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help=(
            "Controls the elite_top_k value used by the solver and the number of elite bitstrings "
            "shown in the output tables and plots."
        ),
    )

    st.divider()
    if st.button("Clear cached Qiskit circuits"):
        dq.clear_circuit_runner_cache()
        st.success("Circuit runner cache cleared.")


input_tab, settings_tab = st.tabs(["QUBO input", "Resolved configuration"])

with input_tab:
    col_left, col_right = st.columns([0.85, 1.25])

    with col_left:
        input_mode = st.radio(
            "Choose input source",
            ["Manual editor", "Built-in example", "Paste matrix", "Upload CSV"],
            horizontal=False,
        )

        H: Optional[np.ndarray] = None
        f: Optional[np.ndarray] = None
        c0: float = 0.0
        instance_name = "generic_qubo"
        sample_recommended_num_qpus: Optional[int] = None
        sample_recommended_capacities: Optional[list[int]] = None

        if input_mode == "Manual editor":
            st.info(
                "Use the n box in the left toolbar. The H matrix and f vector editors appear on the right. "
                "Blank cells are treated as 0."
            )
            c0 = st.number_input("c0", value=0.0, step=1.0, format="%.6f", key="manual_c0")
            instance_name = st.text_input("Instance name", value=f"manual_{problem_size_n}var_qubo")

        elif input_mode == "Built-in example":
            sample_name = st.selectbox("Sample instance", list(samples.keys()))
            sample = samples[sample_name]
            H = sample["H"].copy()
            f = sample["f"].copy()
            c0 = float(sample["c0"])
            instance_name = sample_name.replace(" ", "_").replace("-", "_").lower()
            sample_recommended_num_qpus = sample.get("recommended_num_qpus")
            sample_recommended_capacities = sample.get("recommended_capacities")
            st.info(sample["description"])
            if sample_recommended_num_qpus is not None and sample_recommended_capacities is not None:
                st.success(
                    f"Recommended distributed setting for this built-in example: "
                    f"Number of QPUs = {sample_recommended_num_qpus}, capacities = {sample_recommended_capacities}."
                )
            st.caption("The older toy examples were removed. The built-ins are now 6-, 12-, and 15-variable examples.")

        elif input_mode == "Paste matrix":
            default_H, default_f, default_c0, _ = _upper_triangular_example_6()
            default_text = np.array2string(default_H, separator=" ")
            H_text = st.text_area(
                "H matrix",
                value=default_text,
                height=240,
                help="Accepts Python lists, NumPy-style matrices without commas, CSV rows, whitespace rows, or semicolon-separated rows.",
            )
            f_text = st.text_input("f vector, optional", value=np.array2string(default_f, separator=" "))
            c0 = st.number_input("c0", value=float(default_c0), step=1.0, format="%.6f", key="paste_c0")
            instance_name = st.text_input("Instance name", value="pasted_qubo")
            try:
                H = parse_array_text(H_text, expected_ndim=2)
                if H.shape[0] != H.shape[1]:
                    raise ValueError(f"H must be square; got shape {H.shape}.")
                f = parse_optional_vector(f_text, H.shape[0])
            except Exception as exc:
                st.error(f"Could not parse pasted input: {exc}")

        else:
            st.write("Upload CSV/TXT files. Headers and index columns are tolerated when they are non-numeric.")
            uploaded_H = st.file_uploader("Upload H as CSV/TXT", type=["csv", "txt"])
            uploaded_f = st.file_uploader("Upload f as CSV/TXT, optional", type=["csv", "txt"])
            c0 = st.number_input("c0", value=0.0, step=1.0, format="%.6f", key="upload_c0")
            instance_name = st.text_input("Instance name", value="uploaded_qubo")

            st.download_button(
                "Download blank H CSV template",
                data=matrix_to_csv_bytes(np.zeros((problem_size_n, problem_size_n))),
                file_name=f"blank_H_{problem_size_n}x{problem_size_n}.csv",
                mime="text/csv",
                key="download_blank_H_template",
            )
            st.download_button(
                "Download blank f CSV template",
                data=vector_to_csv_bytes(np.zeros(problem_size_n)),
                file_name=f"blank_f_{problem_size_n}.csv",
                mime="text/csv",
                key="download_blank_f_template",
            )

            if uploaded_H is not None:
                try:
                    H = read_uploaded_csv(uploaded_H, expected_ndim=2)
                    if H.shape[0] != H.shape[1]:
                        raise ValueError(f"H must be square; got shape {H.shape}.")
                    if uploaded_f is not None:
                        f = read_uploaded_csv(uploaded_f, expected_ndim=1)
                        if len(f) != H.shape[0]:
                            raise ValueError(f"f must have length n={H.shape[0]}; got len(f)={len(f)}.")
                    else:
                        f = np.zeros(H.shape[0], dtype=float)
                except Exception as exc:
                    st.error(f"Could not read uploaded files: {exc}")
            else:
                st.info("Upload an H matrix CSV/TXT to continue.")

    with col_right:
        st.subheader("Problem preview")

        if input_mode == "Manual editor":
            st.write(f"Editable H matrix: **{problem_size_n} × {problem_size_n}**")
            H_df = st.data_editor(
                make_blank_H_df(problem_size_n),
                key=f"H_editor_{problem_size_n}",
                use_container_width=True,
                num_rows="fixed",
                height=min(520, 92 + 36 * problem_size_n),
            )
            st.write(f"Editable f vector: **1 × {problem_size_n}**")
            f_df = st.data_editor(
                make_blank_f_df(problem_size_n),
                key=f"f_editor_{problem_size_n}",
                use_container_width=True,
                num_rows="fixed",
                height=105,
            )
            try:
                H = dataframe_to_float_array(H_df, (problem_size_n, problem_size_n))
                f = dataframe_to_float_array(f_df, (1, problem_size_n)).reshape(-1)
            except Exception as exc:
                st.error(f"Manual matrix/vector contains invalid values: {exc}")
                H = None
                f = None

        if H is not None:
            st.write(f"n = **{H.shape[0]}** variables")
            st.write("Quadratic matrix H")
            st.dataframe(pd.DataFrame(H, columns=[f"x{j}" for j in range(H.shape[1])]), use_container_width=True)
            if f is not None:
                st.write("Linear vector f")
                st.dataframe(pd.DataFrame([f], columns=[f"x{j}" for j in range(len(f))]), use_container_width=True)
            st.write(f"c0 = `{float(c0):.6g}`")
            try:
                instance_preview = dq.canonicalize_qubo_from_dense(H=H, f=f, c0=c0, name=instance_name)
                preview_cols = st.columns(3)
                preview_cols[0].metric("Variables", instance_preview.n)
                preview_cols[1].metric("Nonzero linear terms", int(np.sum(np.abs(instance_preview.linear) > 1e-14)))
                preview_cols[2].metric("Quadratic terms", len(instance_preview.quadratic))
            except Exception as exc:
                st.error(f"Canonicalization failed: {exc}")

with settings_tab:
    problem_n_for_config = int(H.shape[0]) if H is not None else int(problem_size_n)
    try:
        capacities = resolve_capacities_for_problem(
            capacities_text,
            problem_n_for_config,
            int(num_qpus),
            sample_recommended_capacities,
        )
        manual_assignment = parse_optional_list_int(manual_assignment_text)
        cfg_errors, cfg_warnings = validate_qpu_settings(
            n=problem_n_for_config,
            selected_mode_labels=list(selected_mode_labels),
            num_qpus=int(num_qpus),
            capacities=capacities,
            manual_assignment=manual_assignment,
        )
        resolved = {
            "actual_problem_n": int(problem_n_for_config),
            "selected_modes_display": selected_mode_labels,
            "backend_modes": modes,
            "p_max": int(p_max),
            "num_qpus": int(num_qpus),
            "capacities": capacities,
            "capacity_sum": int(sum(capacities)),
            "manual_assignment": manual_assignment,
            "shots_train": int(shots_train),
            "final_shots": int(final_shots),
            "avg_k": int(avg_k),
            "optimizer_iters": int(adam_iters),
            "eval_at_each_iter": bool(eval_at_each_iter),
            "random_starts": int(num_random_starts),
            "warm_start_perturbations": int(warm_start_perturbations),
            "parallel_restarts": int(parallel_restarts),
            "sim_method": sim_method,
            "compute_exact_if_small": bool(compute_exact_if_small),
            "exact_threshold_n": int(exact_threshold_n),
            "elite_top_k": int(elite_top_k),
        }
        st.json(resolved)
        for msg in cfg_warnings:
            st.warning(msg)
        for msg in cfg_errors:
            st.error(msg)
        if not cfg_errors:
            st.success("Configuration validation passed.")
        st.info(
            "Capacity rule: the capacities list must have one value per QPU, and the total capacity must be at least the number of variables n. "
            f"A safe default for the current problem is {resolve_capacities_for_problem('', problem_n_for_config, int(num_qpus), sample_recommended_capacities)}."
        )
    except Exception as exc:
        st.error(f"Configuration parse error: {exc}")


# -----------------------------------------------------------------------------
# Background run controls
# -----------------------------------------------------------------------------
problem_n = int(H.shape[0]) if H is not None else int(problem_size_n)
try:
    capacities_for_run = resolve_capacities_for_problem(
        capacities_text,
        problem_n,
        int(num_qpus),
        sample_recommended_capacities,
    )
    manual_assignment_for_run = parse_optional_list_int(manual_assignment_text)
    cfg_errors, cfg_warnings = validate_qpu_settings(
        n=problem_n,
        selected_mode_labels=list(selected_mode_labels),
        num_qpus=int(num_qpus),
        capacities=capacities_for_run,
        manual_assignment=manual_assignment_for_run,
    )
except Exception as exc:
    capacities_for_run = []
    manual_assignment_for_run = None
    cfg_errors = [f"Configuration parse error: {exc}"]
    cfg_warnings = []

active_job = st.session_state.get("active_job")
run_disabled = H is None or f is None or not modes or bool(cfg_errors) or active_job is not None
run_col, note_col = st.columns([1.35, 4.0])
with run_col:
    run_clicked = st.button("Run DQAOA-QUBO demo", type="primary", disabled=run_disabled)
with note_col:
    st.caption(
        "The solver runs in a separate Python process. The page itself does not auto-reload; only the small job-status panel below checks completion every 5 seconds while the run is active."
    )

if cfg_errors:
    with st.expander("Why Run is disabled", expanded=True):
        for msg in cfg_errors:
            st.error(msg)
        st.info(f"Suggested capacities for the current problem: {resolve_capacities_for_problem('', problem_n, int(num_qpus), sample_recommended_capacities)}")

if run_clicked:
    try:
        job_payload = {
            "H": np.asarray(H, dtype=float),
            "f": np.asarray(f, dtype=float),
            "c0": float(c0),
            "name": str(instance_name),
            "modes": list(modes),
            "p_max": int(p_max),
            "num_qpus": int(num_qpus),
            "capacities": capacities_for_run,
            "manual_assignment": manual_assignment_for_run,
            "shots_train": int(shots_train),
            "final_shots": int(final_shots),
            "avg_k": int(avg_k),
            "adam_iters": int(adam_iters),
            "eval_at_each_iter": bool(eval_at_each_iter),
            "adam_lr": float(adam_lr),
            "fd_eps0": float(fd_eps0),
            "seed": int(seed),
            "num_random_starts": int(num_random_starts),
            "warm_start_perturbations": int(warm_start_perturbations),
            "parallel_restarts": int(parallel_restarts),
            "sim_method": str(sim_method),
            "compute_exact_if_small": bool(compute_exact_if_small),
            "exact_threshold_n": int(exact_threshold_n),
            "cplex_time_limit": float(cplex_time_limit) if cplex_time_limit > 0 else None,
            "store_history": bool(store_history),
            "store_final_counts": bool(store_final_counts),
            "elite_top_k": int(elite_top_k),
        }
        problem_payload = {
            "H": np.asarray(H, dtype=float),
            "f": np.asarray(f, dtype=float),
            "c0": float(c0),
            "name": str(instance_name),
            "num_qpus": int(num_qpus),
            "capacities": capacities_for_run,
            "manual_assignment": manual_assignment_for_run,
            "selected_mode_labels": list(selected_mode_labels),
            "backend_modes": list(modes),
            "sample_recommended_num_qpus": sample_recommended_num_qpus,
            "sample_recommended_capacities": sample_recommended_capacities,
        }
        start_background_job(job_payload, problem_payload)
    except Exception as exc:
        st.error(f"Could not start run: {type(exc).__name__}: {exc}")

def _render_active_job_panel_body() -> None:
    active_job = st.session_state.get("active_job")
    if active_job is None:
        return

    status = inspect_background_job()
    state = status.get("state")
    elapsed = float(status.get("elapsed", 0.0) or 0.0)
    pid = status.get("pid") or "—"
    selected_labels = active_job.get("problem", {}).get("selected_mode_labels", [])
    job_id = active_job.get("id")

    if state == "ready":
        title = "Results are ready"
        subtitle = "The solver finished. The Show results button is now active."
        state_label = "Ready"
        dot_html = '<span class="ready-dot"></span>'
    elif state == "failed":
        title = "Run finished with an error"
        subtitle = str(status.get("message", "The solver process failed."))
        state_label = "Error"
        dot_html = '<span class="error-dot"></span>'
    else:
        title = "Solver is running"
        subtitle = "Show results is disabled until the solver writes the result file."
        state_label = "Running"
        dot_html = '<span class="soft-dot"></span>'

    st.markdown(
        f"""
        <div class="running-card">
            <div class="running-title">{dot_html}{title}</div>
            <div class="running-subtitle">
                Job ID: <b>{job_id}</b> &nbsp; | &nbsp; Process ID: <b>{pid}</b> &nbsp; | &nbsp;
                Elapsed time: <b>{elapsed:.1f} seconds</b><br>
                Modes: <b>{', '.join(selected_labels) if selected_labels else 'selected modes'}</b><br>
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Run state", state_label)
    status_cols[1].metric("Elapsed", f"{elapsed:.1f}s")
    status_cols[2].metric("Process ID", str(pid))
    status_cols[3].metric("Results file", "Ready" if state == "ready" else "Not ready")

    action_cols = st.columns([1.25, 1.25, 4.0])
    with action_cols[0]:
        show_clicked = st.button(
            "Show results",
            disabled=(state != "ready"),
            type="primary" if state == "ready" else "secondary",
            key=f"show_results_{job_id}",
            help="This button becomes active only after the solver run is complete.",
        )
    with action_cols[1]:
        stop_clicked_panel = st.button(
            "Stop current run",
            disabled=(state != "running"),
            key=f"stop_current_run_{job_id}",
        )

    if show_clicked:
        if load_ready_results_from_active_job():
            st.rerun()

    if stop_clicked_panel:
        proc = active_job.get("process")
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            st.session_state.pop("active_job", None)
            st.session_state["job_success"] = "Run stopped by user. You can change parameters and start a new run."
            st.rerun()
        except Exception as exc:
            st.error(f"Could not stop the run cleanly: {exc}")

    if state == "failed":
        if status.get("log_tail"):
            with st.expander("Solver process log", expanded=True):
                st.code(str(status.get("log_tail")))
        if st.button("Clear failed run", key=f"clear_failed_run_{job_id}"):
            clear_failed_active_job()
            st.rerun()
    else:
        log_path = Path(active_job.get("log_path", ""))
        if log_path.exists():
            with st.expander("Solver process log", expanded=False):
                st.code(log_path.read_text(errors="replace")[-5000:] or "No log output yet.")


if hasattr(st, "fragment"):
    @st.fragment(run_every=5.0)
    def render_active_job_panel() -> None:
        _render_active_job_panel_body()
else:
    # Fallback for older Streamlit versions. The panel still works, but the
    # Show results button can only update after a normal user interaction.
    def render_active_job_panel() -> None:
        _render_active_job_panel_body()


render_active_job_panel()

if st.session_state.get("job_started"):
    st.success(st.session_state.pop("job_started"))
if st.session_state.get("job_success"):
    st.success(st.session_state.pop("job_success"))
if st.session_state.get("job_error"):
    st.error(st.session_state.pop("job_error"))
    if st.session_state.get("job_traceback"):
        with st.expander("Error traceback / log", expanded=False):
            st.code(st.session_state.pop("job_traceback"))


if "last_results" in st.session_state:
    results = visible_result_fields(st.session_state["last_results"])
    problem = st.session_state.get("last_problem", {})

    st.divider()
    st.header("Results dashboard")
    df = summarize_results_dataframe(results)

    numeric_cols = [
        "best_cost",
        "optimality_gap_to_exact",
        "cross_qpu_terms",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _match_label(v):
        if pd.isna(v):
            return "—"
        return "✅" if bool(v) else "❌"

    clean_df = df.copy()
    for col in ["cost_match_exact", "bitstring_match_exact", "certified_optimal"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].map(_match_label)

    st.subheader("Mode comparison table")
    st.dataframe(
        clean_df,
        use_container_width=True,
        column_config={
            "best_cost": st.column_config.NumberColumn(format="%.8g"),
            "optimality_gap_to_exact": st.column_config.NumberColumn(format="%.3e"),
        },
    )

    metric_cols = st.columns(5)
    successful = [r for r in results if r.get("best_cost") is not None]
    if successful:
        best_res = sorted(successful, key=lambda r: float(r.get("best_cost")))[0]
        exact_cost = best_res.get("exact_optimum_cost")
        metric_cols[0].metric("Best mode", mode_label(best_res.get("mode")))
        metric_cols[1].metric("Best cost", f"{float(best_res.get('best_cost')):.6g}")
        metric_cols[2].metric("Best bitstring", str(best_res.get("best_bitstring")))
        metric_cols[3].metric("Exact cost", "—" if exact_cost is None else f"{float(exact_cost):.6g}")
        metric_cols[4].metric("Cost match", _match_label(best_res.get("cost_match_exact")))
    else:
        st.warning("No selected mode returned a feasible result. Check the status/error details below.")

    tab_summary, tab_charts, tab_elite, tab_params, tab_figures, tab_export = st.tabs(
        ["Summary", "Plots", "Elite bitstrings", "Variational parameters", "Circuit figures", "Export / raw"]
    )

    with tab_summary:
        st.subheader("Clean summaries")
        for r in results:
            with st.expander(f"{mode_label(r.get('mode'))} — {r.get('status')}", expanded=False):
                cols = st.columns(2)
                cols[0].metric("Best cost", "—" if r.get("best_cost") is None else f"{float(r.get('best_cost')):.8g}")
                cols[1].metric("Runtime", "—" if r.get("runtime_to_solution_seconds") is None else f"{float(r.get('runtime_to_solution_seconds')):.3f}s")
                st.json(to_jsonable(visible_result_fields(dq.summarize_result(r))))
                err = r.get("details", {}).get("error_message") if isinstance(r.get("details"), dict) else None
                if err:
                    st.error(err)

    with tab_charts:
        plot_df = build_plot_metrics_dataframe(results)
        chart_df = plot_df.set_index("mode") if not plot_df.empty else pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Best cost by mode")
            if not chart_df.empty and "best_cost" in chart_df:
                st.bar_chart(chart_df[["best_cost"]].dropna())
        with c2:
            st.subheader("Solution accuracy vs exact brute force")
            acc_df = build_solution_accuracy_dataframe(results)
            if acc_df.empty:
                st.info("Accuracy plot requires an exact brute-force reference. Enable brute force or exact baseline for this problem.")
            else:
                st.bar_chart(acc_df.set_index("mode")[["cost_accuracy_percent"]])
                with st.expander("Accuracy details", expanded=False):
                    st.dataframe(
                        acc_df,
                        use_container_width=True,
                        column_config={
                            "best_cost": st.column_config.NumberColumn(format="%.8g"),
                            "exact_cost": st.column_config.NumberColumn(format="%.8g"),
                            "absolute_gap": st.column_config.NumberColumn(format="%.3e"),
                            "relative_gap_percent": st.column_config.NumberColumn(format="%.4f%%"),
                            "cost_accuracy_percent": st.column_config.NumberColumn(format="%.4f%%"),
                        },
                    )
            st.subheader("Cross-QPU terms")
            if not chart_df.empty and "cross_qpu_terms" in chart_df:
                st.bar_chart(chart_df[["cross_qpu_terms"]].dropna())

        if not chart_df.empty and "optimality_gap_to_exact" in chart_df.columns:
            st.subheader("Optimality gap to exact")
            st.bar_chart(chart_df[["optimality_gap_to_exact"]].dropna())

        st.subheader("Top-k elite bitstrings for selected mode")
        qaoa_results_for_plot = [r for r in results if r.get("elite_rows_top_k")]
        if not qaoa_results_for_plot:
            st.info("No elite bitstring rows are available. Run Monolithic QAOA or Distributed QAOA first.")
        else:
            elite_plot_cols = st.columns([1.2, 1.0])
            with elite_plot_cols[0]:
                selected_mode_for_elite_plot = st.selectbox(
                    "Select mode for top-k elite plot",
                    [mode_label(r.get("mode")) for r in qaoa_results_for_plot],
                    key="elite_mode_select_plots",
                )
            selected_plot_result = next(r for r in qaoa_results_for_plot if mode_label(r.get("mode")) == selected_mode_for_elite_plot)
            max_elite = len(selected_plot_result.get("elite_rows_top_k", []))
            with elite_plot_cols[1]:
                top_k_elite_plot = int(min(int(elite_top_k), max_elite)) if max_elite > 0 else 0
                st.metric("Elite bitstrings shown", top_k_elite_plot)
                st.caption(f"Controlled by the left-sidebar setting: Elite top-k bitstrings = {int(elite_top_k)}.")
            elite_plot_df = pd.DataFrame(selected_plot_result.get("elite_rows_top_k", [])).head(int(top_k_elite_plot))
            if not elite_plot_df.empty and {"bitstring", "frequency"}.issubset(elite_plot_df.columns):
                st.bar_chart(elite_plot_df.set_index("bitstring")[["frequency"]])
    with tab_elite:
        qaoa_results = [r for r in results if r.get("elite_rows_top_k")]
        if not qaoa_results:
            st.info("No elite bitstring rows are available. Run Monolithic QAOA or Distributed QAOA first.")
        else:
            elite_controls = st.columns([1.2, 1.0])
            with elite_controls[0]:
                selected_mode_for_elite = st.selectbox(
                    "Mode",
                    [mode_label(r.get("mode")) for r in qaoa_results],
                    key="elite_mode_select",
                )
            selected = next(r for r in qaoa_results if mode_label(r.get("mode")) == selected_mode_for_elite)
            max_elite = len(selected.get("elite_rows_top_k", []))
            with elite_controls[1]:
                top_k_elite = int(min(int(elite_top_k), max_elite)) if max_elite > 0 else 0
                st.metric("Elite bitstrings shown", top_k_elite)
                st.caption(f"Controlled by the left-sidebar setting: Elite top-k bitstrings = {int(elite_top_k)}.")
            elite_df = pd.DataFrame(selected.get("elite_rows_top_k", [])).head(int(top_k_elite))
            st.dataframe(
                elite_df,
                use_container_width=True,
                column_config={
                    "cost": st.column_config.NumberColumn(format="%.8g"),
                    "frequency": st.column_config.NumberColumn(format="%.5f"),
                },
            )
            if not elite_df.empty and {"bitstring", "frequency"}.issubset(elite_df.columns):
                st.subheader("Frequency of top-k selected elite bitstrings")
                st.bar_chart(elite_df.set_index("bitstring")[["frequency"]])
            if not elite_df.empty and {"bitstring", "cost"}.issubset(elite_df.columns):
                st.subheader("Cost of top-k selected elite bitstrings")
                st.bar_chart(elite_df.set_index("bitstring")[["cost"]])
            st.download_button(
                "Download selected elite bitstrings CSV",
                data=elite_df.to_csv(index=False),
                file_name=f"elite_top_{top_k_elite}_{selected_mode_for_elite.replace(' ', '_')}.csv",
                mime="text/csv",
                key="download_selected_elite_bitstrings_csv",
            )

    with tab_params:
        st.subheader("Final optimal variational parameters")
        params_df = build_variational_parameters_dataframe(results)
        if params_df.empty:
            st.info("No QAOA variational parameters are available. Select and run Monolithic QAOA or Distributed QAOA.")
        else:
            st.caption("These are the final optimized QAOA angles returned by the selected quantum mode(s), grouped by layer.")
            st.dataframe(
                params_df,
                use_container_width=True,
                column_config={
                    "gamma": st.column_config.NumberColumn(format="%.10f"),
                    "beta": st.column_config.NumberColumn(format="%.10f"),
                },
            )
            st.download_button(
                "Download variational parameters CSV",
                data=params_df.to_csv(index=False),
                file_name="dqaoa_variational_parameters.csv",
                mime="text/csv",
                key="download_variational_parameters_tab_csv",
            )
    with tab_figures:
        fig_mode_backends = [
            r.get("mode")
            for r in results
            if r.get("mode") in {"monolithic_qaoa", "telegate_explicit_qaoa"}
        ]
        fig_mode_labels = [mode_label(m) for m in fig_mode_backends]
        if not fig_mode_backends:
            st.info("Select Monolithic QAOA or Distributed QAOA to generate circuit figures.")
        else:
            fig_mode_label = st.selectbox("Figure mode", fig_mode_labels, key="fig_mode_select")
            fig_mode = MODE_LABEL_TO_BACKEND[fig_mode_label]
            fig_type = st.radio(
                "Figure type",
                ["Compact schematic", "Paper-style full layer"],
                horizontal=True,
            )
            max_terms = st.number_input("Max quadratic terms to draw", min_value=1, max_value=50, value=3, step=1)
            show_coeffs = st.checkbox("Show numeric coefficients in figure", value=False)
            if st.button("Generate circuit figure"):
                try:
                    timestamp = int(time.time())
                    out_path = OUTPUT_DIR / f"{fig_mode}_{fig_type.lower().replace(' ', '_')}_{timestamp}.png"
                    H_fig = np.asarray(problem["H"], dtype=float)
                    f_fig = None if problem.get("f") is None else np.asarray(problem["f"], dtype=float)
                    c0_fig = float(problem.get("c0", 0.0))
                    name_fig = str(problem.get("name", "qubo"))
                    caps_fig = problem.get("capacities")
                    manual_fig = problem.get("manual_assignment")
                    nq_fig = int(problem.get("num_qpus", 2))

                    if fig_type == "Compact schematic":
                        meta = dq.save_qubo_mode_paper_schematic(
                            H=H_fig,
                            f=f_fig,
                            c0=c0_fig,
                            name=name_fig,
                            mode=fig_mode,
                            p=int(p_max),
                            num_qpus=nq_fig,
                            capacities=caps_fig,
                            manual_assignment=manual_fig,
                            output_path=str(out_path),
                            dpi=300,
                            show_coeff_values=bool(show_coeffs),
                            include_final_measurements=False,
                        )
                    else:
                        if fig_mode == "monolithic_qaoa":
                            meta = dq.save_monolithic_qaoa_style_circuit_figure(
                                H=H_fig,
                                f=f_fig,
                                c0=c0_fig,
                                name=name_fig,
                                p=int(p_max),
                                output_path=str(out_path),
                                dpi=300,
                                show_numeric_coeffs=bool(show_coeffs),
                                max_quadratic_terms_to_draw=int(max_terms),
                            )
                        elif fig_mode == "telegate_explicit_qaoa":
                            meta = dq.save_dqaoa_style_qaoa_circuit_figure(
                                H=H_fig,
                                f=f_fig,
                                c0=c0_fig,
                                name=name_fig,
                                p=int(p_max),
                                num_qpus=nq_fig,
                                capacities=caps_fig,
                                manual_assignment=manual_fig,
                                output_path=str(out_path),
                                dpi=300,
                                show_numeric_coeffs=bool(show_coeffs),
                                max_quadratic_terms_to_draw=int(max_terms),
                            )
                        else:
                            meta = dq.save_qubo_mode_paper_schematic(
                                H=H_fig,
                                f=f_fig,
                                c0=c0_fig,
                                name=name_fig,
                                mode=fig_mode,
                                p=int(p_max),
                                num_qpus=nq_fig,
                                capacities=caps_fig,
                                manual_assignment=manual_fig,
                                output_path=str(out_path),
                                dpi=300,
                                show_coeff_values=bool(show_coeffs),
                                include_final_measurements=False,
                            )

                    st.image(str(out_path), caption=str(out_path.name), use_container_width=True)
                    with open(out_path, "rb") as fh:
                        st.download_button(
                            "Download figure PNG",
                            data=fh.read(),
                            file_name=out_path.name,
                            mime="image/png",
                            key=f"download_figure_png_{fig_mode}_{out_path.stem}",
                        )
                    st.json(to_jsonable(meta))
                except Exception as exc:
                    st.error(f"Figure generation failed: {type(exc).__name__}: {exc}")

    with tab_export:
        json_text = json.dumps(to_jsonable(results), indent=2)
        st.download_button(
            "Download results JSON",
            data=json_text,
            file_name="dqaoa_qubo_results.json",
            mime="application/json",
            key="download_results_json_export",
        )
        st.download_button(
            "Download clean summary CSV",
            data=df.to_csv(index=False),
            file_name="dqaoa_qubo_clean_summary.csv",
            mime="text/csv",
            key="download_clean_summary_csv_export",
        )
        plot_metrics_df = build_plot_metrics_dataframe(results)
        st.download_button(
            "Download full plot metrics CSV",
            data=plot_metrics_df.to_csv(index=False),
            file_name="dqaoa_qubo_plot_metrics.csv",
            mime="text/csv",
            key="download_plot_metrics_csv_export",
        )
        params_export_df = build_variational_parameters_dataframe(results)
        if not params_export_df.empty:
            st.download_button(
                "Download variational parameters CSV",
                data=params_export_df.to_csv(index=False),
                file_name="dqaoa_variational_parameters.csv",
                mime="text/csv",
                key="download_variational_parameters_export_csv",
            )
        st.download_button(
            "Download current H CSV",
            data=matrix_to_csv_bytes(np.asarray(problem["H"], dtype=float)),
            file_name="current_H.csv",
            mime="text/csv",
            key="download_current_H_csv_export",
        )
        st.download_button(
            "Download current f CSV",
            data=vector_to_csv_bytes(np.asarray(problem["f"], dtype=float)),
            file_name="current_f.csv",
            mime="text/csv",
            key="download_current_f_csv_export",
        )
        st.subheader("Raw JSON")
        st.code(json_text[:20000] + ("\n... [truncated in display]" if len(json_text) > 20000 else ""), language="json")
else:
    st.info("Configure a QUBO and click **Run DQAOA-QUBO demo** to generate results.")
