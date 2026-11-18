from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import solver_backend as dq


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def main(job_path: str, result_path: str) -> int:
    job_file = Path(job_path)
    result_file = Path(result_path)
    try:
        with job_file.open("r", encoding="utf-8") as fh:
            job = json.load(fh)

        H = np.asarray(job["H"], dtype=float)
        f = np.asarray(job["f"], dtype=float)

        train_cfg = dq.TrainConfig(
            shots_train=int(job["shots_train"]),
            avg_k=int(job["avg_k"]),
            seed_trans=1234,
            seed_base=int(job["seed"]),
            optimization_level=1,
            sim_method=str(job["sim_method"]),
            batch_evaluations=True,
        )
        obj_cfg = dq.QuboObjectiveConfig(objective_name="mean_cost", delta_close_cost=0.0)
        analysis_cfg = dq.AnalysisConfig(elite_top_k=int(job.get("elite_top_k", 10)), certificate_tol=1e-9)
        adam_cfg = dq.AdamConfig(
            iters=int(job["adam_iters"]),
            lr=float(job["adam_lr"]),
            fd_eps0=float(job["fd_eps0"]),
            seed=int(job["seed"]),
            eval_at_each_iter=bool(job.get("eval_at_each_iter", True)),
        )
        multistart_cfg = dq.MultiStartConfig(
            num_random_starts=int(job["num_random_starts"]),
            warm_start_perturbations=int(job["warm_start_perturbations"]),
            warm_start_sigma=0.10,
            add_plain_warm_start=True,
            keep_top_k_per_depth=3,
            select_by_exact_when_available=False,
            parallel_restarts=int(job["parallel_restarts"]),
        )
        convergence_cfg = dq.ConvergenceConfig()

        results = dq.compare_qubo_modes(
            H=H,
            f=f,
            c0=float(job["c0"]),
            name=str(job["name"]),
            modes=list(job["modes"]),
            p_max=int(job["p_max"]),
            num_qpus=int(job["num_qpus"]),
            capacities=job.get("capacities"),
            manual_assignment=job.get("manual_assignment"),
            train_cfg=train_cfg,
            obj_cfg=obj_cfg,
            analysis_cfg=analysis_cfg,
            final_shots=int(job["final_shots"]),
            adam_cfg=adam_cfg,
            multistart_cfg=multistart_cfg,
            cplex_time_limit=job.get("cplex_time_limit"),
            verbose=False,
            return_full_output=False,
            store_history=bool(job["store_history"]),
            store_final_counts=bool(job["store_final_counts"]),
            compute_exact_if_small=bool(job["compute_exact_if_small"]),
            exact_threshold_n=int(job["exact_threshold_n"]),
            certificate_lower_bound=None,
            convergence_cfg=convergence_cfg,
        )

        payload = {"status": "done", "results": _to_jsonable(results)}
        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return 0

    except Exception as exc:
        payload = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m dqaoa_streamlit_demo.job_runner JOB_JSON RESULT_JSON", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
