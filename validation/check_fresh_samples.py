"""Three seeds on three small cases; independent holdout shots after selection.
This is an additional validation experiment, not the original paper benchmark.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,sys,json,csv,time,platform,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent/'fresh_results');a=p.parse_args();repo=a.repo.resolve();sys.path.insert(0,str(repo/'gui'))
 from dqaoa_streamlit_demo import solver_backend as dq
 out=a.output;out.mkdir(parents=True,exist_ok=True);rows=[];raw=[]
 for name in ['n03_ring','n04_sparse','n05_dense']:
  folder=repo/'examples/qubo_txt'/name;H=np.loadtxt(folder/'H.txt',delimiter=',');f=np.loadtxt(folder/'f.txt',delimiter=',').reshape(-1);n=len(f)
  inst=dq.canonicalize_qubo_from_dense(H,f,.25,name)
  assign=[i%2 for i in range(n)];caps=[assign.count(i) for i in range(2)]
  for seed in [7,19,42]:
   for mode in ['monolithic_qaoa','telegate_explicit_qaoa']:
    dq.clear_circuit_runner_cache()
    tr=dq.TrainConfig(shots_train=128,avg_k=1,seed_trans=1234,seed_base=seed,optimization_level=1,sim_method='matrix_product_state',batch_evaluations=True)
    adam=dq.AdamConfig(iters=6,lr=.08,seed=seed)
    ms=dq.MultiStartConfig(num_random_starts=2,warm_start_perturbations=1,warm_start_sigma=.10,keep_top_k_per_depth=3,select_by_exact_when_available=False,parallel_restarts=1)
    res=dq.solve_qubo_mode(H=H,f=f,c0=.25,name=name,mode=mode,p_max=1,num_qpus=2,capacities=caps,manual_assignment=assign,train_cfg=tr,adam_cfg=adam,multistart_cfg=ms,final_shots=512,compute_exact_if_small=False,return_full_output=True)
    actual=res['details']['allocation'];remote=res['details']['remote_mode']
    runner=dq.get_cached_qubo_runner(inst,res['p'],actual,remote,tr,measure=True)
    hold_seed=100000+seed
    counts=runner.run_counts(res['gammas'],res['betas'],shots=2048,seed_sim=hold_seed)
    d=dq.decoded_distribution_from_counts(counts,n,explicit_telegate=mode=='telegate_explicit_qaoa')
    # Reference only now, after parameters frozen and independent sampling complete.
    ref=dq.solve_qubo_bruteforce(inst)
    costs={bs:dq.qubo_value_from_bitstring(bs,inst) for bs in d}
    mean=sum(d[b]*costs[b] for b in d);popt=sum(d[b] for b in d if abs(costs[b]-ref['best_cost'])<1e-9)
    row=dict(case=name,mode=mode,seed=seed,holdout_seed=hold_seed,selected_best_cost=res['best_cost'],holdout_best_cost=min(costs.values()),reference_cost=ref['best_cost'],holdout_mean_cost=mean,holdout_optimal_probability=popt,holdout_shots=2048)
    rows.append(row);raw.append(dict(**row,gammas=res['gammas'],betas=res['betas'],assignment=res['details']['assignment'],training=asdict(tr),optimizer=asdict(adam),multistart=asdict(ms),p_max=1,candidate_shots=512,compute_exact_if_small=False,decoded_holdout_distribution=d))
    print(row,flush=True)
 with (out/'per_seed.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 grouped=[]
 for name in ['n03_ring','n04_sparse','n05_dense']:
  for mode in ['monolithic_qaoa','telegate_explicit_qaoa']:
   rr=[r for r in rows if r['case']==name and r['mode']==mode];means=[r['holdout_mean_cost'] for r in rr];probs=[r['holdout_optimal_probability'] for r in rr]
   grouped.append(dict(case=name,mode=mode,seeds=3,mean_cost=float(np.mean(means)),mean_cost_seed_sd=float(np.std(means,ddof=1)),optimal_probability=float(np.mean(probs)),optimal_probability_seed_sd=float(np.std(probs,ddof=1)),holdout_optimum_runs=sum(abs(r['holdout_best_cost']-r['reference_cost'])<1e-9 for r in rr)))
 with (out/'summary.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=grouped[0]);w.writeheader();w.writerows(grouped)
 (out/'details.json').write_text(json.dumps(raw,indent=2)+'\n')
 import qiskit,qiskit_aer,psutil
 (out/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,logical_cpus=os.cpu_count(),memory_bytes=psutil.virtual_memory().total,backend_sha256=hashlib.sha256(Path(dq.__file__).read_bytes()).hexdigest(),thread_limits={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},parallel_restarts=1),indent=2)+'\n')
if __name__=='__main__':main()
