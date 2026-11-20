"""Five cold-cache, fixed-workload timings of the current solver (not historical stages)."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,sys,time,json,csv,platform
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent/'timing_results');a=p.parse_args()
sys.path.insert(0,str(a.repo.resolve()/'gui'))
from dqaoa_streamlit_demo import solver_backend as dq
folder=a.repo/'examples/qubo_txt/n03_ring';H=np.loadtxt(folder/'H.txt',delimiter=',');f=np.loadtxt(folder/'f.txt',delimiter=',')
a.output.mkdir(parents=True,exist_ok=True)
rows=[]
for repeat in range(5):
 for mode in ['monolithic_qaoa','telegate_explicit_qaoa']:
  dq.clear_circuit_runner_cache()
  tr=dq.TrainConfig(shots_train=128,avg_k=1,seed_trans=1234,seed_base=7,optimization_level=1,sim_method='matrix_product_state',batch_evaluations=True)
  adam=dq.AdamConfig(iters=6,lr=.08,seed=7)
  ms=dq.MultiStartConfig(num_random_starts=2,keep_top_k_per_depth=3,select_by_exact_when_available=False,parallel_restarts=1)
  start=time.perf_counter()
  r=dq.solve_qubo_mode(H,f,.25,name='n03_ring',mode=mode,p_max=1,num_qpus=2,capacities=[2,1],manual_assignment=[0,1,0],train_cfg=tr,adam_cfg=adam,multistart_cfg=ms,final_shots=512,compute_exact_if_small=False)
  rows.append(dict(mode=mode,repeat=repeat+1,wall_seconds=time.perf_counter()-start,best_cost=r['best_cost']))
  print(rows[-1],flush=True)
with (a.output/'per_run.csv').open('w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for mode in ['monolithic_qaoa','telegate_explicit_qaoa']:
 times=[r['wall_seconds'] for r in rows if r['mode']==mode]
 summary.append(dict(mode=mode,repeats=5,mean_seconds=np.mean(times),sd_seconds=np.std(times,ddof=1),median_seconds=np.median(times)))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import qiskit,qiskit_aer,psutil
cpu=platform.processor()
if Path('/proc/cpuinfo').exists():cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),cpu)
(a.output/'protocol.json').write_text(json.dumps(dict(case='n03_ring',p_max=1,train_shots=128,candidate_final_shots=512,adam_iters=6,restarts=2,learning_rate=.08,seed=7,seed_trans=1234,parallel_workers=1,thread_limits={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},cache='cleared before every timed call',includes='canonicalization, allocation, construction, transpilation, optimization, candidate sampling, selection, result construction',excludes='Python imports, input loading, plotting, GUI, classical reference, independent holdout sampling',interpretation='Current implementation only; does not isolate historical stage effects or establish quantum hardware runtime.',python=sys.version,os=platform.platform(),processor=cpu,logical_cpus=os.cpu_count(),host_visible_memory_bytes=psutil.virtual_memory().total,numpy=np.__version__,qiskit=qiskit.__version__,aer=qiskit_aer.__version__),indent=2)+'\n')
