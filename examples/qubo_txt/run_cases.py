"""Run selected text cases with the repository GUI backend, then evaluate references."""
from pathlib import Path
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ['n03_ring', 'n03_sparse', 'n03_dense', 'n04_ring', 'n05_sparse', 'n06_ring', 'n08_ring', 'n12_ring']

def reference(H, f, c0):
    import numpy as np
    n=len(f)
    z=((np.arange(1 << n)[:,None] >> np.arange(n-1,-1,-1)) & 1).astype(float)
    costs=np.einsum('bi,ij,bj->b',z,H,z)+z@f+c0
    minimum=float(costs.min())
    ids=np.flatnonzero(np.abs(costs-minimum)<1e-9)
    return {'best_cost':minimum,'best_bitstring':format(int(ids[0]),f'0{n}b'),'number_of_minimizers':len(ids)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=DEFAULT_CASES)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--gui',type=Path,default=ROOT.parents[1]/'gui')
    parser.add_argument('--output',type=Path,default=ROOT/'local_results')
    parser.add_argument('--timeout',type=int,default=600,help='Per-case limit in seconds (both quantum modes).')
    args=parser.parse_args()
    gui=args.gui.resolve()
    backend=gui/'dqaoa_streamlit_demo'/'solver_backend.py'
    if not backend.exists(): parser.error(f'GUI backend not found: {backend}. Copy examples into the DQAOA repository or use --gui PATH.')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key]='1'
    env['PYTHONPATH']=str(gui)+os.pathsep+env.get('PYTHONPATH','')
    import numpy as np
    import psutil
    cpu=platform.processor()
    if Path('/proc/cpuinfo').exists():
        cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),cpu)
    meta={'python':sys.version,'os':platform.platform(),'processor':cpu,'logical_cpus':os.cpu_count(),
          'host_visible_memory_bytes':psutil.virtual_memory().total,
          'versions':{p:importlib.metadata.version(p) for p in ['numpy','qiskit','qiskit-aer','pandas','matplotlib']},
          'backend_sha256':hashlib.sha256(backend.read_bytes()).hexdigest(),
          'job_runner_sha256':hashlib.sha256((backend.parent/'job_runner.py').read_bytes()).hexdigest(),
          'threads':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},
          'parallel_restarts':1,'concurrent_cases':1,'timing':'One end-to-end subprocess measurement per case, including both modes and imports; no reference computation. Classical simulation only.',
          'sampling_note':'Backend final samples participate in candidate selection; these are not independent held-out performance estimates.'}
    (out/'environment.txt').write_text(json.dumps(meta,indent=2)+'\n')
    cases=sorted(p.name for p in ROOT.glob('n*_*') if p.is_dir()) if args.all else args.cases
    rows=[]
    for name in cases:
        folder=ROOT/name
        cfg=json.loads((folder/'settings.txt').read_text())
        assert cfg['select_by_exact_when_available'] is False and cfg['compute_exact_if_small'] is False
        H=np.loadtxt(folder/'H.txt',delimiter=','); f=np.loadtxt(folder/'f.txt',delimiter=',').reshape(-1)
        assert H.shape==(cfg['n'],cfg['n']) and len(f)==cfg['n']
        assert sum(cfg['capacities'])==cfg['n']
        job=dict(cfg,H=H.tolist(),f=f.tolist())
        jobpath=out/(name+'_job.txt'); resultpath=out/(name+'_raw.tmp')
        jobpath.write_text(json.dumps(job,indent=2)+'\n')
        print('Running',name,flush=True); t0=time.perf_counter()
        with (out/(name+'_log.txt')).open('w') as log:
            try:
                proc=subprocess.run([sys.executable,'-m','dqaoa_streamlit_demo.job_runner',str(jobpath),str(resultpath)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                payload=json.loads(resultpath.read_text()) if resultpath.exists() else {'status':'error','error_message':f'exit code {proc.returncode}'}
            except subprocess.TimeoutExpired:
                payload={'status':'timeout','error_message':f'exceeded {args.timeout} seconds'}
        wall=time.perf_counter()-t0
        # Classical enumeration happens only after quantum execution and is never passed to the solver.
        ref=reference(H,f,cfg['c0'])
        report={'case':name,'settings':cfg,'wall_seconds_both_modes':wall,'classical_reference':ref,'status':payload['status'],'results':[]}
        for r in payload.get('results',[]):
            clean={k:r.get(k) for k in ['mode','status','best_bitstring','best_cost','p','gammas','betas','angles','runtime_to_solution_seconds','remote_cx_count','cross_qpu_terms'] if k in r}
            if clean.get('best_cost') is not None:
                bits=np.array([int(b) for b in clean['best_bitstring']])
                cost=float(bits@H@bits+f@bits+cfg['c0'])
                assert abs(cost-clean['best_cost'])<1e-8
                clean['gap_to_classical_reference']=cost-ref['best_cost']
                clean['matches_reference_cost']=abs(cost-ref['best_cost'])<1e-9
            report['results'].append(clean)
            rows.append({'case':name,'mode':r['mode'],'status':r.get('status'),'best_bitstring':r.get('best_bitstring'),'best_cost':r.get('best_cost'),'reference_cost':ref['best_cost'],'gap':clean.get('gap_to_classical_reference'),'runtime_to_solution_seconds':r.get('runtime_to_solution_seconds')})
        if not report['results']:
            report['error']=payload.get('error_message','Unknown failure')
            rows.append({'case':name,'mode':'both','status':payload['status']})
        (out/(name+'_results.txt')).write_text(json.dumps(report,indent=2)+'\n')
        if resultpath.exists(): resultpath.unlink()
        with (out/'summary.txt').open('w',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=['case','mode','status','best_bitstring','best_cost','reference_cost','gap','runtime_to_solution_seconds'],delimiter='\t'); writer.writeheader(); writer.writerows(rows)
        print(name,payload['status'],round(wall,2),'seconds',flush=True)
    return 1 if any(r.get('best_cost') is None for r in rows) else 0

if __name__=='__main__':
    raise SystemExit(main())
