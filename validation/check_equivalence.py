"""Exact branch-summed circuit verification plus independent Aer shot comparison.
Usage: python check_equivalence.py --repo PATH_TO_DQAOA
No optimization or classical reference is used.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse,sys,json,csv,hashlib,platform
from pathlib import Path
import numpy as np
from qiskit.quantum_info import DensityMatrix, Statevector, Operator, partial_trace
from qiskit import transpile
from qiskit_aer import AerSimulator


def exact_dynamic(qc):
    """Unnormalized density branches; marginalize a classical bit after its last use.
    All measurement outcomes are summed, never postselected.
    Supports the actual builder's unitary gates, reset, measurement and if_else.
    Final data measurements must be absent (measure=False).
    """
    last={}
    for t,ci in enumerate(qc.data):
        if ci.operation.name=='if_else':
            bit,val=ci.operation.condition
            last[qc.find_bit(bit).index]=t
    branches={():DensityMatrix.from_label('0'*qc.num_qubits).data}
    for t,ci in enumerate(qc.data):
        op=ci.operation; qs=[qc.find_bit(q).index for q in ci.qubits]
        updated={}
        def add(key,rho):
            # Keep only measurement outcomes needed by future conditional gates.
            key=tuple((b,v) for b,v in key if last.get(b,-1)>t)
            updated[key]=updated.get(key,0)+rho
        for key,rho in branches.items():
            if op.name=='measure':
                b=qc.find_bit(ci.clbits[0]).index
                for v in [0,1]:
                    mask=(((np.arange(len(rho))>>qs[0])&1)==v)
                    projected=rho*mask[:,None]*mask[None,:]
                    add(tuple(sorted(dict(key,**{}).items()))+((b,v),),projected)
            elif op.name=='reset': add(key,DensityMatrix(rho).reset(qs).data)
            elif op.name=='if_else':
                bit,val=op.condition; b=qc.find_bit(bit).index
                which=0 if dict(key)[b]==val else 1
                if which<len(op.blocks) and op.blocks[which] is not None:
                    block=op.blocks[which]
                    for inner in block.data:
                        subq=[qs[block.find_bit(q).index] for q in inner.qubits]
                        rho=DensityMatrix(rho).evolve(Operator(inner.operation),qargs=subq).data
                add(key,rho)
            elif op.name=='barrier': add(key,rho)
            else: add(key,DensityMatrix(rho).evolve(Operator(op),qargs=qs).data)
        branches=updated
    assert all(not key for key in branches)
    return DensityMatrix(sum(branches.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--shots',type=int,default=4096);p.add_argument('--output',type=Path,default=Path(__file__).parent/'results');a=p.parse_args()
    repo=a.repo.resolve();sys.path.insert(0,str(repo/'gui'))
    from dqaoa_streamlit_demo import solver_backend as dq
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    rows=[]; records=[]
    for name in ['n03_ring','n04_sparse','n05_dense']:
        folder=repo/'examples'/'qubo_txt'/name
        H=np.loadtxt(folder/'H.txt',delimiter=',');f=np.loadtxt(folder/'f.txt',delimiter=',').reshape(-1);n=len(f)
        inst=dq.canonicalize_qubo_from_dense(H,f,.25,name)
        # Explicitly export all coefficients and rotations, including constant offset.
        h=-inst.linear/2
        for (i,j),b in inst.quadratic.items(): h[i]-=b/4;h[j]-=b/4
        alpha=.25+sum(inst.linear)/2+sum(inst.quadratic.values())/4
        for depth in [1,2]:
            gammas=[.37,.81][:depth];betas=[.23,.49][:depth]
            mono={i:(0,i) for i in range(n)}
            qc_m,gm,bm,_=dq.build_parametrized_qubo_qaoa_circuit(inst,depth,mono,measure=False,remote_mode='abstract')
            bind=dict(zip(gm,gammas));bind.update(zip(bm,betas)); qc_m=qc_m.assign_parameters({k:v for k,v in bind.items() if k in qc_m.parameters})
            state=Statevector.from_instruction(qc_m);rho_m=DensityMatrix(state).data; pm=state.probabilities()
            # Independent diagonal-phase reference from C(x), not from the circuit's Ising construction.
            from qiskit import QuantumCircuit
            ref=np.ones(2**n,dtype=complex)/np.sqrt(2**n)
            z=((np.arange(2**n)[:,None]>>np.arange(n))&1)
            costs=np.einsum('bi,ij,bj->b',z,H,z)+z@f+.25
            for g,b in zip(gammas,betas):
                ref*=np.exp(-1j*g*costs)
                mix=QuantumCircuit(n);mix.rx(2*b,range(n)); ref=Statevector(ref).evolve(mix).data
            direct_error=float(np.max(np.abs(np.outer(ref,ref.conj())-rho_m)))
            for kind,assign in [('single',[0]*n),('two',[i%2 for i in range(n)]),('multi',list(range(n)))]:
                alloc={i:(q,sum(v==q for v in assign[:i])) for i,q in enumerate(assign)}
                qc,gp,bp,stats=dq.build_parametrized_qubo_qaoa_circuit(inst,depth,alloc,measure=False,remote_mode='telegate_explicit')
                bind=dict(zip(gp,gammas));bind.update(zip(bp,betas));qc=qc.assign_parameters({k:v for k,v in bind.items() if k in qc.parameters})
                rho=partial_trace(exact_dynamic(qc),list(range(n,n+2))).data
                pd=np.diag(rho).real
                assert abs(np.trace(rho)-1)<1e-10
                # Compare shot distributions using the same fixed parameters; decode complete output.
                qc.measure(range(n),range(n));m=qc_m.copy();m.measure_all()
                backend=AerSimulator(method='matrix_product_state',max_parallel_threads=1)
                counts=[]
                for circ,seed,explicit in [(m,1701,False),(qc,2903,True)]:
                    compiled=transpile(circ,backend,optimization_level=1,seed_transpiler=1234)
                    c=backend.run(compiled,shots=a.shots,seed_simulator=seed).result().get_counts()
                    counts.append(dq.decoded_distribution_from_counts(c,n,explicit_telegate=explicit))
                keys=[format(i,f'0{n}b')[::-1] for i in range(2**n)]
                obs=[np.array([d.get(k,0.) for k in keys]) for d in counts]
                shot_tvd=float(np.abs(obs[0]-obs[1]).sum()/2)
                # Calibrate observed TVD under the exact shared distribution; 2000 parametric null draws.
                rng=np.random.default_rng(991+len(rows));pnull=np.maximum(pm,0);pnull/=pnull.sum()
                null=np.abs(rng.multinomial(a.shots,pnull,size=2000)-rng.multinomial(a.shots,pnull,size=2000)).sum(axis=1)/(2*a.shots)
                row=dict(case=name,p=depth,allocation=kind,qpus=len(set(assign)),remote_cx=stats['remote_cx'],exact_tvd=float(np.abs(pm-pd).sum()/2),max_probability_error=float(np.max(np.abs(pm-pd))),max_density_entry_error=float(np.max(np.abs(rho_m-rho))),independent_cost_phase_error=direct_error,shot_tvd=shot_tvd,null_tvd_95=float(np.quantile(null,.95)),null_tail_fraction=float((1+sum(null>=shot_tvd))/(len(null)+1)))
                rows.append(row)
                records.append(dict(**row,H=H.tolist(),f=f.tolist(),c0=.25,gammas=gammas,betas=betas,assignment=assign,alpha=float(alpha),h=h.tolist(),J={f'{i},{j}':v/4 for (i,j),v in inst.quadratic.items()},rz_angles=[(2*g*h).tolist() for g in gammas],rzz_angles=[{f'{i},{j}':g*v/2 for (i,j),v in inst.quadratic.items()} for g in gammas],rx_angles=[2*b for b in betas],bitstrings=keys,monolithic_probabilities=pm.tolist(),distributed_probabilities=pd.tolist(),sampled_monolithic=obs[0].tolist(),sampled_distributed=obs[1].tolist()))
                print(row,flush=True)
    (out/'full_distributions.json').write_text(json.dumps(records,indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    import qiskit,qiskit_aer
    (out/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,shots=a.shots,transpiler_seed=1234,monolithic_seed=1701,distributed_seed=2903,backend_sha256=hashlib.sha256(Path(dq.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    assert max(r['exact_tvd'] for r in rows)<1e-10
    assert max(r['independent_cost_phase_error'] for r in rows)<1e-10
if __name__=='__main__':main()
