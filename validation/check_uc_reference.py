"""Independent per-period enumeration for the simplified UC model in Eq. (21).
No ADMM or QAOA code is imported. Convex dispatch uses marginal-cost bisection.
"""
from pathlib import Path
import itertools,json
import numpy as np
A=np.array([185,170,275,220,150.]); B=np.array([25,21,22,24.5,24.7]); C=np.array([.058,.016,.0165,.011,.007])
lo=np.array([10,5,5,10,20.]);hi=np.array([80,50,60,85,110.]);demand=[60.,130.,280.]
rows=[]
for L in demand:
 candidates=[]
 for bits in itertools.product([0,1],repeat=5):
  z=np.array(bits); lower=lo*z;upper=hi*z
  if lower.sum()>L+1e-9 or upper.sum()<L-1e-9: continue
  left=float(np.min(B+2*C*lower)-1);right=float(np.max(B+2*C*upper)+1)
  for _ in range(120):
   mid=(left+right)/2;p=np.clip((mid-B)/(2*C),lower,upper)
   if p.sum()<L:left=mid
   else:right=mid
  p=np.clip(((left+right)/2-B)/(2*C),lower,upper)
  candidates.append(dict(z=list(bits),p=p.tolist(),cost=float(A@z+B@p+C@(p*p)),balance_error=float(abs(p.sum()-L)),bound_violation=float(max(0,np.max(lower-p),np.max(p-upper)))))
 best=min(candidates,key=lambda x:x['cost']);best['demand']=L;best['feasible_patterns']=len(candidates);rows.append(best)
result=dict(periods=rows,total_cost=sum(r['cost'] for r in rows),patterns_enumerated_per_period=32,model_scope='No intertemporal coupling; balance and generation bounds only.')
p=Path(__file__).parent/'uc_reference.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
