"""Regenerate the supplied H.txt and f.txt values in a separate destination."""
from pathlib import Path
import argparse
import numpy as np

def generate(destination):
    for n in range(3,16):
        for k,family in enumerate(['ring','sparse','dense']):
            rng=np.random.default_rng(20261118+100*n+k)
            H=np.zeros((n,n)); f=rng.integers(-12,13,n)/10
            for i in range(n):
                H[i,i]=rng.integers(-4,5)/10
                for j in range(i+1,n):
                    edge=(j==i+1 or (i==0 and j==n-1))
                    if family=='dense' or edge or (family=='sparse' and rng.random()<.20):
                        w=int(rng.choice([v for v in range(-12,13) if v]))/10
                        H[i,j]=H[j,i]=w/2
            folder=destination/f'n{n:02d}_{family}'; folder.mkdir(parents=True,exist_ok=True)
            np.savetxt(folder/'H.txt',H,delimiter=',',fmt='%.3f')
            np.savetxt(folder/'f.txt',f[None,:],delimiter=',',fmt='%.3f')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    generate(parser.parse_args().destination)
