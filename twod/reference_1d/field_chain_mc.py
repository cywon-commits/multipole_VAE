import numpy as np, time
from heisenberg_chain_estimators import sample_vmf, score_matching, pseudo_lik, names, P, langevin
from scipy.optimize import brentq
n = np.array([0,0,1.0])
def sweep(S, J, kap, rng):
    N=S.shape[1]
    for par in (0,1):
        idx=np.arange(par,N,2)
        h = np.zeros((S.shape[0],len(idx),3)) + kap*n
        h[:, idx+1<N] += J*S[:, (idx+1)[idx+1<N]]
        h[:, idx-1>=0] += J*S[:, (idx-1)[idx-1>=0]]
        k=np.linalg.norm(h,axis=2); hh=h/k[...,None]
        new = sample_vmf(hh.reshape(-1,3), k.reshape(-1), rng).reshape(h.shape)
        S[:,idx]=new
    return S
def run(J,kap,N=100,C=200,therm=200,nsamp=5,gap=10,seed=0):
    rng=np.random.default_rng(seed); S=np.tile(n,(C,N,1)).astype(float)
    for _ in range(therm): sweep(S,J,kap,rng)
    out=[]
    for _ in range(nsamp):
        for _ in range(gap): sweep(S,J,kap,rng)
        out.append(S.copy())
    return np.concatenate(out)
