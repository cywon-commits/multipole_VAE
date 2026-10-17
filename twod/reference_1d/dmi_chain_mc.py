import numpy as np
from heisenberg_chain_estimators import sample_vmf, langevin, score_matching, pseudo_lik, names, P

def heff(S, i, J, D):
    # local field on site i (open chain), W = sum_i J S_i.S_{i+1} + D.(S_i x S_{i+1})
    N = S.shape[1]; h = np.zeros((S.shape[0],3))
    if i+1 < N: h += J*S[:,i+1] + np.cross(S[:,i+1], D)
    if i-1 >= 0: h += J*S[:,i-1] + np.cross(D, S[:,i-1])
    return h

def sweep(S, J, D, rng):
    N = S.shape[1]
    for par in (0,1):                       # checkerboard: sites of one parity are independent
        for i in range(par, N, 2):
            h = heff(S,i,J,D); k = np.linalg.norm(h,axis=1)
            S[:,i] = sample_vmf(h/k[:,None], k, rng) if np.all(k>1e-9) else S[:,i]
        for i in range(par, N, 2):          # over-relaxation (microcanonical reflection)
            h = heff(S,i,J,D); hh = h/np.linalg.norm(h,axis=1,keepdims=True)
            S[:,i] = 2*(S[:,i]*hh).sum(1,keepdims=True)*hh - S[:,i]
    return S

def run(J, D, N=100, chains=200, therm=300, nsamp=4, gap=50, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(chains,N,3)); S /= np.linalg.norm(S,axis=2,keepdims=True)
    for _ in range(therm): sweep(S,J,D,rng)
    out = []
    for _ in range(nsamp):
        for _ in range(gap): sweep(S,J,D,rng)
        out.append(S.copy())
    return np.concatenate(out)          # (chains*nsamp, N, 3)
