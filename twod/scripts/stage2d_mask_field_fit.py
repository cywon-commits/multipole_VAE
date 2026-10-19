"""Stage 2d-A: shared-field compromise in the M1 mask decoder (no VAE; global parameters only).

Model: interfacial SkX (J = D = 1, beta = 6, B = 0.4). Unmasked sites: exact pseudo-likelihood with a uniform field h_z;
masked sites (M1): independent vMF with field h_m. Fit (J, D, h_z) with
   shared:   h_m = h_z (what a mask-blind decoder with collapsed z must do)
   separate: h_m free (the proposed fix)
Chat run (L = 24, 144 samples, aligned random blocks):
   shared   -> dJ/J = +3.5..+5.1 %, dD/D = -11.4..-11.8 %, h_z = 1.50..1.57  (pulled to the masked optimum)
   separate -> |dJ/J| <= 0.7 %, |dD/D| <= 1.0 %, h_z = 2.40..2.45, h_m = 1.48 = L^{-1}(<S_z>_masked)
usage: python stage2d_mask_field_fit.py [--quick]
"""
from _common import *
from scipy.optimize import minimize

a = parse(__doc__)
L = 24 if a.quick else 48
ch, ns = (24, 6) if a.quick else (40, 8)
beta = 6.0
make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, 0.4)).scaled(b)
S0 = anneal(make, L, ch, np.linspace(0.5, beta, 12), 40 if a.quick else 100, seed=a.seed + 1)
S, _ = run_mc(make(beta), L, chains=ch, therm=100, nsamp=ns, gap=20, seed=a.seed + 2, init=S0)
rng = np.random.default_rng(a.seed)

def masks(ell, rho):
    nb = L//ell; M = np.zeros((len(S), L, L), bool)
    for s in range(len(S)):
        for b in rng.permutation(nb*nb)[:int(round(rho*nb*nb))]:
            i, j = divmod(b, nb); M[s, i*ell:(i+1)*ell, j*ell:(j+1)*ell] = True
    return M

def loss(p, M, shared=True):
    J, D, hz = p[:3]; hm = hz if shared else p[3]
    h = local_field(S, Hamiltonian.heisenberg_dmi(J, D, 'interfacial', B=(0, 0, hz)))
    k = np.linalg.norm(h, axis=-1)
    return np.where(M, log_A(abs(hm)) - hm*S[..., 2], log_A(k) - (h*S).sum(-1)).sum()/len(S)

opt = dict(xatol=1e-5, fatol=1e-6, maxiter=6000)
base = minimize(loss, [6.0, 6.0, 2.4], args=(np.zeros((len(S), L, L), bool),), method='Nelder-Mead', options=opt).x
print(f"no mask: J={base[0]:.4f} D={base[1]:.4f} hz={base[2]:.4f}")
for ell in (4, 8):
    for rho in (0.25, 0.5):
        M = masks(ell, rho)
        r = minimize(loss, base, args=(M,), method='Nelder-Mead', options=opt).x
        r2 = minimize(loss, np.r_[base, 1.5], args=(M, False), method='Nelder-Mead', options=opt).x
        print(f"ell={ell} rho={rho}: shared dJ/J={100*(r[0]/base[0]-1):+.2f}% dD/D={100*(r[1]/base[1]-1):+.2f}% hz={r[2]:.3f} | "
              f"separate dJ/J={100*(r2[0]/base[0]-1):+.2f}% dD/D={100*(r2[1]/base[1]-1):+.2f}% hz={r2[2]:.3f} h_m={r2[3]:.3f}", flush=True)
