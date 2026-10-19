"""Stage 2e-C: the soft (h_z, J, D) valley of the unmasked pseudo-likelihood (interfacial SkX, J = D = 1, beta = 6, B = 0.4).
Fix h_z = h0 - delta, refit (J, D): the slopes dJ/J and dD/D per unit field deficit define the valley.
Chat run (L = 24, 144 samples): dJ/J = +4.6..+4.8 %/unit, dD/D = -13.3..-13.4 %/unit;
Stage 2d M1 regression (20 runs): +5.1 %/unit (intercept +0.19 %), -13.9 %/unit (intercept -0.29 %).
usage: python stage2e_valley.py [--quick]   (full: L = 48)"""
from _common import *
from scipy.optimize import minimize

a = parse(__doc__)
L = 24 if a.quick else 48
ch, ns = (24, 6) if a.quick else (40, 8)
beta = 6.0
make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, 0.4)).scaled(b)
S0 = anneal(make, L, ch, np.linspace(0.5, beta, 12), 40 if a.quick else 100, seed=a.seed + 1)
S, _ = run_mc(make(beta), L, chains=ch, therm=100, nsamp=ns, gap=20, seed=a.seed + 2, init=S0)

def pl(J, D, hz):
    h = local_field(S, Hamiltonian.heisenberg_dmi(J, D, 'interfacial', B=(0, 0, hz)))
    k = np.linalg.norm(h, axis=-1)
    return (log_A(k) - (h*S).sum(-1)).sum()/len(S)

opt = dict(xatol=1e-6, fatol=1e-8, maxiter=4000)
J0, D0, h0 = minimize(lambda p: pl(*p), [6, 6, 2.4], method='Nelder-Mead', options=opt).x
print(f"free fit: J={J0:.4f} D={D0:.4f} hz={h0:.4f}")
for d in (0.1, 0.2, 0.4, 0.6):
    J, D = minimize(lambda p: pl(p[0], p[1], h0 - d), [J0, D0], method='Nelder-Mead', options=opt).x
    print(f"deficit {d}: dJ/J per unit {100*(J/J0-1)/d:+.2f}%  dD/D per unit {100*(D/D0-1)/d:+.2f}%  "
          f"cost (J,D adapt) {pl(J, D, h0-d) - pl(J0, D0, h0):.3f}  cost (J,D fixed) {pl(J0, D0, h0-d) - pl(J0, D0, h0):.3f} nat/sample")
