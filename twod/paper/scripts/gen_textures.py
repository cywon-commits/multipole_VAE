"""Fig. 1c / Fig. S5 data: spin texture samples. Stage 1c (J = D = 1 interfacial, beta = 6, L = 48): B = 0.4 (SkX) and B = 1.5 (polarised),
re-generated with a fixed seed and 2 chains (annealing as in stage1c_skx.py); E1 sample from the stored ensemble (results/stage2/E1.npz).
Writes data/fig1c_{polarized,skx,e1}.csv (x, y, Sx, Sy, Sz)."""
import os
from common import *
sys.path.insert(0, str(TWOD/'scripts'))
from _common import *            # noqa  (anneal, run_mc, Hamiltonian ...)
L, beta = 48, 6.0; betas = np.linspace(0.5, beta, 12)
for tag, B in (('skx', 0.4), ('polarized', 1.5)):
    make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, B)).scaled(b)
    S0 = anneal(make, L, 2, betas, 100, seed=0); S, _ = run_mc(make(beta), L, chains=2, therm=100, nsamp=1, gap=40, seed=0, init=S0)
    s = S[0]; x, y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    save_csv(f'fig1c_{tag}.csv', pd.DataFrame(dict(x=x.ravel(), y=y.ravel(), Sx=s[..., 0].ravel(), Sy=s[..., 1].ravel(), Sz=s[..., 2].ravel())))
    print(tag, 'Q =', observables.topological_charge(S)[0], 'mz', s[..., 2].mean(), flush=True)
z = np.load(RES/'stage2'/'E1.npz'); s = z['spins'][0]; n = z['field_dir'][0]; Ls = s.shape[0]; x, y = np.meshgrid(np.arange(Ls), np.arange(Ls), indexing='ij')
save_csv('fig1c_e1.csv', pd.DataFrame(dict(x=x.ravel(), y=y.ravel(), Sx=s[..., 0].ravel(), Sy=s[..., 1].ravel(), Sz=s[..., 2].ravel(), nx=n[0], ny=n[1], nz=n[2])))
print('E1 sample field direction', n)
