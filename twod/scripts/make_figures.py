"""Figures + SkX autocorrelation for REPORT.md. Uses results/stage1a/summary.json; re-runs 1c (B=0.4, 1.5) with the
same settings as stage1c_skx.py (seed 0) to save the texture, Q histogram and tau_int."""
from _common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

a = parse(__doc__); os.makedirs(f'{a.out}/fig', exist_ok=True)
s = json.load(open(f'{a.out}/stage1a/summary.json'))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k, v in s.items():
    if k.startswith('J='):
        C, P = np.array(v['C']), np.array(v['Cpred'])
        ax[0].semilogy(range(len(C)), C, 'o', label=f'MC {k}'); ax[0].semilogy(range(len(P)), P, '-', c=ax[0].lines[-1].get_color())
ax[0].set_xlabel('r'); ax[0].set_ylabel('C(r)'); ax[0].legend(fontsize=7); ax[0].set_title('transverse C(r), lines = 2G(r)')
mw = s['mermin_wagner']; ks = sorted(mw, key=float)
ax[1].semilogx([float(k) for k in ks], [mw[k][0] for k in ks], 'o', label='MC')
ax[1].semilogx([float(k) for k in ks], [mw[k][1] for k in ks], '-', label='lattice 2G(0)')
ax[1].set_xlabel('kappa'); ax[1].set_ylabel('<|delta|^2>'); ax[1].legend(); ax[1].set_title('Mermin-Wagner trend (J=5)')
fig.tight_layout(); fig.savefig(f'{a.out}/fig/stage1a.png', dpi=130)

L, ch, ns, gap, th = 48, 60, 16, 40, 400
beta = 6.0; betas = np.linspace(0.5, beta, 12); out = {}
fig, ax = plt.subplots(2, 2, figsize=(8, 7))
for j, B in enumerate([0.4, 1.5]):
    make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, B)).scaled(b)
    S0 = anneal(make, L, ch, betas, 100, seed=a.seed); H = make(beta)
    S, cid = run_mc(H, L, chains=ch, therm=th, nsamp=ns, gap=gap, seed=a.seed, init=S0)
    Q = observables.topological_charge(S)
    # tau_int of the log-weight and of Q along a continued run (1 sweep resolution)
    rng = np.random.default_rng(1); T = S0.copy()
    for _ in range(th): sweep(T, H, rng)
    e, q = [], []
    for _ in range(600): sweep(T, H, rng); e.append(log_weight(T, H)); q.append(observables.topological_charge(T))
    e, q = np.array(e), np.array(q)
    te = float(np.median([observables.autocorr_time(e[:, c]) for c in range(ch)]))
    tq = float(np.median([observables.autocorr_time(q[:, c]) for c in range(ch)])) if q.std() > 0 else 0.0
    out[f'B={B}'] = dict(tau_logw=te, tau_Q=tq, gap=gap, Qmean=float(Q.mean()), Qstd=float(Q.std()))
    im = S[0, 0] if S.ndim == 5 else S[0]
    ax[0, j].imshow(im[..., 2].T, origin='lower', cmap='RdBu', vmin=-1, vmax=1); ax[0, j].set_title(f'B={B}: S_z texture')
    ax[1, j].hist(Q.ravel(), bins=20); ax[1, j].set_title(f'B={B}: Q distribution'); ax[1, j].set_xlabel('Q')
fig.tight_layout(); fig.savefig(f'{a.out}/fig/stage1c.png', dpi=130)
json.dump(out, open(f'{a.out}/stage1c/tau.json', 'w'), indent=2); print(out)
