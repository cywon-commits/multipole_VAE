import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin import *                        # noqa
from s2spin import estimators as E          # noqa

def parse(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--quick', action='store_true', help='small sizes for a smoke run (minutes)')
    ap.add_argument('--out', default='results')
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()

def log_weight(S, H):
    return E.features(S) @ H.theta()

def tau_energy(H, L, chains=16, sweeps=400, seed=0, init='random'):
    """Integrated autocorrelation time (in sweeps) of the chain-averaged log-weight after burn-in."""
    rng = np.random.default_rng(seed)
    S = init.copy() if isinstance(init, np.ndarray) else random_unit((chains, L, L), rng)
    for _ in range(sweeps//2): sweep(S, H, rng)
    e = []
    for _ in range(sweeps):
        sweep(S, H, rng); e.append(log_weight(S, H))
    e = np.array(e)
    return float(np.median([observables.autocorr_time(e[:, c]) for c in range(chains)]))

def anneal(make_H, L, chains, betas, sweeps_per, seed=0):
    """make_H(beta) -> Hamiltonian. Returns configurations after the last beta."""
    rng = np.random.default_rng(seed)
    S = random_unit((chains, L, L), rng)
    for b in betas:
        H = make_H(b)
        for _ in range(sweeps_per): sweep(S, H, rng)
    return S

def param_rows(true, res, show_all=False):
    th, se = res['theta_full'], res.get('se')
    rows = ['| param | true | estimate | z |', '|---|---|---|---|']
    for k in res['idx']:
        z = (th[k] - true[k])/se[list(res['idx']).index(k)] if se is not None else np.nan
        if show_all or abs(true[k]) > 0 or abs(z) > 3:
            e = f"{th[k]:+.4f} ± {se[list(res['idx']).index(k)]:.4f}" if se is not None else f"{th[k]:+.4f}"
            rows.append(f"| {NAMES[k]} | {true[k]:+.4f} | {e} | {z:+.2f} |")
    return rows

def zero_param_maxz(true, res):
    idx = list(res['idx']); th = res['theta_full'][idx]; se = res['se']; tr = true[idx]
    m = np.abs(tr) == 0
    return float(np.max(np.abs(th[m]/se[m]))) if m.any() else float('nan')

def save(out, stage, summary, md):
    d = os.path.join(out, stage); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    with open(os.path.join(d, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
