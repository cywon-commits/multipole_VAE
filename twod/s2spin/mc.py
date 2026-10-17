"""Heat-bath (exact conditional vMF) + over-relaxation Monte Carlo on the periodic square lattice.

Spins: array (..., Lx, Ly, 3). Checkerboard update (Lx, Ly must be even); all chains vectorized.
The conditional of site r given its neighbours is vMF with field h_eff(r) = dW/dS_r (requires Q = 0).
"""
import numpy as np
from .vmf import sample_vmf, random_unit

def local_field(S, H):
    """h_eff(r) = h + J_x S(r+x) + J_x^T S(r-x) + J_y S(r+y) + J_y^T S(r-y)  (ambient gradient of W, Q=0 part)."""
    f = np.broadcast_to(H.h, S.shape).copy()
    for J, ax in ((H.Jx, -3), (H.Jy, -2)):
        f += np.roll(S, -1, axis=ax) @ J.T
        f += np.roll(S, +1, axis=ax) @ J
    return f

def _checker(Lx, Ly):
    return (np.add.outer(np.arange(Lx), np.arange(Ly)) % 2)

def sweep(S, H, rng, n_overrelax=1):
    if np.any(H.Q != 0):
        raise NotImplementedError("MC supports Q = 0 only (single-site l=2 needs a Kent sampler).")
    Lx, Ly = S.shape[-3], S.shape[-2]
    assert Lx % 2 == 0 and Ly % 2 == 0, "checkerboard needs even L"
    chk = _checker(Lx, Ly)
    for par in (0, 1):
        m = (chk == par)
        h = local_field(S, H)[..., m, :]
        k = np.linalg.norm(h, axis=-1)
        mu = h / np.where(k > 0, k, 1.0)[..., None]
        S[..., m, :] = sample_vmf(mu, k, rng)
    for _ in range(n_overrelax):
        for par in (0, 1):
            m = (chk == par)
            h = local_field(S, H)[..., m, :]
            hh = h / np.linalg.norm(h, axis=-1, keepdims=True)
            s = S[..., m, :]
            S[..., m, :] = 2*(s*hh).sum(-1, keepdims=True)*hh - s
    return S

def run_mc(H, L, chains=100, therm=500, nsamp=8, gap=50, seed=0, init='random', n_overrelax=1, monitor=None):
    """Returns (samples[nsamp*chains, L, L, 3], chain_id[nsamp*chains]).
    init: 'random' | 'field' (all along h_hat, or z) | ndarray (chains, L, L, 3).
    monitor(S, sweep_index) is called every gap sweeps (e.g. to record energy for autocorrelation)."""
    rng = np.random.default_rng(seed)
    if isinstance(init, np.ndarray):
        S = init.copy()
    elif init == 'random':
        S = random_unit((chains, L, L), rng)
    elif init == 'field':
        n = H.h/np.linalg.norm(H.h) if np.linalg.norm(H.h) > 0 else np.array([0, 0, 1.0])
        S = np.broadcast_to(n, (chains, L, L, 3)).copy()
    else:
        raise ValueError(init)
    t = 0
    for _ in range(therm):
        sweep(S, H, rng, n_overrelax); t += 1
    out = []
    for _ in range(nsamp):
        for _ in range(gap):
            sweep(S, H, rng, n_overrelax); t += 1
        if monitor is not None:
            monitor(S, t)
        out.append(S.copy())
    samples = np.concatenate(out)
    chain_id = np.tile(np.arange(S.shape[0]), nsamp)
    return samples, chain_id
