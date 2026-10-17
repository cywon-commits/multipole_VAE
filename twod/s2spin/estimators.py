"""Partition-function-free estimators for the translation-invariant model of model.py.

Parameter vector theta (26):  for mu in (x, y): J0_mu, D_mu (3), G_mu (5)   -> 18
                              h (3), Q (5)                                    -> 8
W(S) = theta . Phi(S) is linear in theta.

score_matching:  J_SM = E sum_r [ 1/2 |S_r x g_r|^2 + Delta_r W ],  g_r = dW/dS_r (ambient),
                 Delta_{S^2} Y_l = -l(l+1) Y_l  ->  weights 4 (bond, linear in each end), 2 (h), 6 (Q).
                 Quadratic in theta: theta* = M^{-1} b  (no normalizer, works with Q).
pseudo_likelihood: closed-form vMF conditionals (requires Q excluded), L-BFGS, convex in theta.
"""
import numpy as np
from scipy.optimize import minimize
from .model import EPS, GAMMA_BASIS, Hamiltonian
from .vmf import langevin, log_A
from .mc import local_field

NAMES = []
for mu in ('x', 'y'):
    NAMES += [f'J0_{mu}', f'Dx_{mu}', f'Dy_{mu}', f'Dz_{mu}'] + [f'G{k+1}_{mu}' for k in range(5)]
NAMES += ['hx', 'hy', 'hz'] + [f'Q{k+1}' for k in range(5)]
P = len(NAMES)
_BOND_MATS = [np.eye(3)] + [EPS[a] for a in range(3)] + list(GAMMA_BASIS)   # 9 per direction
_LAP = np.array([4.0]*18 + [2.0]*3 + [6.0]*5)

def _kind(k):
    if k < 18: return 'bond', (-3 if k < 9 else -2), _BOND_MATS[k % 9]
    if k < 21: return 'h', None, k - 18
    return 'Q', None, GAMMA_BASIS[k - 21]

def theta_to_hamiltonian(theta, idx=None):
    th = np.zeros(P)
    if idx is None: th[:] = theta
    else: th[np.asarray(idx)] = theta
    Jx = sum(th[k]*_BOND_MATS[k] for k in range(9))
    Jy = sum(th[9+k]*_BOND_MATS[k] for k in range(9))
    Q = sum(th[21+k]*GAMMA_BASIS[k] for k in range(5))
    return Hamiltonian(Jx, Jy, th[18:21].copy(), Q)

def tie_matrix(idx, groups):
    """Linear constraints theta_sub = T phi. groups: list of lists of names that share one free parameter;
    names in idx not mentioned in any group stay free. Example (isotropic exchange, x/y tied):
    tie_matrix(idx, [['J0_x','J0_y']])."""
    idx = list(idx); cols = []; used = set()
    for g in groups:
        c = np.zeros(len(idx))
        for nm in g:
            c[idx.index(NAMES.index(nm))] = 1; used.add(NAMES.index(nm))
        cols.append(c)
    for j, k in enumerate(idx):
        if k not in used:
            c = np.zeros(len(idx)); c[j] = 1; cols.append(c)
    return np.array(cols).T

def _grads(S, idx):
    """g_k(r) = dPhi_k/dS_r for k in idx: array (p, n, Lx, Ly, 3)."""
    out = np.empty((len(idx),) + S.shape)
    for j, k in enumerate(idx):
        kind, ax, B = _kind(k)
        if kind == 'bond':
            out[j] = np.roll(S, -1, axis=ax) @ B.T + np.roll(S, +1, axis=ax) @ B
        elif kind == 'h':
            out[j] = 0.0; out[j][..., B] = 1.0
        else:
            out[j] = 2*S @ B
    return out

def features(S, idx=None):
    """Phi_k(S) per sample: (n, p)."""
    idx = list(range(P)) if idx is None else list(idx)
    f = np.empty((S.shape[0], len(idx)))
    for j, k in enumerate(idx):
        kind, ax, B = _kind(k)
        if kind == 'bond':
            f[:, j] = ((S @ B)*np.roll(S, -1, axis=ax)).reshape(S.shape[0], -1).sum(1)
        elif kind == 'h':
            f[:, j] = S[..., B].reshape(S.shape[0], -1).sum(1)
        else:
            f[:, j] = ((S @ B)*S).reshape(S.shape[0], -1).sum(1)
    return f

def score_matching(S, idx=None, T=None, groups=None, chunk=16, ridge=0.0):
    """Returns dict(theta, theta_full, M, b, se, cond).
    S: (n, Lx, Ly, 3). idx: parameter indices (default all 26). T: tie matrix (see tie_matrix).
    groups: label per sample for leave-one-group-out jackknife errors (use chain_id from run_mc)."""
    idx = list(range(P)) if idx is None else list(idx)
    p = len(idx); n = S.shape[0]
    T = np.eye(p) if T is None else T
    groups = np.zeros(n, int) if groups is None else np.asarray(groups)
    labels, ginv = np.unique(groups, return_inverse=True)
    G = len(labels)
    Mg = np.zeros((G, p, p)); bg = np.zeros((G, p)); cnt = np.zeros(G)
    lap = _LAP[idx]
    for s0 in range(0, n, chunk):
        Sc = S[s0:s0+chunk]
        t = np.cross(Sc[None], _grads(Sc, idx))                   # torques (p, c, Lx, Ly, 3)
        tt = t.reshape(t.shape[0], t.shape[1], -1)
        Ms = np.einsum('knx,lnx->nkl', tt, tt)
        bs = lap*features(Sc, idx)
        gi = ginv[s0:s0+chunk]
        np.add.at(Mg, gi, Ms); np.add.at(bg, gi, bs); np.add.at(cnt, gi, 1)
    def solve(M, b, m):
        M = T.T @ (M/m) @ T; b = T.T @ (b/m)
        M = M + ridge*np.eye(M.shape[0])
        return T @ np.linalg.solve(M, b), M
    Mt, bt = Mg.sum(0), bg.sum(0)
    theta, Mred = solve(Mt, bt, n)
    se = None
    if G > 1:
        jk = np.array([solve(Mt - Mg[g], bt - bg[g], n - cnt[g])[0] for g in range(G)])
        se = np.sqrt((G - 1)/G*((jk - jk.mean(0))**2).sum(0))
    d = np.sqrt(np.diag(Mred))
    cond = np.linalg.cond(Mred/np.outer(d, d))
    full = np.zeros(P); full[idx] = theta
    return dict(theta=theta, theta_full=full, idx=idx, M=Mt/n, b=bt/n, se=se, cond=cond)

def pl_loss_grad(S, theta, idx, chunk=32):
    """Mean (per sample) negative log pseudo-likelihood and its gradient w.r.t. theta[idx]."""
    H = theta_to_hamiltonian(theta, idx)
    n = S.shape[0]; loss = 0.0; gfull = np.zeros(P)
    for s0 in range(0, n, chunk):
        Sc = S[s0:s0+chunk]
        h = local_field(Sc, H)
        k = np.linalg.norm(h, axis=-1)
        loss += (log_A(k) - (h*Sc).sum(-1)).sum()
        R = (langevin(k)/np.where(k > 0, k, 1))[..., None]*h - Sc          # dL/dh_eff
        for mu, ax in ((0, -3), (1, -2)):
            dJ = R.reshape(-1, 3).T @ np.roll(Sc, -1, axis=ax).reshape(-1, 3) \
               + Sc.reshape(-1, 3).T @ np.roll(R, -1, axis=ax).reshape(-1, 3)
            for q in range(9):
                gfull[9*mu + q] += (dJ*_BOND_MATS[q]).sum()
        gfull[18:21] += R.reshape(-1, 3).sum(0)
    return loss/n, gfull[list(idx)]/n

def pseudo_likelihood(S, idx=None, T=None, theta0=None, chunk=32, maxiter=500):
    """Exact-conditional pseudo-likelihood (vMF conditionals). Q parameters not allowed.
    Returns dict(theta, theta_full, loss, success)."""
    idx = list(range(21)) if idx is None else list(idx)
    if any(k >= 21 for k in idx):
        raise ValueError("pseudo_likelihood: Q needs a Kent normalizer; use score_matching for Q.")
    p = len(idx); T = np.eye(p) if T is None else T
    def fg(phi):
        l, g = pl_loss_grad(S, T @ phi, idx, chunk)
        return l, T.T @ g
    phi0 = np.zeros(T.shape[1]) if theta0 is None else np.linalg.lstsq(T, np.asarray(theta0), rcond=None)[0]
    r = minimize(fg, phi0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter))
    th = T @ r.x; full = np.zeros(P); full[idx] = th
    return dict(theta=th, theta_full=full, idx=idx, loss=r.fun, success=r.success)
