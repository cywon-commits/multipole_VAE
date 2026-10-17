"""Diagnostics for what a VAE latent / decoded field encodes (Stage 2b).

Rationale: with a translation-equivariant encoder the texture position is carried by WHERE features sit in
the latent grid, not by latent values, and the absolute 'shift' label of E2 is relative to an unknown base
position of each chain. So (i) test equivariance on paired inputs, and (ii) compare Fourier phases of the
decoded field with those of the data texture at the texture's own wavevectors (label derived from data).
"""
import numpy as np

def texture_peaks(F, n_peaks=3, exclude_zero=True):
    """Dominant wavevectors of scalar fields F (n, Lx, Ly) from the mean power spectrum.
    Returns integer index pairs (m, n) for k = 2 pi (m/Lx, n/Ly); one of each +-k pair."""
    F = F - F.mean(axis=(1, 2), keepdims=True)
    P = (np.abs(np.fft.fft2(F))**2).mean(0)
    Lx, Ly = P.shape
    if exclude_zero: P[0, 0] = 0
    order = np.argsort(P.ravel())[::-1]
    out = []
    for o in order:
        m, n = np.unravel_index(o, P.shape)
        m = m if m <= Lx//2 else m - Lx; n = n if n <= Ly//2 else n - Ly
        if (-m, -n) in out or (m, n) in out: continue
        out.append((int(m), int(n)))
        if len(out) == n_peaks: break
    return out

def fourier_amp(F, k):
    """Complex amplitude of fields F (n, Lx, Ly) at integer wavevector k = (m, n)."""
    Lx, Ly = F.shape[1:3]
    x = np.arange(Lx)[:, None]; y = np.arange(Ly)[None, :]
    ph = np.exp(-2j*np.pi*(k[0]*x/Lx + k[1]*y/Ly))
    return (F*ph).sum(axis=(1, 2))/(Lx*Ly)

def phase_lock(F_data, F_model, ks):
    """For each k: R = |< exp(i(phi_model - phi_data)) >| over samples (1 = locked, ~1/sqrt(n) = unrelated)
    and the median amplitude ratio |A_model|/|A_data|."""
    res = []
    for k in ks:
        a, b = fourier_amp(F_data, k), fourier_amp(F_model, k)
        ok = (np.abs(a) > 0) & (np.abs(b) > 0)
        R = np.abs(np.mean(np.exp(1j*(np.angle(b[ok]) - np.angle(a[ok]))))) if ok.any() else np.nan
        res.append(dict(k=k, R=float(R), amp_ratio=float(np.median(np.abs(b)/np.maximum(np.abs(a), 1e-300)))))
    return res

def shift_consistency(fn, X, shifts, axes=(1, 2)):
    """fn: X (n, Lx, Ly, ...) -> field (n, Lx, Ly, c) (e.g. decoded h of the mean latent).
    Returns, per shift a, the mean cosine similarity between fn(T_a X) and T_a fn(X) (1 = equivariant)."""
    base = fn(X); out = {}
    for a in shifts:
        lhs = fn(np.roll(X, a, axis=axes)); rhs = np.roll(base, a, axis=axes)
        num = (lhs*rhs).reshape(len(X), -1).sum(1)
        den = np.linalg.norm(lhs.reshape(len(X), -1), axis=1)*np.linalg.norm(rhs.reshape(len(X), -1), axis=1)
        out[tuple(a)] = float(np.mean(num/np.maximum(den, 1e-300)))
    return out

def pl_fisher_field(S, h_eff, e=(0, 0, 1.0)):
    """Pseudo-likelihood Fisher information per sample for a uniform field strength along e:
    I = sum_i Var_cond(S_i . e), using vMF conditionals with fields h_eff (n, Lx, Ly, 3).
    Var(S.e) = cos^2(a) (1 - 2L/k - L^2) + sin^2(a) L/k, a = angle(e, h_hat)."""
    from .vmf import langevin
    e = np.asarray(e, float)/np.linalg.norm(e)
    k = np.linalg.norm(h_eff, axis=-1); L = langevin(k)
    c2 = ((h_eff @ e)/np.maximum(k, 1e-300))**2
    var = c2*(1 - 2*L/np.maximum(k, 1e-12) - L**2) + (1 - c2)*L/np.maximum(k, 1e-12)
    return var.reshape(len(S), -1).sum(1)
