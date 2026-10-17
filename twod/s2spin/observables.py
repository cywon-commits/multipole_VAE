"""Observables: transverse correlations, topological charge, magnetization, autocorrelation time."""
import numpy as np

def transverse_corr(S, n, rmax, axis=-3):
    """C(r) = < delta(r0) . delta(r0 + r e_axis) >, delta = S - (S.n) n. Returns array (rmax+1,)."""
    n = np.asarray(n, float)/np.linalg.norm(n)
    d = S - (S @ n)[..., None]*n
    return np.array([(d*np.roll(d, -r, axis=axis)).sum(-1).mean() for r in range(rmax + 1)])

def connected_corr(S, rmax, axis=-3):
    """C(r) = <S(r0).S(r0+r)> - <S>.<S> (site-averaged)."""
    m = S.reshape(-1, 3).mean(0)
    return np.array([(S*np.roll(S, -r, axis=axis)).sum(-1).mean() - m @ m for r in range(rmax + 1)])

def _solid_angle(a, b, c):
    num = np.einsum('...a,...a->...', a, np.cross(b, c))
    den = 1 + (a*b).sum(-1) + (b*c).sum(-1) + (c*a).sum(-1)
    return 2*np.arctan2(num, den)

def topological_charge(S):
    """Berg-Luscher lattice charge per sample (exactly integer for any configuration on a torus)."""
    s00 = S; s10 = np.roll(S, -1, -3); s01 = np.roll(S, -1, -2); s11 = np.roll(s10, -1, -2)
    om = _solid_angle(s00, s10, s11) + _solid_angle(s00, s11, s01)
    return om.reshape(S.shape[0], -1).sum(1)/(4*np.pi)

def magnetization(S):
    return S.reshape(S.shape[0], -1, 3).mean(1)

def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time (Sokal window) of a 1D series."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    f = np.fft.rfft(x, 2*n); acf = np.fft.irfft(f*np.conj(f))[:n]; acf /= acf[0]
    tau = 1.0
    for w in range(1, n):
        tau = 1 + 2*acf[1:w+1].sum()
        if w >= c*tau: break
    return tau
