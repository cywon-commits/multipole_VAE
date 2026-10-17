"""Analytic predictions (sec. 2.3 of the design doc), square lattice, periodic L x L.

Large-kappa spin-wave (Gaussian) limit around a uniform background n with field kappa n and FM J:
    precision K = kappa I + J L_lat ;  per transverse component  G(r) = (1/L^2) sum_q e^{iqr}/(kappa + J eps_q)
    eps_q = 4 - 2cos qx - 2cos qy ;  <delta0.delta_r> = 2 G(r) ;  continuum G = K0(r/xi)/(2 pi J), xi = sqrt(J/kappa)
Independent (J = 0) decoder fit to the same data:  kappa_ind ~ 2/<|delta|^2> = 1/G(0)   (> kappa).
Collinear identifiability: with Gamma = gamma diag(1,1,-2) on bond mu (axis = n = z), the Gaussian precision depends
only on J_mu + gamma_mu and kappa + 2 sum_mu (J_mu - 2 gamma_mu): flat directions (dJ_mu, dgamma_mu, dkappa) = (1, -1, -6)
per bond direction; tied isotropic (dJ, dgamma, dkappa) = (1, -1, -12).  Note G2 = sqrt(6) gamma in GAMMA_BASIS units.
"""
import numpy as np
from scipy.special import k0

def gaussian_G(L, kappa, J):
    q = 2*np.pi*np.arange(L)/L
    eps = 4 - 2*np.cos(q)[:, None] - 2*np.cos(q)[None, :]
    return np.real(np.fft.ifft2(1.0/(kappa + J*eps)))

def transverse_corr_pred(L, kappa, J, rmax):
    G = gaussian_G(L, kappa, J)
    return 2*G[:rmax + 1, 0]

def kappa_ind_pred(L, kappa, J):
    return 1.0/gaussian_G(L, kappa, J)[0, 0]

def delta2_disk(kappa, J):
    """Disk-BZ approximation <|delta|^2> = ln(1 + 4 pi J/kappa)/(2 pi J)  (log divergence as kappa -> 0)."""
    return np.log1p(4*np.pi*J/kappa)/(2*np.pi*J)

def continuum_corr(r, kappa, J):
    xi = np.sqrt(J/kappa)
    return k0(np.asarray(r, float)/xi)/(np.pi*J)

def moriya_expected(kind):
    """Expected DMI directions per bond: dict mu -> unit vector."""
    if kind == 'interfacial': return {'x': np.array([0, 1., 0]), 'y': np.array([-1., 0, 0])}
    if kind == 'bulk': return {'x': np.array([1., 0, 0]), 'y': np.array([0, 1., 0])}
    raise ValueError(kind)
