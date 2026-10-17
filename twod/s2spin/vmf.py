"""von Mises-Fisher (Langevin) distribution on S^2: p(S|h) = exp(h.S)/A(|h|), A = 4 pi sinh(k)/k."""
import numpy as np

def langevin(k):
    """L(k) = coth k - 1/k = <S.h_hat>; series for small k."""
    k = np.asarray(k, dtype=float)
    small = np.abs(k) < 1e-3
    ks = np.where(small, 1.0, k)
    return np.where(small, k/3 - k**3/45, 1/np.tanh(ks) - 1/ks)

def log_A(k):
    """log(4 pi sinh k / k), numerically stable for all k >= 0."""
    k = np.asarray(k, dtype=float)
    small = k < 1e-4
    ks = np.where(small, 1.0, k)
    big = ks + np.log1p(-np.exp(-2*ks)) - np.log(2) - np.log(ks)
    return np.log(4*np.pi) + np.where(small, k**2/6, big)

def random_unit(shape, rng):
    v = rng.normal(size=tuple(shape) + (3,))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def sample_vmf(mu, kappa, rng):
    """Exact sampler. mu: (...,3) unit axes; kappa: (...) >= 0. Inverse CDF for w = cos(theta)."""
    mu = np.asarray(mu, dtype=float)
    kappa = np.broadcast_to(np.asarray(kappa, float), mu.shape[:-1])
    u = rng.random(mu.shape[:-1])
    tiny = kappa < 1e-8
    ks = np.where(tiny, 1.0, kappa)
    w = np.where(tiny, 2*u - 1, 1 + np.log(u + (1 - u)*np.exp(-2*ks))/ks)
    w = np.clip(w, -1.0, 1.0)
    phi = 2*np.pi*rng.random(mu.shape[:-1])
    helper = np.where(np.abs(mu[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    e1 = np.cross(mu, helper); e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(mu, e1)
    s = np.sqrt(1 - w**2)[..., None]
    return w[..., None]*mu + s*(np.cos(phi)[..., None]*e1 + np.sin(phi)[..., None]*e2)
