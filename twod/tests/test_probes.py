import numpy as np
from s2spin import probes, sample_vmf, langevin

def test_peaks_and_phase_lock():
    rng = np.random.default_rng(0); L = 48; n = 64
    x = np.arange(L)[:, None]; y = np.arange(L)[None, :]
    ph = rng.uniform(0, 2*np.pi, n)[:, None, None]
    F = np.cos(2*np.pi*(6*x + 0*y)/L + ph) + 0.1*rng.normal(size=(n, L, L))
    ks = probes.texture_peaks(F, 1)
    assert ks[0] in [(6, 0), (-6, 0)]
    G = 0.5*np.cos(2*np.pi*6*x/L + ph + 0.7)                 # locked, constant offset
    H = np.cos(2*np.pi*6*x/L + rng.uniform(0, 2*np.pi, n)[:, None, None])   # unrelated
    assert probes.phase_lock(F, G, ks)[0]['R'] > 0.99
    assert probes.phase_lock(F, H, ks)[0]['R'] < 0.4

def test_shift_consistency():
    rng = np.random.default_rng(1); X = rng.normal(size=(4, 16, 16, 3))
    blur = lambda A: (A + np.roll(A, 1, 1) + np.roll(A, 1, 2))/3
    assert min(probes.shift_consistency(blur, X, [(3, 5), (1, 0)]).values()) > 0.999
    fixed = lambda A: blur(A)*np.linspace(0, 1, 16)[None, :, None, None]    # breaks translation
    assert min(probes.shift_consistency(fixed, X, [(5, 0)]).values()) < 0.9

def test_pl_fisher_matches_sampling():
    rng = np.random.default_rng(2)
    for k, a in [(3.0, 0.0), (5.0, 1.0)]:
        h = k*np.array([np.sin(a), 0, np.cos(a)])
        s = sample_vmf(np.tile(h/k, (200000, 1)), k, rng)
        I = probes.pl_fisher_field(np.zeros((1, 1, 1, 3)), h[None, None, None, :])[0]
        assert abs(I - s[:, 2].var()) < 3e-3
