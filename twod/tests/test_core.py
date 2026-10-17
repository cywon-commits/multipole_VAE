"""Fast checks (< 2 min total): pytest -q tests/"""
import numpy as np, pytest
from s2spin import *
from s2spin import estimators as E

def test_vmf_moments():
    rng = np.random.default_rng(0)
    for k in [0.1, 2.0, 20.0]:
        s = sample_vmf(np.tile([0, 0, 1.0], (200000, 1)), k, rng)
        assert abs(s[:, 2].mean() - langevin(k)) < 5e-3
        assert abs((1 - s[:, 2]**2).mean() - 2*langevin(k)/k) < 5e-3     # <sin^2> = 2L(k)/k

def test_log_A_continuity():
    k = np.array([0, 1e-5, 1e-4, 1e-3, 1, 50, 700])
    assert np.all(np.isfinite(log_A(k)))
    assert abs(log_A(1e-4 - 1e-9) - log_A(1e-4 + 1e-9)) < 1e-8

def test_decompose_roundtrip():
    H = Hamiltonian.heisenberg_dmi(1.3, 0.7, 'interfacial', B=(0.1, 0, 0.4))
    H2 = theta_to_hamiltonian(H.theta())
    assert np.allclose(H2.Jx, H.Jx) and np.allclose(H2.Jy, H.Jy) and np.allclose(H2.h, H.h)

def test_independent_sites_mc():
    H = Hamiltonian(np.zeros((3, 3)), np.zeros((3, 3)), np.array([0, 0, 3.0]))
    S, _ = run_mc(H, 8, chains=200, therm=1, nsamp=2, gap=1, seed=1)
    assert abs(S[..., 2].mean() - langevin(3.0)) < 0.01

def test_pl_gradient():
    rng = np.random.default_rng(0)
    H = Hamiltonian.heisenberg_dmi(1.0, 0.5, 'interfacial', B=(0, 0, 0.3))
    S, _ = run_mc(H, 8, chains=10, therm=20, nsamp=2, gap=5, seed=2)
    idx = list(range(21)); th = H.theta()[idx] + 0.1*rng.normal(size=21)
    _, g = pl_loss_grad(S, th, idx)
    gn = np.array([(pl_loss_grad(S, th + 1e-5*e, idx)[0] - pl_loss_grad(S, th - 1e-5*e, idx)[0])/2e-5 for e in np.eye(21)])
    assert np.abs(g - gn).max() < 1e-6

def test_sm_objective_matches_direct():
    """theta.M.theta/2 - theta.b equals the direct torque + Laplacian sum."""
    rng = np.random.default_rng(3)
    H = Hamiltonian.heisenberg_dmi(1.0, 0.5, 'bulk', B=(0, 0, 0.3))
    S, _ = run_mc(H, 8, chains=10, therm=20, nsamp=2, gap=5, seed=4)
    r = score_matching(S, list(range(21)))
    th = rng.normal(size=21)
    Ht = theta_to_hamiltonian(th, list(range(21)))
    g = local_field(S, Ht)
    torque = 0.5*(np.cross(S, g)**2).sum()
    bonds = ((S @ Ht.Jx)*np.roll(S, -1, axis=-3)).sum() + ((S @ Ht.Jy)*np.roll(S, -1, axis=-2)).sum()
    minus_lap = 2*(Ht.h*S).sum() + 4*bonds          # -sum_r Delta_r W
    direct = (torque - minus_lap)/S.shape[0]
    assert abs(direct - (0.5*th @ r['M'] @ th - th @ r['b'])) < 1e-8

def test_recovery_dmi_small():
    H = Hamiltonian.heisenberg_dmi(1.0, 0.6, 'interfacial', B=(0, 0, 0.4))
    S, cid = run_mc(H, 12, chains=60, therm=150, nsamp=6, gap=10, seed=3)
    r = score_matching(S, list(range(21)), groups=cid)
    z = (r['theta_full'][:21] - H.theta()[:21])/r['se']
    assert np.abs(z).max() < 4.0

def test_topological_charge_integer():
    rng = np.random.default_rng(5)
    Q = observables.topological_charge(random_unit((5, 10, 10), rng))
    assert np.allclose(Q, np.round(Q), atol=1e-8)
