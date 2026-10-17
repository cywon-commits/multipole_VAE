"""Run before Stage 2. Skipped if torch is not installed."""
import numpy as np, pytest
torch = pytest.importorskip("torch")
from s2spin import *
from s2spin import torch_losses as TL

def _data():
    H = Hamiltonian.heisenberg_dmi(1.0, 0.5, 'interfacial', B=(0, 0, 0.3))
    S, _ = run_mc(H, 8, chains=6, therm=20, nsamp=2, gap=5, seed=2)
    return H, S

def test_pl_matches_numpy():
    H, S = _data()
    idx = list(range(21)); th = H.theta()[idx]
    l_np, _ = pl_loss_grad(S, th, idx)
    St = torch.tensor(S)
    l_t = TL.coupled_pl_loss(St, torch.tensor(H.h), torch.tensor(H.Jx), torch.tensor(H.Jy)).mean().item()
    assert abs(l_np - l_t) < 1e-8

def test_sm_matches_numpy():
    H, S = _data()
    r = score_matching(S, list(range(21)))
    th = H.theta()[:21]
    J_np = 0.5*th @ r['M'] @ th - th @ r['b']
    St = torch.tensor(S)
    J_t = TL.score_matching_loss(St, torch.tensor(H.h), torch.tensor(H.Jx), torch.tensor(H.Jy)).mean().item()
    assert abs(J_np - J_t) < 1e-8

def test_per_sample_J_broadcast():
    H, S = _data()
    St = torch.tensor(S); n = St.shape[0]
    a = TL.coupled_pl_loss(St, torch.tensor(H.h), torch.tensor(H.Jx), torch.tensor(H.Jy))
    b = TL.coupled_pl_loss(St, torch.tensor(H.h), torch.tensor(H.Jx).expand(n, 3, 3), torch.tensor(H.Jy).expand(n, 3, 3))
    assert torch.allclose(a, b)
