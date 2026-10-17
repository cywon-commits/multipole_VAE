import numpy as np, pytest
torch = pytest.importorskip("torch")
from s2spin.vae import SpinVAE
from s2spin import torch_losses as TL

def _S(n=4, L=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, L, L, 3, generator=g, dtype=torch.float64), dim=-1)

def test_empty_mask_equals_plain_pl_field():
    S = _S(); h = torch.randn(4, 8, 8, 3, dtype=torch.float64)
    Jx, Jy = torch.eye(3, dtype=torch.float64)*2, torch.eye(3, dtype=torch.float64)*3
    mk = torch.zeros(4, 8, 8, dtype=torch.float64)
    assert torch.allclose(TL.effective_field(S*(1 - mk).unsqueeze(-1), h, Jx, Jy), TL.effective_field(S, h, Jx, Jy))

def test_masked_field_drops_masked_neighbours():
    S = _S(1, 6); h = torch.zeros(1, 6, 6, 3, dtype=torch.float64)
    Jx = torch.eye(3, dtype=torch.float64)*2; Jy = torch.eye(3, dtype=torch.float64)*3
    mk = torch.zeros(1, 6, 6, dtype=torch.float64); mk[0, 2:4, 2:4] = 1
    g = TL.effective_field(S*(1 - mk).unsqueeze(-1), h, Jx, Jy)[0, 2, 2]
    # site (2,2): x-neighbours (1,2) unmasked, (3,2) masked; y-neighbours (2,1) unmasked, (2,3) masked
    assert torch.allclose(g, 2*S[0, 1, 2] + 3*S[0, 2, 1])

def test_masked_loss_split_and_zero_mask():
    torch.manual_seed(0)
    m = SpinVAE(8, True, mask_ch=True, dmi='interfacial', spur=[[0, 0, 1.]]).double()
    S = _S(); z = torch.zeros(4, 8, 8)
    o = m.losses(S, stochastic=False, mask=z)
    assert torch.allclose(o['rec'], o['rec_unmasked']) and float(o['rec_masked']) == 0.0
    mk = (torch.rand(4, 8, 8) < 0.25).double()
    o = m.losses(S, stochastic=False, mask=mk)
    assert torch.allclose(o['rec'], o['rec_masked'] + o['rec_unmasked'])

def test_given_B_decoder_depends_on_B_and_is_equivariant():
    from scipy.spatial.transform import Rotation
    torch.manual_seed(2)
    m = SpinVAE(8, True, given_B=True).double()          # no fixed spurion: only B z_hat enters
    S = _S(2); B = torch.tensor([0.3, 0.5], dtype=torch.float64)
    mu1, mu0, _, _ = m.encode(S)
    h1, h2 = m.decode(mu1, mu0, B), m.decode(mu1, mu0, 2*B)
    assert (h1 - h2).abs().max() > 1e-6
    R = torch.tensor(Rotation.random(1, random_state=3).as_matrix()[0])
    # rotate latent vectors and B z_hat together: the decoded field rotates (B z_hat is passed as a vector channel)
    rot = lambda V: torch.einsum('ab,nkbxy->nkaxy', R, V)
    Bz = torch.stack([torch.zeros_like(B), torch.zeros_like(B), B], -1)
    spur_r = (Bz @ R.T)[:, None, :]
    hr = m.dec(rot(mu1), mu0, 8, spur_r)
    h0 = m.dec(mu1, mu0, 8, Bz[:, None, :])
    assert torch.allclose(hr, h0 @ R.T, atol=1e-9, rtol=1e-6*float(h0.abs().max()) + 1e-6)
