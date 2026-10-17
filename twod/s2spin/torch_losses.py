"""PyTorch versions of the decoder losses for the VAE stage (Stage 2).

NOT EXECUTED in the environment where this package was written (torch unavailable there).
tests/test_torch_losses.py checks them against the numpy reference (estimators.pl_loss_grad,
estimators.score_matching objective) and must pass before Stage 2 starts.

Shapes: S (n, Lx, Ly, 3) unit spins; h_site (n, Lx, Ly, 3) decoder field (or broadcastable);
Jx, Jy (3, 3) global or (n, 3, 3) per-sample bond tensors. Log-weight convention as in model.py.
"""
import math
import torch

def log_A(k):
    """log(4 pi sinh k / k), stable, finite gradients at k -> 0."""
    ks = k.clamp_min(1e-4)
    big = ks + torch.log1p(-torch.exp(-2*ks)) - math.log(2) - torch.log(ks)
    return math.log(4*math.pi) + torch.where(k < 1e-4, k**2/6, big)

def _bond(S, J, dim):
    """J S(r+mu) + J^T S(r-mu) for global (3,3) or per-sample (n,3,3) J."""
    Sp, Sm = torch.roll(S, -1, dims=dim), torch.roll(S, 1, dims=dim)
    if J.dim() == 2:
        return Sp @ J.T + Sm @ J
    return torch.einsum('nbc,nxyc->nxyb', J, Sp) + torch.einsum('ncb,nxyc->nxyb', J, Sm)

def effective_field(S, h_site, Jx=None, Jy=None):
    h = h_site.expand_as(S)
    if Jx is not None: h = h + _bond(S, Jx, -3)
    if Jy is not None: h = h + _bond(S, Jy, -2)
    return h

def vmf_nll(S, h):
    """-log p(S|h) per site for the independent (effective-field) decoder."""
    return log_A(h.norm(dim=-1)) - (h*S).sum(-1)

def coupled_pl_loss(S, h_site, Jx, Jy):
    """Negative log pseudo-likelihood per sample (sum over sites) for the coupled decoder."""
    return vmf_nll(S, effective_field(S, h_site, Jx, Jy)).sum(dim=(-1, -2))

def _bond_energy(S, J, dim):
    Sp = torch.roll(S, -1, dims=dim)
    JS = Sp @ J.T if J.dim() == 2 else torch.einsum('nbc,nxyc->nxyb', J, Sp)
    return (S*JS).sum(dim=(-1, -2, -3))

def score_matching_loss(S, h_site, Jx, Jy):
    """Hyvarinen objective on (S^2)^N per sample: sum_r 1/2|S_r x g_r|^2 + Delta W,
    with Delta W = -2 sum h.S - 4 sum_bonds S^T J S'  (l(l+1) weights; Q terms not included here)."""
    g = effective_field(S, h_site, Jx, Jy)
    torque = torch.linalg.cross(S, g, dim=-1)
    term1 = 0.5*(torque**2).sum(dim=(-1, -2, -3))
    lap = -2*(h_site.expand_as(S)*S).sum(dim=(-1, -2, -3)) - 4*(_bond_energy(S, Jx, -3) + _bond_energy(S, Jy, -2))
    return term1 + lap
