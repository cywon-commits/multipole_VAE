"""SO(3)_S-equivariant, translation-equivariant (periodic conv) VAE for S^2 spin fields (Stage 2).

Feature types per site: vectors V (B, C, 3, H, W) (l=1) and scalars s (B, C, H, W) (l=0).
Layers mix channels/space linearly with kernels shared over the 3 spin components (equivariant), and use
rotation-invariant gates built from |V| and s (vector-neuron style).  Constant "spurion" vectors (e.g. z_hat
for the field, x_hat/y_hat for the DMI pattern) can be appended as extra input channels: they rotate with the
Hamiltonian, so the model is equivariant under the joint rotation of (spins, spurions).

Latent: K1 vector channels (l=1, isotropic Gaussian noise, scalar log-variance) and K0 scalar channels (l=0) on a
g x g grid (g = 1: one global latent).  Decoder outputs the l=1 effective field h(r); the coupled decoder adds
global bonds J_mu = J0 I + D_mu[x] (learnable scalars J0, D; the DMI pattern is a fixed spurion tensor).
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from . import torch_losses as TL

def vnorm(V):
    return torch.sqrt((V**2).sum(2) + 1e-8)

class VNConv(nn.Module):
    def __init__(self, cv_in, cs_in, cv_out, cs_out, k=3, gate=True):
        super().__init__()
        pad = dict(padding=k//2, padding_mode='circular') if k > 1 else {}
        self.cv = nn.Conv2d(cv_in, cv_out, k, bias=False, **pad)
        self.cs = nn.Conv2d(cs_in, cs_out, k, **pad)
        self.mix = nn.Conv2d(cs_out + cv_out, cs_out + cv_out, 1)
        self.cs_out, self.cv_out, self.gate = cs_out, cv_out, gate

    def forward(self, V, s):
        B, C, _, H, W = V.shape
        V1 = self.cv(V.permute(0, 2, 1, 3, 4).reshape(B*3, C, H, W)).reshape(B, 3, self.cv_out, H, W).permute(0, 2, 1, 3, 4)
        h = self.mix(torch.cat([self.cs(s), vnorm(V1)], 1))
        s1, g = h[:, :self.cs_out], h[:, self.cs_out:]
        return V1*torch.sigmoid(g).unsqueeze(2), F.silu(s1)

def pool(V, s):
    B, C, _, H, W = V.shape
    V = F.avg_pool2d(V.permute(0, 2, 1, 3, 4).reshape(B*3, C, H, W), 2).reshape(B, 3, C, H//2, W//2).permute(0, 2, 1, 3, 4)
    return V, F.avg_pool2d(s, 2)

def spurion_channels(spur, B, H, W, dtype, device):
    """spur: (n_sp, 3) constant vectors, or (B, n_sp, 3) per-sample -> (B, n_sp, 3, H, W)."""
    sp = torch.as_tensor(spur, dtype=dtype, device=device)
    if sp.dim() == 2: sp = sp.expand(B, -1, -1)
    return sp[..., None, None].expand(-1, -1, -1, H, W)

class Encoder(nn.Module):
    def __init__(self, K1=2, K0=2, C=8, n_layers=3, n_down=0, n_sp=0, global_pool=True):
        super().__init__()
        self.n_sp, self.n_down, self.global_pool, self.K1, self.K0 = n_sp, n_down, global_pool, K1, K0
        cv, cs = 1 + n_sp, 1
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(VNConv(cv, cs, C, C)); cv, cs = C, C
        self.mu1 = nn.Conv2d(C, K1, 1, bias=False)
        self.head = nn.Conv2d(C + C, 2*K0 + K1, 1)          # from [s, |V|] -> mu0, logvar0, logvar1

    def forward(self, S, spur=None):
        B, H, W, _ = S.shape
        V = S.permute(0, 3, 1, 2).unsqueeze(1)
        if self.n_sp: V = torch.cat([V, spurion_channels(spur, B, H, W, S.dtype, S.device)], 1)
        s = torch.ones(B, 1, H, W, dtype=S.dtype, device=S.device)
        for i, l in enumerate(self.layers):
            V, s = l(V, s)
            if i < self.n_down: V, s = pool(V, s)
        if self.global_pool:
            V, s = V.mean((-1, -2), keepdim=True), s.mean((-1, -2), keepdim=True)
        Bq, C, _, h, w = V.shape
        mu1 = self.mu1(V.permute(0, 2, 1, 3, 4).reshape(Bq*3, C, h, w)).reshape(Bq, 3, self.K1, h, w).permute(0, 2, 1, 3, 4)
        o = self.head(torch.cat([s, vnorm(V)], 1))
        return mu1, o[:, :self.K0], o[:, self.K0:2*self.K0], o[:, 2*self.K0:]

class Decoder(nn.Module):
    def __init__(self, K1=2, K0=2, C=8, n_layers=3, n_sp=0):
        super().__init__()
        self.n_sp = n_sp
        cv, cs = K1 + n_sp, K0
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(VNConv(cv, cs, C, C)); cv, cs = C, C
        self.out = nn.Conv2d(C, 1, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))        # overall field magnitude (invariant)

    def forward(self, z1, z0, L, spur=None):
        """z1 (B,K1,3,g,g), z0 (B,K0,g,g) -> h (B, L, L, 3)."""
        B = z1.shape[0]; g = z1.shape[-1]
        V = z1
        if g != L:
            V = F.interpolate(V.reshape(B, -1, g, g), size=(L, L), mode='nearest').reshape(B, z1.shape[1], 3, L, L)
            s = F.interpolate(z0, size=(L, L), mode='nearest')
        else:
            s = z0
        if self.n_sp: V = torch.cat([V, spurion_channels(spur, B, L, L, V.dtype, V.device)], 1)
        for l in self.layers: V, s = l(V, s)
        C = V.shape[1]
        h = self.out(V.permute(0, 2, 1, 3, 4).reshape(B*3, C, L, L)).reshape(B, 3, L, L)
        return self.scale*h.permute(0, 2, 3, 1)

def bond_tensors(J0, D, kind, dtype=torch.float32):
    """Global nearest-neighbour bonds; kind in {None, 'interfacial', 'bulk'}: DMI pattern is a fixed spurion."""
    I = torch.eye(3, dtype=dtype)
    if kind is None: return J0*I, J0*I
    ex, ey = torch.tensor([1., 0, 0], dtype=dtype), torch.tensor([0., 1, 0], dtype=dtype)
    Dx, Dy = (D*ey, -D*ex) if kind == 'interfacial' else (D*ex, D*ey)
    eps = torch.zeros(3, 3, 3, dtype=dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1; eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
    A = lambda d: torch.einsum('abc,a->bc', eps, d)
    return J0*I + A(Dx), J0*I + A(Dy)

class SpinVAE(nn.Module):
    """coupled=False: independent (local-field) decoder. coupled=True: + global J0 (and D if dmi given)."""
    def __init__(self, L, coupled, K1=2, K0=2, C=8, n_down=0, global_pool=True, spur=None, dmi=None, J_init=1.0):
        super().__init__()
        self.L, self.coupled, self.dmi = L, coupled, dmi
        self.spur = None if spur is None else torch.as_tensor(spur, dtype=torch.float32)
        n_sp = 0 if spur is None else len(spur)
        self.enc = Encoder(K1, K0, C, n_down=n_down, n_sp=n_sp, global_pool=global_pool)
        self.dec = Decoder(K1, K0, C, n_sp=n_sp)
        self.J0 = nn.Parameter(torch.tensor(float(J_init))) if coupled else None
        self.D = nn.Parameter(torch.tensor(0.0)) if (coupled and dmi) else None

    def encode(self, S):
        return self.enc(S, self.spur)

    def sample(self, mu1, mu0, lv0, lv1, stochastic=True):
        if not stochastic: return mu1, mu0
        return (mu1 + torch.exp(0.5*lv1).unsqueeze(2)*torch.randn_like(mu1), mu0 + torch.exp(0.5*lv0)*torch.randn_like(mu0))

    def bonds(self, dtype=torch.float32):
        if not self.coupled: return None, None
        return bond_tensors(self.J0, self.D if self.D is not None else 0., self.dmi, dtype)

    def kl_per_dim(self, mu1, mu0, lv0, lv1):
        """(K0 + K1,) batch-mean KL per latent channel, summed over grid positions."""
        k0 = 0.5*(mu0**2 + lv0.exp() - 1 - lv0)
        k1 = 0.5*((mu1**2).sum(2) + 3*lv1.exp() - 3 - 3*lv1)
        return torch.cat([k0.sum((-1, -2)), k1.sum((-1, -2))], 1).mean(0)

    def losses(self, S, beta=1.0, free_bits=0.0, stochastic=True, detach_J=False):
        mu1, mu0, lv0, lv1 = self.encode(S)
        z1, z0 = self.sample(mu1, mu0, lv0, lv1, stochastic)
        h = self.dec(z1, z0, self.L, self.spur)
        Jx, Jy = self.bonds(S.dtype)
        if self.coupled:
            rec = TL.coupled_pl_loss(S, h, Jx.detach(), Jy.detach()) if detach_J else TL.coupled_pl_loss(S, h, Jx, Jy)
            # SM objective for the global couplings alone: gradient only into J0 / D, h treated as given
            sm = TL.score_matching_loss(S, h.detach(), Jx, Jy)
        else:
            rec = TL.vmf_nll(S, h).sum((-1, -2)); sm = None
        kl = self.kl_per_dim(mu1, mu0, lv0, lv1)
        kl_term = kl.clamp_min(free_bits).sum()
        return dict(rec=rec.mean(), sm=None if sm is None else sm.mean(), kl=kl, loss=rec.mean() + beta*kl_term)
