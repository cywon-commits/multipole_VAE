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
    def __init__(self, K1=2, K0=2, C=8, n_layers=3, n_down=0, n_sp=0, global_pool=True, mask_in=False):
        super().__init__()
        self.n_sp, self.n_down, self.global_pool, self.K1, self.K0, self.mask_in = n_sp, n_down, global_pool, K1, K0, mask_in
        cv, cs = 1 + n_sp, 1 + int(mask_in)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(VNConv(cv, cs, C, C)); cv, cs = C, C
        self.mu1 = nn.Conv2d(C, K1, 1, bias=False)
        self.head = nn.Conv2d(C + C, 2*K0 + K1, 1)          # from [s, |V|] -> mu0, logvar0, logvar1

    def forward(self, S, spur=None, mask=None):
        B, H, W, _ = S.shape
        V = S.permute(0, 3, 1, 2).unsqueeze(1)
        if self.n_sp: V = torch.cat([V, spurion_channels(spur, B, H, W, S.dtype, S.device)], 1)
        s = torch.ones(B, 1, H, W, dtype=S.dtype, device=S.device)
        if self.mask_in: s = torch.cat([s, mask.to(S.dtype).unsqueeze(1)], 1)
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
    def __init__(self, K1=2, K0=2, C=8, n_layers=3, n_sp=0, mask_ch=False):
        super().__init__()
        self.n_sp, self.mask_ch = n_sp, mask_ch
        cv, cs = K1 + n_sp, K0 + int(mask_ch)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(VNConv(cv, cs, C, C)); cv, cs = C, C
        self.out = nn.Conv2d(C, 1, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))        # overall field magnitude (invariant)

    def forward(self, z1, z0, L, spur=None, mask=None):
        """z1 (B,K1,3,g,g), z0 (B,K0,g,g) -> h (B, L, L, 3). mask (B,L,L) in {0,1}: extra scalar input channel."""
        B = z1.shape[0]; g = z1.shape[-1]
        V = z1
        if g != L:
            V = F.interpolate(V.reshape(B, -1, g, g), size=(L, L), mode='nearest').reshape(B, z1.shape[1], 3, L, L)
            s = F.interpolate(z0, size=(L, L), mode='nearest')
        else:
            s = z0
        if self.mask_ch: s = torch.cat([s, mask.to(s.dtype).unsqueeze(1)], 1)
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
    def __init__(self, L, coupled, K1=2, K0=2, C=8, n_down=0, global_pool=True, spur=None, dmi=None, J_init=1.0,
                 given_B=False, mask_ch=False, enc_mask=False, mask_mode='shared'):
        super().__init__()
        self.L, self.coupled, self.dmi = L, coupled, dmi
        self.spur = None if spur is None else torch.as_tensor(spur, dtype=torch.float32)
        n_sp = 0 if spur is None else len(spur)
        self.enc = Encoder(K1, K0, C, n_down=n_down, n_sp=n_sp, global_pool=global_pool, mask_in=enc_mask)
        self.enc_mask, self.mask_mode = enc_mask, mask_mode
        self.given_B, self.mask_ch = given_B, mask_ch
        self.dec = Decoder(K1, K0, C, n_sp=n_sp + int(given_B), mask_ch=mask_ch)
        self.J0 = nn.Parameter(torch.tensor(float(J_init))) if coupled else None
        self.D = nn.Parameter(torch.tensor(0.0)) if (coupled and dmi) else None
        if mask_mode in ('m1hm', 'm2hm'):                          # separate uniform field for masked sites
            self.hm = nn.Parameter(torch.tensor([0., 0., 1.5]))
        if mask_mode in ('m2', 'm2hm'):                             # separate couplings for masked sites
            self.J0m = nn.Parameter(torch.tensor(float(J_init))); self.Dm = nn.Parameter(torch.tensor(0.0)) if dmi else None

    def dec_spur(self, B=None, n=None):
        """Decoder spurions: the fixed ones, plus (given_B) the per-sample field vector B z_hat."""
        if not self.given_B: return self.spur
        fixed = self.spur.expand(n, -1, -1) if self.spur is not None else torch.zeros(n, 0, 3)
        Bz = torch.stack([torch.zeros_like(B), torch.zeros_like(B), B], -1)[:, None, :]
        return torch.cat([Bz, fixed], 1)

    def decode(self, z1, z0, B=None, mask=None):
        return self.dec(z1, z0, self.L, self.dec_spur(B, z1.shape[0]), mask)

    def encode(self, S, mask=None):
        return self.enc(S, self.spur, mask)

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

    def losses(self, S, beta=1.0, free_bits=0.0, stochastic=True, detach_J=False, B=None, mask=None, zero_z=False):
        mu1, mu0, lv0, lv1 = self.encode(S, mask if self.enc_mask else None)
        z1, z0 = self.sample(mu1, mu0, lv0, lv1, stochastic)
        if zero_z: z1, z0 = torch.zeros_like(z1), torch.zeros_like(z0)
        h = self.decode(z1, z0, B, mask)
        Jx, Jy = self.bonds(S.dtype)
        rec_m = rec_u = None
        if self.coupled and mask is not None:
            # sites in M: drop neighbours that are also in M; sites outside M: ordinary pseudo-likelihood
            Jx_, Jy_ = (Jx.detach(), Jy.detach()) if detach_J else (Jx, Jy)
            m = mask.to(S.dtype)
            g_full = TL.effective_field(S, h, Jx_, Jy_)
            if self.mask_mode == 'm1':                                     # masked sites: independent vMF(h), no bonds
                g_mask = h
            elif self.mask_mode == 'm1hm':                                 # masked sites: independent vMF(h_m), h_m learnable and uniform
                g_mask = self.hm.expand_as(h)
            elif self.mask_mode == 'm2hm':                                 # own field h_m and own couplings, observed neighbours only
                Jxm, Jym = bond_tensors(self.J0m, self.Dm if self.Dm is not None else 0., self.dmi, S.dtype)
                g_mask = TL.effective_field(S*(1 - m).unsqueeze(-1), self.hm.view(1, 1, 1, 3), Jxm, Jym)
            elif self.mask_mode == 'm2':                                   # masked sites: own couplings, observed neighbours only
                Jxm, Jym = bond_tensors(self.J0m, self.Dm if self.Dm is not None else 0., self.dmi, S.dtype)
                g_mask = TL.effective_field(S*(1 - m).unsqueeze(-1), h, Jxm, Jym)
            else:
                g_mask = TL.effective_field(S*(1 - m).unsqueeze(-1), h, Jx_, Jy_)      # h + bonds from unmasked neighbours
            g = torch.where(m.unsqueeze(-1) > 0, g_mask, g_full)
            nll = TL.vmf_nll(S, g)
            rec = nll.sum((-1, -2)); rec_m = (nll*m).sum((-1, -2)); rec_u = (nll*(1 - m)).sum((-1, -2))
            sm = None
        elif self.coupled:
            rec = TL.coupled_pl_loss(S, h, Jx.detach(), Jy.detach()) if detach_J else TL.coupled_pl_loss(S, h, Jx, Jy)
            # SM objective for the global couplings alone: gradient only into J0 / D, h treated as given
            sm = TL.score_matching_loss(S, h.detach(), Jx, Jy)
        else:
            rec = TL.vmf_nll(S, h).sum((-1, -2)); sm = None
        kl = self.kl_per_dim(mu1, mu0, lv0, lv1)
        kl_term = kl.clamp_min(free_bits).sum()
        return dict(rec=rec.mean(), sm=None if sm is None else sm.mean(), kl=kl, loss=rec.mean() + beta*kl_term,
                    rec_masked=None if rec_m is None else rec_m.mean(), rec_unmasked=None if rec_u is None else rec_u.mean())
