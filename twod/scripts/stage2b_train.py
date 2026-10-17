"""Stage 2b training/diagnostics: E1 robustness (converge-to-tolerance, several seeds), E3 control-parameter
mixture (coupled decoder should put B in an l=0 latent channel), masked-block reconstruction on E2.
modes: indep | pl | sm_alt (see stage2_train.py); --given_B feeds the true B z_hat to the decoder (control);
--mask_l l: random l x l blocks (25%) are removed from the decoder's conditioning (encoder sees everything)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin import *                       # noqa
from s2spin import probes                  # noqa
from s2spin.vae import SpinVAE             # noqa

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--ens', default='E1'); ap.add_argument('--mode', default='pl', choices=['indep', 'pl', 'sm_alt'])
ap.add_argument('--given_B', action='store_true'); ap.add_argument('--mask_l', type=int, default=0)
ap.add_argument('--mask_frac', type=float, default=0.25)
ap.add_argument('--epochs', type=int, default=80); ap.add_argument('--converge', action='store_true')
ap.add_argument('--min_epochs', type=int, default=30); ap.add_argument('--max_epochs', type=int, default=250)
ap.add_argument('--tol', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--jlr_mult', type=float, default=10.0)
ap.add_argument('--beta', type=float, default=1.0); ap.add_argument('--free_bits', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--K1', type=int, default=2); ap.add_argument('--K0', type=int, default=2)
ap.add_argument('--C', type=int, default=8); ap.add_argument('--out', default='results'); ap.add_argument('--name', default=None)
a = ap.parse_args()
name = a.name or f"{a.ens}_{a.mode}{'_givenB' if a.given_B else ''}{f'_mask{a.mask_l}' if a.mask_l else ''}_s{a.seed}"
d = os.path.join(a.out, 'stage2b', name); os.makedirs(d, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('NT', 1)))

z = np.load(os.path.join(a.out, 'stage2', f'{a.ens}.npz'))
S = torch.tensor(z['spins']); L = S.shape[1]; n_all = len(S)
cid = z['chain_id']; val = (cid % 5 == 0)
tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
Bt = torch.tensor(z['B'], dtype=torch.float32) if 'B' in z else None
if a.ens == 'E1':
    spur, dmi, kw = None, None, dict(global_pool=True, n_down=0)
elif a.ens == 'E3':
    spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]; dmi = 'interfacial'; kw = dict(global_pool=True, n_down=2)
else:
    spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]; dmi = 'interfacial'; kw = dict(global_pool=False, n_down=3)
m = SpinVAE(L, a.mode != 'indep', K1=a.K1, K0=a.K0, C=a.C, spur=spur, dmi=dmi, given_B=a.given_B, mask_ch=a.mask_l > 0, **kw)

def make_mask(n, gen=None):
    nb = L//a.mask_l
    blk = (torch.rand(n, nb, nb, generator=gen) < a.mask_frac).float()
    mk = blk.repeat_interleave(a.mask_l, 1).repeat_interleave(a.mask_l, 2)
    sh = torch.randint(0, L, (n, 2), generator=gen)
    return torch.stack([torch.roll(mk[i], (int(sh[i, 0]), int(sh[i, 1])), (0, 1)) for i in range(n)])
vgen = torch.Generator().manual_seed(123)
Mva = make_mask(len(va_idx), vgen) if a.mask_l else None
kwargs_for = lambda idx, mk=None: dict(B=Bt[idx] if a.given_B else None, mask=mk)

Jp = [p for n_, p in m.named_parameters() if n_ in ('J0', 'D')]
mp = [p for n_, p in m.named_parameters() if n_ not in ('J0', 'D')]
opt = torch.optim.Adam([dict(params=mp, lr=a.lr)] + ([dict(params=Jp, lr=a.jlr_mult*a.lr)] if a.mode == 'pl' else []))
optJ = torch.optim.Adam(Jp, lr=a.jlr_mult*a.lr) if a.mode == 'sm_alt' else None
n_ep = a.max_epochs if a.converge else a.epochs
sched = None if a.converge else torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
hist = []; t0 = time.time(); stopped = None
track = lambda h: h['J0'] if h['J0'] is not None else h['hnorm']
for ep in range(n_ep):
    perm = np.random.permutation(tr_idx); m.train(); acc = []
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i+a.bs]; b = S[idx]; mk = make_mask(len(idx)) if a.mask_l else None
        o = m.losses(b, a.beta, a.free_bits, detach_J=(a.mode == 'sm_alt'), **kwargs_for(idx, mk))
        opt.zero_grad(); o['loss'].backward(); opt.step()
        if optJ is not None:
            sm = m.losses(b, a.beta, a.free_bits, detach_J=False, **kwargs_for(idx, mk))['sm']
            optJ.zero_grad(); sm.backward(); optJ.step()
        acc.append([o['rec'].item(), o['kl'].sum().item()])
    if sched: sched.step()
    m.eval()
    with torch.no_grad():
        ov = m.losses(S[va_idx], a.beta, a.free_bits, stochastic=False, **kwargs_for(va_idx, Mva))
        mu1, mu0, _, _ = m.encode(S[va_idx]); hv = m.decode(mu1, mu0, **{k: v for k, v in kwargs_for(va_idx, Mva).items()})
    hist.append(dict(ep=ep, rec=float(np.mean([x[0] for x in acc])), kl=float(np.mean([x[1] for x in acc])), val_rec=ov['rec'].item(),
                     hnorm=float(hv.norm(dim=-1).mean()), J0=None if m.J0 is None else m.J0.item(), D=None if m.D is None else m.D.item()))
    if ep % 5 == 0 or ep == n_ep - 1: print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hist[-1].items()}, f'{time.time()-t0:.0f}s', flush=True)
    if a.converge and ep + 1 >= max(a.min_epochs, 20):
        q = np.array([track(h) for h in hist]); w1, w0 = q[-10:].mean(), q[-20:-10].mean()
        if abs(w1 - w0)/abs(w1) < a.tol: stopped = ep + 1; break

# ---------------- diagnostics ----------------
m.eval(); res = dict(name=name, args=vars(a), hist=hist, epochs_run=len(hist), converged_at=stopped)
with torch.no_grad():
    mu1, mu0, lv0, lv1 = m.encode(S[va_idx])
    kl = m.kl_per_dim(mu1, mu0, lv0, lv1)
    res['kl_per_dim_val'] = kl.tolist(); res['kl_total'] = float(kl.sum())          # order: K0 scalars, K1 vectors
    h = m.decode(mu1, mu0, **kwargs_for(va_idx, Mva))
    res['h_norm_mean'] = float(h.norm(dim=-1).mean())
    if m.coupled: res['J0'] = m.J0.item(); res['D'] = None if m.D is None else m.D.item()
    if m.coupled and a.mask_l:
        o = m.losses(S[va_idx], a.beta, a.free_bits, stochastic=False, **kwargs_for(va_idx, Mva))
        res['val_rec_masked_per_site'] = float(o['rec_masked'])/float(Mva.sum(dim=(1, 2)).mean())
        res['val_rec_unmasked_per_site'] = float(o['rec_unmasked'])/float((1 - Mva).sum(dim=(1, 2)).mean())
        # phase locking of decoded h_z with the data S_z inside the masked region (mean over the region removed)
        Sz = S[va_idx][..., 2].numpy().astype(float); hz = h[..., 2].numpy().astype(float); mk = Mva.numpy().astype(float)
        mean_in = lambda F: (F*mk).sum((1, 2), keepdims=True)/mk.sum((1, 2), keepdims=True)
        Fd, Fm = (Sz - mean_in(Sz))*mk, (hz - mean_in(hz))*mk
        ks = probes.texture_peaks(Sz, 3)
        res['phase_lock_masked'] = probes.phase_lock(Fd, Fm, ks)
        # control: lock of unrelated pairs (shuffled samples) to show the chance level
        res['phase_lock_masked_chance'] = probes.phase_lock(Fd, np.roll(Fm, 1, axis=0), ks)

if a.ens == 'E1':
    n = torch.tensor(z['field_dir'][va_idx], dtype=torch.float32)
    res['latent_l1_abs_cos_with_n'] = [(torch.nn.functional.normalize(mu1[:, k, :, 0, 0], dim=-1)*n).sum(-1).abs().mean().item() for k in range(a.K1)]

if a.ens == 'E3':
    B = z['B'].astype(float); sig = float(np.median(z['sigma_B'])); beta = float(z['beta'])
    def lat(stochastic):
        torch.manual_seed(7)
        with torch.no_grad():
            u = m.encode(S); z1, z0 = m.sample(*u, stochastic=stochastic)
            hh = m.decode(z1, z0, B=Bt if a.given_B else None)
        return torch.cat([z0.flatten(1), z1.flatten(1)], 1).numpy().astype(float), hh[..., 2].mean((1, 2)).numpy().astype(float)/beta
    inner = (cid % 5 == 1) & ~val; fit = ~val & ~inner
    def ridge(Xa, ya, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-8; Xs = (Xa - mu)/sd
        W = np.linalg.solve(Xs.T@Xs + lam*np.eye(Xs.shape[1]), Xs.T@(ya - ya.mean()))
        return lambda Xb: ((Xb - mu)/sd)@W + ya.mean()
    def probe(X, cols):
        Xc = X[:, cols]; lams = [1e-3, 1e-1, 10., 1000.]
        best = min(lams, key=lambda l: ((B[inner] - ridge(Xc[fit], B[fit], l)(Xc[inner]))**2).sum())
        P = ridge(Xc[~val], B[~val], best)(Xc[val])
        return float(1 - ((B[val] - P)**2).sum()/((B[val] - B[val].mean())**2).sum())
    for tag, stoch in (('mean', False), ('sampled', True)):
        X, hz = lat(stoch)
        res[f'B_probe_R2_heldout_{tag}_z'] = dict(all=probe(X, list(range(X.shape[1]))), per_scalar_channel=[probe(X, [k]) for k in range(a.K0)])
        A = np.stack([B, np.ones_like(B)], 1); coef = np.linalg.lstsq(A[~val], hz[~val], rcond=None)[0]; pred = A[val]@coef
        res[f'decoded_hz_over_beta_vs_B_{tag}_z'] = dict(slope=float(coef[0]), intercept=float(coef[1]),
            R2_val=float(1 - ((hz[val] - pred)**2).sum()/((hz[val] - hz[val].mean())**2).sum()))
    r = float(B.max() - B.min()); res['theory'] = dict(sigma_B_median=sig, KL_bound=float(np.log(r/sig) - 0.5*np.log(2*np.pi*np.e)), R2_max=float(1 - sig**2/(r**2/12)))

torch.save(m.state_dict(), os.path.join(d, 'model.pt'))
json.dump(res, open(os.path.join(d, 'result.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ('hist', 'args')}, indent=1, default=float))
