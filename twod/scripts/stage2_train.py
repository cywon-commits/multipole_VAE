"""Stage 2: train the (independent | coupled) SO(3)_S-equivariant VAE on E1/E2 and run the diagnostics.
modes: indep (local-field vMF decoder), pl (coupled, J from the pseudo-ELBO), sm_alt (coupled, J learned only by
score matching, decoder by pseudo-likelihood). Writes <out>/stage2/<name>/result.json (+ model.pt)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin import *                       # noqa
from s2spin.vae import SpinVAE             # noqa
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--ens', default='E1'); ap.add_argument('--mode', default='indep', choices=['indep', 'pl', 'sm_alt'])
ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--free_bits', type=float, default=0.0); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--K1', type=int, default=2); ap.add_argument('--K0', type=int, default=2)
ap.add_argument('--C', type=int, default=8); ap.add_argument('--out', default='results'); ap.add_argument('--name', default=None)
ap.add_argument('--jlr_mult', type=float, default=10.0)
ap.add_argument('--no_mc', action='store_true')
a = ap.parse_args()
name = a.name or f"{a.ens}_{a.mode}_b{a.beta}_s{a.seed}"
d = os.path.join(a.out, 'stage2', name); os.makedirs(d, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(int(os.environ.get('NT', 4)))

z = np.load(os.path.join(a.out, 'stage2', f'{a.ens}.npz'))
S = torch.tensor(z['spins']); L = S.shape[1]
cid = z['chain_id']; val = (cid % 5 == 0)                     # hold out whole chains
tr_idx, va_idx = np.where(~val)[0], np.where(val)[0]
if a.ens == 'E1':
    spur, dmi, kw = None, None, dict(global_pool=True, n_down=0)
else:
    spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]; dmi = 'interfacial'; kw = dict(global_pool=False, n_down=3)
m = SpinVAE(L, a.mode != 'indep', K1=a.K1, K0=a.K0, C=a.C, spur=spur, dmi=dmi, **kw)
Jp = [p for n_, p in m.named_parameters() if n_ in ('J0', 'D')]
mp = [p for n_, p in m.named_parameters() if n_ not in ('J0', 'D')]
groups = [dict(params=mp, lr=a.lr)] + ([dict(params=Jp, lr=a.jlr_mult*a.lr)] if a.mode == 'pl' else [])
opt = torch.optim.Adam(groups)
optJ = torch.optim.Adam(Jp, lr=a.jlr_mult*a.lr) if a.mode == 'sm_alt' else None
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
hist = []; t0 = time.time()
for ep in range(a.epochs):
    perm = np.random.permutation(tr_idx); m.train(); acc = []
    for i in range(0, len(perm), a.bs):
        b = S[perm[i:i+a.bs]]
        o = m.losses(b, a.beta, a.free_bits, detach_J=(a.mode == 'sm_alt'))
        opt.zero_grad(); o['loss'].backward(); opt.step()
        if optJ is not None:
            sm = m.losses(b, a.beta, a.free_bits, detach_J=False)['sm']
            optJ.zero_grad(); sm.backward(); optJ.step()
        acc.append([o['rec'].item(), o['kl'].sum().item()])
    sched.step()
    m.eval()
    with torch.no_grad():
        ov = m.losses(S[va_idx], a.beta, a.free_bits, stochastic=False)
    hist.append(dict(ep=ep, rec=float(np.mean([x[0] for x in acc])), kl=float(np.mean([x[1] for x in acc])), val_rec=ov['rec'].item(),
                     J0=None if m.J0 is None else m.J0.item(), D=None if m.D is None else m.D.item()))
    if ep % 5 == 0 or ep == a.epochs - 1: print(name, hist[-1], f'{time.time()-t0:.0f}s', flush=True)

# ---------------- diagnostics ----------------
m.eval(); res = dict(name=name, args=vars(a), hist=hist)
with torch.no_grad():
    mu1, mu0, lv0, lv1 = m.encode(S[va_idx])
    res['kl_per_dim_val'] = m.kl_per_dim(mu1, mu0, lv0, lv1).tolist()      # order: K0 scalars, K1 vectors
    h = m.dec(mu1, mu0, L, m.spur)
    res['h_norm_mean'] = float(h.norm(dim=-1).mean()); res['h_norm_std_over_samples'] = float(h.norm(dim=-1).mean((1, 2)).std())
if m.coupled: res['J0'] = m.J0.item(); res['D'] = None if m.D is None else m.D.item()

# equivariance (float64 copy, spurions are rotated together with the spins in E1: none)
if a.ens == 'E1':
    m64 = SpinVAE(L, m.coupled, K1=a.K1, K0=a.K0, C=a.C, spur=spur, dmi=dmi, **kw).double(); m64.load_state_dict(m.state_dict())
    R = torch.tensor(Rotation.random(1, random_state=5).as_matrix()[0])
    Sv = S[va_idx[:16]].double()
    with torch.no_grad():
        u = m64.encode(Sv); ur = m64.encode(Sv @ R.T)
        rot = lambda V: torch.einsum('ab,nkbxy->nkaxy', R, V)
        e_mu1 = ((ur[0] - rot(u[0])).norm()/u[0].norm()).item(); e_mu0 = ((ur[1] - u[1]).norm()/u[1].norm()).item()
        hh = m64.dec(u[0], u[1], L); hr = m64.dec(rot(u[0]), u[1], L)
        e_h = ((hr - hh @ R.T).norm()/hh.norm()).item()
    res['equivariance_rel_err'] = dict(mu1=e_mu1, mu0=e_mu0, h=e_h)
    n = torch.tensor(z['field_dir'][va_idx], dtype=torch.float32)
    cos = [(torch.nn.functional.normalize(mu1[:, k, :, 0, 0], dim=-1)*n).sum(-1).abs().mean().item() for k in range(a.K1)]
    res['latent_l1_abs_cos_with_n'] = cos; res['latent_l1_norm_mean'] = [mu1[:, k].norm(dim=1).mean().item() for k in range(a.K1)]
    hm = torch.nn.functional.normalize(h.mean((1, 2)), dim=-1)
    res['h_dir_cos_with_n'] = (hm*n).sum(-1).mean().item()
    # sampling test: bath MC from the decoder's Hamiltonian
    if not a.no_mc:
        kap = res['h_norm_mean']
        Jd = res.get('J0', 0.0)
        if m.coupled:
            Hd = Hamiltonian.heisenberg_dmi(Jd, 0, None, B=(0, 0, kap))
            Sg, _ = run_mc(Hd, L, chains=100, therm=400, nsamp=10, gap=20, seed=11, init='field')
            C = observables.transverse_corr(Sg, [0, 0, 1], 6); Ct = analytic.transverse_corr_pred(L, 10.0, 5.0, 6)
            Cd = analytic.transverse_corr_pred(L, kap, Jd, 6)
            Sd = S[va_idx[:200]].numpy().astype(float); nd = z['field_dir'][va_idx[:200]]
            Cdat = np.mean([observables.transverse_corr(Sd[i:i+1], nd[i], 6) for i in range(len(Sd))], 0)
            res['sample_C'] = dict(model_MC=C.tolist(), data=Cdat.tolist(), gauss_true_2G=Ct.tolist(), gauss_decoder_params=Cd.tolist())
        else:
            # independent decoder: sites are conditionally independent vMF(h(z)); C(r>0)=0 by construction
            k = h.norm(dim=-1).mean().item()
            res['sample_C'] = dict(note='independent sites => C(r>0)=0; C(0)=<|delta|^2> ~ 2/kappa_dec', two_over_kappa=2/k)

if a.ens == 'E2':
    with torch.no_grad():
        Sall = S
        feats = []
        for i in range(0, len(S), 128):
            u = m.encode(S[i:i+128]); feats.append(torch.cat([u[0].flatten(1), u[1].flatten(1)], 1))
        X = torch.cat(feats).numpy().astype(float)
        hb = m.dec(*m.sample(*m.encode(S[va_idx]), stochastic=False), L, m.spur)
        res['h_norm_mean_E2'] = float(hb.norm(dim=-1).mean())
        res['cos_h_S_mean'] = float((torch.nn.functional.normalize(hb, dim=-1)*S[va_idx]).sum(-1).mean())
    c4 = z['c4']; Qv = z['Q'].astype(float); sh = z['shift'].astype(float)
    T = {'c4_onehot': np.eye(4)[c4], 'Q': Qv[:, None],
         'shift_cs': np.stack([np.cos(2*np.pi*sh[:, 0]/L), np.sin(2*np.pi*sh[:, 0]/L), np.cos(2*np.pi*sh[:, 1]/L), np.sin(2*np.pi*sh[:, 1]/L)], 1)}
    inner = (cid % 5 == 1) & ~val; fit = ~val & ~inner
    def ridge(Xa, Ya, lam):
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-8; Xs = (Xa - mu)/sd
        W = np.linalg.solve(Xs.T@Xs + lam*np.eye(Xs.shape[1]), Xs.T@(Ya - Ya.mean(0)))
        return lambda Xb: ((Xb - mu)/sd)@W + Ya.mean(0)
    def r2(Y, P): return float(1 - ((Y - P)**2).sum()/((Y - Y.mean(0))**2).sum())
    pr = {}
    for k, Y in T.items():
        lams = [1., 10., 100., 1000.]
        best = min(lams, key=lambda l: ((Y[inner] - ridge(X[fit], Y[fit], l)(X[inner]))**2).sum())
        f = ridge(X[~val], Y[~val], best); P = f(X[val])
        pr[k] = dict(lam=best, acc=float((P.argmax(1) == Y[val].argmax(1)).mean())) if k == 'c4_onehot' else dict(lam=best, R2_heldout=r2(Y[val], P))
    # shift is only defined modulo the lattice period of the texture; also report baseline chance for c4 = 0.25
    res['linear_probe_heldout_chains'] = pr
torch.save(m.state_dict(), os.path.join(d, 'model.pt'))
json.dump(res, open(os.path.join(d, 'result.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ('hist', 'args')}, indent=1, default=float))
