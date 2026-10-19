"""Stage 2d-A1: spatial structure of the decoder field h_z(r) in the trained mask runs (no retraining), and the
regression of the J, D bias on the field deficit of the unmasked region."""
import glob, json, os, sys
import numpy as np, torch
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import langevin                # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

out = 'results'; z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]
cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]; Sz = S[va][..., 2].numpy().astype(float)
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
def make_mask(n, l, frac, gen):
    nb = L//l; blk = (torch.rand(n, nb, nb, generator=gen) < frac).float()
    mk = blk.repeat_interleave(l, 1).repeat_interleave(l, 2); sh = torch.randint(0, L, (n, 2), generator=gen)
    return torch.stack([torch.roll(mk[i], (int(sh[i, 0]), int(sh[i, 1])), (0, 1)) for i in range(n)])
def cheb_dist(M, dmax=12):
    """Chebyshev distance of every site to the nearest masked site (periodic), capped at dmax."""
    d = np.where(M, 0, dmax).astype(int); cur = M.copy()
    for k in range(1, dmax):
        cur = cur | np.roll(cur, 1, 1) | np.roll(cur, -1, 1) | np.roll(cur, 1, 2) | np.roll(cur, -1, 2)
        cur = cur | np.roll(cur, 1, 1) | np.roll(cur, -1, 1)           # keep Chebyshev (8-neighbour) growth
        # (above already dilates along both axes; the two passes make the footprint a square)
        d = np.where((d == dmax) & cur, k, d)
    return d
R_RF = 3                                                      # decoder: three 3x3 convolutions on the full-resolution grid
names = sorted(glob.glob(f'{out}/stage2b/E2_pl_mask*_s*')) + sorted(glob.glob(f'{out}/stage2b/E2_K_m1_*')) + \
        sorted(glob.glob(f'{out}/stage2b/E2_U_m1_*')) + sorted(glob.glob(f'{out}/stage2b/E2_K_m2_*'))
rows = []
for d_ in names:
    name = os.path.basename(d_)
    if 'nowarm' in name or 'r1.0' in name or not os.path.exists(f'{d_}/model.pt'): continue
    r = json.load(open(f'{d_}/result.json')); a = r['args']
    m = SpinVAE(L, True, spur=spur, dmi='interfacial', global_pool=False, n_down=3, mask_ch=True, enc_mask=a.get('enc_mask', False),
                mask_mode=a.get('mask_mode', 'shared'))
    m.load_state_dict(torch.load(f'{d_}/model.pt')); m.eval()
    M = make_mask(len(va), a['mask_l'], a['mask_frac'], torch.Generator().manual_seed(123))
    with torch.no_grad():
        mu1, mu0, _, _ = m.encode(S[va], M if a.get('enc_mask', False) else None)
        h = m.decode(mu1, mu0, mask=M)
    hz = h[..., 2].numpy().astype(float); Mb = M.numpy() > 0
    dist = cheb_dist(Mb)
    prof = {int(k): float(hz[(dist == k) & ~Mb].mean()) for k in range(1, 9) if ((dist == k) & ~Mb).any()}
    far = float(hz[(dist >= 8) & ~Mb].mean()) if ((dist >= 8) & ~Mb).any() else float('nan')
    unm = float(hz[~Mb].mean()); msk = float(hz[Mb].mean())
    hm_star = brentq(lambda k: langevin(k) - Sz[Mb].mean(), 1e-3, 100)
    f_near = float(((dist <= R_RF) & ~Mb).sum()/(~Mb).sum())
    rows.append(dict(name=name, mask_l=a['mask_l'], rho=a['mask_frac'], mode=a.get('mask_mode', 'shared'), enc_mask=a.get('enc_mask', False),
                     J=r['J0'], D=r['D'], dJ=100*(r['J0']/6 - 1), dD=100*(r['D']/6 - 1), h_masked=msk, h_unmasked=unm, h_far=far,
                     h_profile=prof, h_m_star=float(hm_star), f_near=f_near, deficit=2.4 - unm, h_std_all=float(hz.std())))
    print(name, 'mode', rows[-1]['mode'], 'h_masked %.3f unmasked %.3f far %.3f (h_m*=%.3f) deficit %.3f f_near %.2f dJ %+.2f%% dD %+.2f%% hstd %.3f' % (msk, unm, far, hm_star, 2.4 - unm, f_near, rows[-1]['dJ'], rows[-1]['dD'], rows[-1]['h_std_all']), flush=True)
json.dump(rows, open(f'{out}/stage2b/stage2d_A1_profiles.json', 'w'), indent=1, default=float)
# regression of the shared-J runs (Stage 2b shared + M1 + U) on the field deficit and f_near
for label, sel in (('all runs with shared J (shared,m1)', [r for r in rows if r['mode'] in ('shared', 'm1')]), ('m1 only', [r for r in rows if r['mode'] == 'm1'])):
    x = np.array([r['deficit'] for r in sel]); f = np.array([r['f_near'] for r in sel])
    for tgt in ('dJ', 'dD'):
        y = np.array([r[tgt] for r in sel]); sl, ic = np.polyfit(x, y, 1)
        print(label, tgt, 'vs deficit: r=%.2f slope=%.2f %%/unit intercept=%.2f%% | vs f_near: r=%.2f' % (np.corrcoef(x, y)[0, 1], sl, ic, np.corrcoef(f, y)[0, 1]))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for r in rows:
    if r['mask_l'] == 8 and r['rho'] in (0.25, 0.5) and r['mode'] in ('shared', 'm1') and r['name'].endswith('s0'):
        ds = sorted(r['h_profile']); ax[0].plot([0] + ds, [r['h_masked']] + [r['h_profile'][k] for k in ds], 'o-', label=f"{r['name']}")
ax[0].axhline(2.4, c='k', ls='--', lw=0.8); ax[0].set_xlabel('distance d to nearest masked site (0 = masked)'); ax[0].set_ylabel('<h_z>'); ax[0].legend(fontsize=6)
sel = [r for r in rows if r['mode'] in ('shared', 'm1')]
ax[1].scatter([r['deficit'] for r in sel], [r['dJ'] for r in sel], c=[0 if r['mode'] == 'shared' else 1 for r in sel])
ax[1].set_xlabel('field deficit 2.4 - <h_z>_unmasked'); ax[1].set_ylabel('J bias [%]')
fig.tight_layout(); fig.savefig(f'{out}/fig/stage2d_A1.png', dpi=130)
