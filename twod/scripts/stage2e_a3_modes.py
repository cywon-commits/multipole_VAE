"""Stage 2e-A3: identity of the top-SNR latent dimensions of the beta=1 continued E2 independent solutions. Decoder response
dh_k = dh/dz_k (central difference around the posterior mean) projected on {local translation d_x h, d_y h}, {local breathing
((r - r_core).grad) h around the nearest core of the decoded h_z}, rest. Top-20 SNR dims vs 20 random dims."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from scipy.ndimage import gaussian_filter, maximum_filter

out = 'results'; z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; va = np.where(z['chain_id'] % 5 == 0)[0][:24]
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]; eps = 0.05; rng = np.random.default_rng(0)
res = {}
for s in (0, 1):
    m = SpinVAE(L, False, spur=spur, dmi='interfacial', global_pool=False, n_down=3); m.load_state_dict(torch.load(f'{out}/stage2b/E2_ramp_s{s}/model.pt')); m.eval()
    spec = np.load(f'{out}/stage2e/spec_b1_s{s}.npy'); K0, K1, g = m.enc.K0, m.enc.K1, 6
    with torch.no_grad():
        mu1, mu0, _, _ = m.encode(S[va]); h0 = m.decode(mu1, mu0).numpy().astype(float)
    # basis fields per sample
    gx = (np.roll(h0, -1, 1) - np.roll(h0, 1, 1))/2; gy = (np.roll(h0, -1, 2) - np.roll(h0, 1, 2))/2
    br = np.zeros_like(h0); xs = np.arange(L)
    for i in range(len(va)):
        hz = gaussian_filter(h0[i, ..., 2], 1.0, mode='wrap'); pk = (hz == maximum_filter(hz, size=7, mode='wrap')) & (hz > hz.mean() + hz.std())
        cs = np.argwhere(pk)
        if len(cs) == 0: continue
        X, Y = np.meshgrid(xs, xs, indexing='ij'); best = None
        dx = np.stack([((X - c[0] + L//2) % L) - L//2 for c in cs]); dy = np.stack([((Y - c[1] + L//2) % L) - L//2 for c in cs]); k = (dx**2 + dy**2).argmin(0)
        rx = np.take_along_axis(dx, k[None], 0)[0]; ry = np.take_along_axis(dy, k[None], 0)[0]
        br[i] = rx[..., None]*gx[i] + ry[..., None]*gy[i]
    def respond(idx):
        a1, a0 = mu1.clone(), mu0.clone(); b1, b0 = mu1.clone(), mu0.clone()
        if idx < K0*g*g:
            c, r = divmod(idx, g*g); i, j = divmod(r, g); a0[:, c, i, j] += eps; b0[:, c, i, j] -= eps
        else:
            q = idx - K0*g*g; c, r = divmod(q, 3*g*g); comp, r = divmod(r, g*g); i, j = divmod(r, g); a1[:, c, comp, i, j] += eps; b1[:, c, comp, i, j] -= eps
        with torch.no_grad(): return ((m.decode(a1, a0) - m.decode(b1, b0))/(2*eps)).numpy().astype(float)
    def frac(idx):
        dh = respond(idx); fr = []
        for i in range(len(va)):
            v = dh[i].ravel(); nv = v @ v
            if nv < 1e-12: continue
            Bm = np.stack([gx[i].ravel(), gy[i].ravel(), br[i].ravel()], 1)
            def ex(cols):
                A = Bm[:, cols]; c = np.linalg.lstsq(A, v, rcond=None)[0]; return float(((A@c)**2).sum()/nv)
            fr.append((ex([0, 1]), ex([2]), ex([0, 1, 2])))
        return np.mean(fr, 0) if fr else np.full(3, np.nan), float(np.sqrt(nv))
    order = np.argsort(spec)[::-1]; top = order[:20]; rnd = rng.choice(288, 20, replace=False)
    r = {}
    for tag, ids in (('top20', top), ('random20', rnd)):
        F = np.array([frac(int(k))[0] for k in ids]); r[tag] = dict(translation=float(np.nanmean(F[:, 0])), breathing=float(np.nanmean(F[:, 1])), both=float(np.nanmean(F[:, 2])), rest=float(1 - np.nanmean(F[:, 2])),
                                                                        KL_mean=float(spec[ids].mean()), dims=[int(k) for k in ids])
    res[f's{s}'] = r; print('seed', s, {t: {k: round(v, 3) for k, v in d.items() if k not in ('dims',)} for t, d in r.items()}, flush=True)
json.dump(res, open(f'{out}/stage2e/A3_modes.json', 'w'), indent=1, default=float)
