"""Stage 2e-B: read the continued K+M1 (l=8, rho=0.5) beta=1 solutions: J, D bias, h_z profile, phase lock of the decoded masked-region
field with the data texture, and the position relative to the (h_z, J, D) valley line."""
import json, os, sys
import numpy as np, torch
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import langevin, probes        # noqa

out = 'results'; z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; va = np.where(z['chain_id'] % 5 == 0)[0]
Sz = S[va][..., 2].numpy().astype(float); spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
def make_mask(n, l, frac, gen):
    nb = L//l; blk = (torch.rand(n, nb, nb, generator=gen) < frac).float()
    mk = blk.repeat_interleave(l, 1).repeat_interleave(l, 2); sh = torch.randint(0, L, (n, 2), generator=gen)
    return torch.stack([torch.roll(mk[i], (int(sh[i, 0]), int(sh[i, 1])), (0, 1)) for i in range(n)])
def cheb(M, dmax=12):
    d = np.where(M, 0, dmax).astype(int); cur = M.copy()
    for k in range(1, dmax):
        cur = cur | np.roll(cur, 1, 1) | np.roll(cur, -1, 1) | np.roll(cur, 1, 2) | np.roll(cur, -1, 2)
        cur = cur | np.roll(cur, 1, 1) | np.roll(cur, -1, 1); d = np.where((d == dmax) & cur, k, d)
    return d
res = {}
for name in ['E2_K1_ramp_s0', 'E2_K1_ramp_s1', 'E2_K_m1_l8_r0.5_s0', 'E2_K_m1_l8_r0.5_s1']:
    r = json.load(open(f'{out}/stage2b/{name}/result.json')); a = r['args']
    m = SpinVAE(L, True, spur=spur, dmi='interfacial', global_pool=False, n_down=3, mask_ch=True, enc_mask=a.get('enc_mask', False), mask_mode=a['mask_mode'])
    m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval()
    M = make_mask(len(va), a['mask_l'], a['mask_frac'], torch.Generator().manual_seed(123)); Mb = M.numpy() > 0
    with torch.no_grad():
        mu1, mu0, _, _ = m.encode(S[va], M); h = m.decode(mu1, mu0, mask=M)
        zero = m.decode(torch.zeros_like(mu1), torch.zeros_like(mu0), mask=M)
    hz = h[..., 2].numpy().astype(float); dist = cheb(Mb)
    far = float(hz[(dist >= 8) & ~Mb].mean()); unm = float(hz[~Mb].mean()); msk = float(hz[Mb].mean())
    prof = {int(k): float(hz[(dist == k) & ~Mb].mean()) for k in range(1, 9) if ((dist == k) & ~Mb).any()}
    mean_in = lambda F: (F*Mb).sum((1, 2), keepdims=True)/Mb.sum((1, 2), keepdims=True)
    ks = probes.texture_peaks(Sz, 3); Fd, Fm = (Sz - mean_in(Sz))*Mb, (hz - mean_in(hz))*Mb
    pl = probes.phase_lock(Fd, Fm, ks); ch = probes.phase_lock(Fd, np.roll(Fm, 1, axis=0), ks)
    # correlation of the decoded masked-region field with the data inside the mask (same samples)
    cm = float(np.corrcoef(Fd[Mb], Fm[Mb])[0, 1])
    delta = 2.4 - far
    res[name] = dict(KL=r['kl_total'], J=r['J0'], D=r['D'], dJ=100*(r['J0']/6 - 1), dD=100*(r['D']/6 - 1), h_masked=msk, h_unmasked=unm, h_far=far, deficit_far=delta,
                     h_profile=prof, valley_pred_dJ_4p7=4.7*delta, valley_pred_dD_m13p3=-13.3*delta, R_k_masked=[p['R'] for p in pl], R_k_chance=[p['R'] for p in ch],
                     corr_masked_field_vs_data=cm, h_masked_spatial_std=float(hz[Mb].std()), hstd_all=float(hz.std()))
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items() if k != 'h_profile'}), flush=True)
json.dump(res, open(f'{out}/stage2e/B_kplus.json', 'w'), indent=1, default=float)
