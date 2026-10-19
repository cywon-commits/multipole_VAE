"""Stage 2d-B3: SNR of the texture-phase mode. Shift the decoded field (posterior mean) by a sub-lattice delta
(Fourier interpolation) along x and y and measure the curvature of the mean reconstruction loss at delta = 0:
I_shift = d^2 rec / d delta^2 (per lattice spacing^2); SNR_phase = I_shift * p^2/12 with p the SkX period.
usage: python stage2d_b3_shift.py <run name under results/stage2b> [indep|m1hm]"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import torch_losses as TL, probes   # noqa

name = sys.argv[1]; out = 'results'
z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]
r = json.load(open(f'{out}/stage2b/{name}/result.json')); a = r['args']
m = SpinVAE(L, a['mode'] != 'indep', spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=False, n_down=3,
            mask_ch=a['mask_l'] > 0, enc_mask=a.get('enc_mask', False), mask_mode=a.get('mask_mode', 'shared'))
m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval()
assert a['mask_l'] == 0, 'B3 is written for the unmasked independent decoder'
with torch.no_grad():
    mu1, mu0, _, _ = m.encode(S[va]); h = m.decode(mu1, mu0).double()
kx = torch.fft.fftfreq(L).double()
def shifted(h, dx, dy):
    F = torch.fft.fft2(h, dim=(1, 2))
    ph = torch.exp(-2j*np.pi*(kx[:, None]*dx + kx[None, :]*dy))
    return torch.fft.ifft2(F*ph[None, :, :, None], dim=(1, 2)).real
def rec(hh): return TL.vmf_nll(S[va].double(), hh).sum((1, 2)).mean().item()
ks = probes.texture_peaks(S[va][..., 2].numpy().astype(float), 3); p = float(np.mean([L/np.hypot(*k) for k in ks]))
res = dict(name=name, KL_total=r['kl_total'], period_p=p, peaks=ks)
for ax, (fx, fy) in (('x', (1, 0)), ('y', (0, 1))):
    d = 0.05; r0 = rec(shifted(h, 0, 0)); I = (rec(shifted(h, fx*d, fy*d)) + rec(shifted(h, -fx*d, -fy*d)) - 2*r0)/d**2
    res[f'I_shift_{ax}'] = I; res[f'SNR_phase_{ax}'] = I*p**2/12
    res[f'rec_shift_scan_{ax}'] = {str(dd): rec(shifted(h, fx*dd, fy*dd)) - r0 for dd in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)}
json.dump(res, open(f'{out}/stage2b/{name}_B3.json', 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
