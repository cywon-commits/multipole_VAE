"""Stage 2b-B: re-analysis of the trained E2 decoders with the phase / equivariance probes (no retraining).
indep (beta=0.01, free bits 1): Fourier-phase lock R(k) of decoded h_z with the data S_z texture, and shift consistency.
coupled (beta=1): amplitude of the spatial modulation of decoded h_z at the texture wavevectors (prediction ~ 0)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin import probes                  # noqa
from s2spin.vae import SpinVAE             # noqa

out = 'results'; z = np.load(f'{out}/stage2/E2.npz')
S = torch.tensor(z['spins']); L = S.shape[1]; cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
def load(name, coupled):
    m = SpinVAE(L, coupled, spur=spur, dmi='interfacial', global_pool=False, n_down=3)
    m.load_state_dict(torch.load(f'{out}/stage2/{name}/model.pt')); return m.eval()
def decode_fn(m):
    def fn(X):
        with torch.no_grad():
            mu1, mu0, _, _ = m.encode(torch.tensor(X, dtype=torch.float32))
            return m.decode(mu1, mu0).numpy().astype(float)
    return fn
Xv = S[va].numpy().astype(float); Sz = Xv[..., 2]; ks = probes.texture_peaks(Sz, 3)
res = dict(peaks=ks, n_val=len(va))
for tag, name, coupled in (('indep_b0.01_fb1', 'E2_indep_b0.01_fb1', False), ('coupled_b1', 'E2_pl_b1.0_jlr50', True)):
    fn = decode_fn(load(name, coupled)); H = fn(Xv); hz = H[..., 2]
    r = dict(phase_lock=probes.phase_lock(Sz, hz, ks), chance=probes.phase_lock(Sz, np.roll(hz, 1, axis=0), ks))
    r['h_z_spatial_std_over_mean'] = float(np.mean(hz.std((1, 2))/np.abs(hz.mean((1, 2)))))
    r['shift_consistency'] = {str(k): v for k, v in probes.shift_consistency(fn, Xv[:60], [(1, 0), (3, 5), (8, 0), (13, 21)]).items()}
    res[tag] = r
os.makedirs(f'{out}/stage2b', exist_ok=True)
json.dump(res, open(f'{out}/stage2b/E2_reanalysis.json', 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
