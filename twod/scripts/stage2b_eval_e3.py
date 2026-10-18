"""Stage 2b-C addendum: does the E3 independent decoder carry texture? Fourier-phase lock R(k) of decoded h_z with S_z
(validation chains) and the spatial modulation of decoded h_z, for every E3_indep_s* model."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin import probes                  # noqa
from s2spin.vae import SpinVAE             # noqa
out = 'results'; z = np.load(f'{out}/stage2/E3.npz'); S = torch.tensor(z['spins']); L = S.shape[1]
va = np.where(z['chain_id'] % 5 == 0)[0]; Xv = S[va]; Sz = Xv[..., 2].numpy().astype(float); ks = probes.texture_peaks(Sz, 3)
res = dict(peaks=ks)
for d in sorted(glob.glob(f'{out}/stage2b/E3_indep_s*')):
    m = SpinVAE(L, False, spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=True, n_down=2)
    m.load_state_dict(torch.load(f'{d}/model.pt')); m.eval()
    with torch.no_grad():
        mu1, mu0, _, _ = m.encode(Xv); hz = m.decode(mu1, mu0)[..., 2].numpy().astype(float)
    res[os.path.basename(d)] = dict(phase_lock=probes.phase_lock(Sz, hz, ks), chance=probes.phase_lock(Sz, np.roll(hz, 1, axis=0), ks),
                                    hz_spatial_std_over_mean=float(np.mean(hz.std((1, 2))/np.abs(hz.mean((1, 2))))))
json.dump(res, open(f'{out}/stage2b/E3_indep_phase.json', 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
