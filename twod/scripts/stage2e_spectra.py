"""Stage 2e-A1: per-latent-dimension KL spectra (validation samples) of the continued-training solutions of the E2 independent
decoder at beta in {0.01, 0.03, 0.1, 0.3, 1, 3}; saves results/stage2e/spec_b{beta}_s{seed}.npy (288 dims: 72 scalar dims
[K0=2 channels x 36 cells] followed by 216 vector components [K1=2 x 3 comps x 36 cells]) and the PCA-based variant
(information of each posterior-mean principal component, 1/2 ln(Var(mu_k)/<sigma_k^2> + 1))."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa

out = 'results'; os.makedirs(f'{out}/stage2e', exist_ok=True)
z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; va = np.where(z['chain_id'] % 5 == 0)[0]
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
src = lambda b, s: {0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}')
for s in (0, 1):
    for b in (0.01, 0.03, 0.1, 0.3, 1, 3):
        n = src(b, s)
        m = SpinVAE(L, False, spur=spur, dmi='interfacial', global_pool=False, n_down=3); m.load_state_dict(torch.load(f'{out}/stage2b/{n}/model.pt')); m.eval()
        with torch.no_grad():
            mu1, mu0, lv0, lv1 = m.encode(S[va])
            k0 = (0.5*(mu0**2 + lv0.exp() - 1 - lv0)).mean(0).flatten()
            k1 = (0.5*(mu1**2 + lv1.exp().unsqueeze(2) - 1 - lv1.unsqueeze(2))).mean(0).flatten()
            spec = torch.cat([k0, k1]).numpy().astype(float)
            # PCA variant: posterior-mean covariance eigenbasis; information per component
            feats = torch.cat([mu0.flatten(1), mu1.flatten(1)], 1).numpy().astype(float); sg2 = torch.cat([lv0.exp().expand_as(mu0).flatten(1), lv1.exp().unsqueeze(2).expand_as(mu1).flatten(1)], 1).mean(0).numpy().astype(float)
        C = np.cov(feats.T); w, V = np.linalg.eigh(C); w = np.maximum(w[::-1], 0); V = V[:, ::-1]
        sg_rot = np.einsum('ik,i,ik->k', V, sg2, V)             # posterior variance projected on each eigenvector (diagonal approx.)
        pca = 0.5*np.log(w/np.maximum(sg_rot, 1e-12) + 1)
        np.save(f'{out}/stage2e/spec_b{b}_s{s}.npy', spec); np.save(f'{out}/stage2e/pca_b{b}_s{s}.npy', pca)
        print(n, 'beta', b, 'dims', len(spec), 'KL sum %.1f' % spec.sum(), 'active(>0.02) %d' % (spec > 0.02).sum(), 'pca info sum %.1f active %d' % (pca.sum(), (pca > 0.02).sum()), flush=True)
