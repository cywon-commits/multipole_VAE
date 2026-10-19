"""Stage 2d-C: (a) effective SNR of the toy VAE that reproduces each converged E3 run, (b) Bayes-optimality of the decoder
(decoded h_z/beta vs a binned E[B|z]), (c) posterior sigma(B) profile of the used channel."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from stage2d_toy_vae import solve          # noqa

out = 'results'; z = np.load(f'{out}/stage2/E3.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; cid = z['chain_id']; B = z['B'].astype(float)
beta = float(z['beta']); res = {}
for name in sys.argv[1:]:
    r = json.load(open(f'{out}/stage2b/{name}/result.json'))
    m = SpinVAE(L, True, spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=True, n_down=2)
    m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval()
    kl = np.array(r['kl_per_dim_val']); used = int(kl.argmax()); is_vec = used >= m.enc.K0
    torch.manual_seed(3)
    with torch.no_grad():
        mu1, mu0, lv0, lv1 = m.encode(S); z1, z0 = m.sample(mu1, mu0, lv0, lv1, True)
        hz = m.decode(z1, z0)[..., 2].mean((1, 2)).numpy().astype(float)/beta
        if is_vec: k = used - m.enc.K0; zz = z1[:, k, 2, 0, 0].numpy().astype(float); sig = lv1[:, k, 0, 0].mul(0.5).exp().numpy().astype(float)
        else: zz = z0[:, used, 0, 0].numpy().astype(float); sig = lv0[:, used, 0, 0].mul(0.5).exp().numpy().astype(float)
    # (a) toy SNR scan
    obs = (r['B_probe_R2_heldout_sampled_z']['all'], float(kl.max()))
    scan = [(s, solve('uniform', s, is_vec, 1.0)) for s in np.arange(10, 19, 1.0)]
    best = min(scan, key=lambda t: ((t[1]['R2_sample'] - obs[0])/0.015)**2 + ((t[1]['KL'] - obs[1])/0.05)**2)
    # (b) Bayes check: bin z into quantile bins, E[B|z] = bin mean of B; compare with decoded h_z/beta
    q = np.quantile(zz, np.linspace(0, 1, 21)); idx = np.clip(np.digitize(zz, q[1:-1]), 0, 19)
    EB = np.array([B[idx == i].mean() for i in range(20)])[idx]; Hb = np.array([hz[idx == i].mean() for i in range(20)])[idx]
    rms_dec_vs_EB = float(np.sqrt(np.mean((hz - EB)**2))); rms_binned_dec_vs_EB = float(np.sqrt(np.mean((Hb - EB)**2)))
    # (c) sigma(B) profile
    edges = np.quantile(B, np.linspace(0, 1, 6)); bi = np.clip(np.digitize(B, edges[1:-1]), 0, 4); sp = [float(sig[bi == i].mean()) for i in range(5)]
    res[name] = dict(l=1 if is_vec else 0, obs_R2=obs[0], obs_KL=obs[1], toy_best_snr=float(best[0]), toy_R2=best[1]['R2_sample'], toy_KL=best[1]['KL'],
                     rms_decoded_minus_E_B_given_z=rms_dec_vs_EB, rms_binned_decoded_minus_E_B_given_z=rms_binned_dec_vs_EB, rms_B_total=float(B.std()),
                     sigma_by_B_quintile=sp, sigma_ratio_max_min=max(sp)/min(sp))
    print(name, json.dumps(res[name], default=float), flush=True)
json.dump(res, open(f'{out}/stage2b/stage2d_C_checks.json', 'w'), indent=1, default=float)
