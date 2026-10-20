"""Inference-only (no training) re-computation of per-sample values from the stored checkpoints, for Fig. 4a/4c/4d/4e, Fig. S3, Fig. S5:
 E1: |cos| between the used l=1 latent (posterior mean) and the true field direction n (validation chains) for E1_{indep,pl,sm_alt}_s{0,1,2};
 E3: per validation sample true B, decoded h_z/beta from posterior-mean z and from a posterior-sampled z (seed 7), used channel, posterior sigma;
 E2: continued independent solution (E2_ramp_s0, beta=1): decoded h_z and data S_z for 3 validation samples, plus phase lock R(k)."""
from common import *
import torch
from s2spin.vae import SpinVAE
from s2spin import probes
R = RES/'stage2b'
# ---- E1 alignment ----
z = np.load(RES/'stage2'/'E1.npz'); S = torch.tensor(z['spins']); cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]; n = torch.tensor(z['field_dir'][va], dtype=torch.float32)
rows = []
for mode in ('indep', 'pl', 'sm_alt'):
    for s in (0, 1, 2):
        r = jload(R/f'E1_{mode}_s{s}'/'result.json'); m = SpinVAE(32, mode != 'indep', K1=2, K0=2, C=8, spur=None, dmi=None, global_pool=True, n_down=0)
        m.load_state_dict(torch.load(R/f'E1_{mode}_s{s}'/'model.pt')); m.eval()
        with torch.no_grad(): mu1, mu0, lv0, lv1 = m.encode(S[va])
        kl = np.array(r['kl_per_dim_val']); used = int(kl.argmax()) - m.enc.K0                    # vector channel with the largest KL
        v = torch.nn.functional.normalize(mu1[:, used, :, 0, 0], dim=-1); cs = (v*n).sum(-1).abs().numpy()
        for i, cval in enumerate(cs): rows.append(dict(mode=mode, seed=s, sample=i, abs_cos=float(cval), KL_used=float(kl.max())))
save_csv('fig4a_e1_align.csv', pd.DataFrame(rows))
# ---- E3 per-sample ----
z = np.load(RES/'stage2'/'E3.npz'); S = torch.tensor(z['spins']); cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]; B = z['B'][va]; beta = float(z['beta'])
rows = []
for name in ('E3_conv_pl_s0', 'E3_conv_pl_s1', 'E3_conv_sm_alt_s0', 'E3_conv_sm_alt_s1'):
    r = jload(R/name/'result.json'); m = SpinVAE(48, True, spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=True, n_down=2)
    m.load_state_dict(torch.load(R/name/'model.pt')); m.eval(); kl = np.array(r['kl_per_dim_val']); used = int(kl.argmax())
    torch.manual_seed(7)
    with torch.no_grad():
        mu1, mu0, lv0, lv1 = m.encode(S[va]); hm = m.decode(mu1, mu0)[..., 2].mean((1, 2)).numpy()/beta
        z1, z0 = m.sample(mu1, mu0, lv0, lv1, True); hs = m.decode(z1, z0)[..., 2].mean((1, 2)).numpy()/beta
        sig = (lv0[:, used, 0, 0] if used < m.enc.K0 else lv1[:, used - m.enc.K0, 0, 0]).mul(0.5).exp().numpy()
    for i in range(len(va)): rows.append(dict(run=name, B=float(B[i]), hz_mean_z=float(hm[i]), hz_sampled_z=float(hs[i]), sigma_used=float(sig[i]), used_channel=used))
save_csv('fig4c_e3_scatter.csv', pd.DataFrame(rows))
# ---- E2 continued solution: decoded field vs data ----
z = np.load(RES/'stage2'/'E2.npz'); S = torch.tensor(z['spins']); va = np.where(z['chain_id'] % 5 == 0)[0]
m = SpinVAE(48, False, spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=False, n_down=3); m.load_state_dict(torch.load(R/'E2_ramp_s0'/'model.pt')); m.eval()
with torch.no_grad(): mu1, mu0, _, _ = m.encode(S[va]); h = m.decode(mu1, mu0)[..., 2].numpy()
Sz = S[va][..., 2].numpy(); ks = probes.texture_peaks(Sz.astype(float), 3); pl = probes.phase_lock(Sz.astype(float), h.astype(float), ks)
for j, i in enumerate((0, 8, 16)):
    x, y = np.meshgrid(np.arange(48), np.arange(48), indexing='ij')
    save_csv(f'figS5_e2_sample{j}.csv', pd.DataFrame(dict(x=x.ravel(), y=y.ravel(), Sz=Sz[i].ravel(), hz=h[i].ravel())))
save_csv('figS5_phaselock.csv', pd.DataFrame([dict(kx=p['k'][0], ky=p['k'][1], R=p['R'], amp_ratio=p['amp_ratio']) for p in pl]))
print('done', [round(p['R'], 3) for p in pl])
