"""Stage 2c-B: re-judgement of the four coupled E3 runs (no retraining).
 1. residual-symmetry subspace: z-fraction of the used l=1 channel's posterior-mean variance and the wasted
    transverse KL; 2. information identity I = -1/2 ln(1-R2_sample) vs KL_used; 3. decoder Fisher information I_dec
    (curvature of the reconstruction loss under a shift of the decoded field by beta*dB along z) vs I_PL = 1/sigma_B^2,
    and the rate-distortion optimum predicted from it."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import torch_losses as TL      # noqa

out = 'results'; z = np.load(f'{out}/stage2/E3.npz'); S = torch.tensor(z['spins']); L = S.shape[1]
cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]; B = z['B'].astype(float); beta = float(z['beta'])
varB = (0.6 - 0.2)**2/12; I_PL = 1/float(np.median(z['sigma_B']))**2
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
res = dict(I_PL=I_PL, varB=varB); table = []
for name in ['E3_pl_s0', 'E3_pl_s1', 'E3_sm_alt_s0', 'E3_sm_alt_s1']:
    r = json.load(open(f'{out}/stage2b/{name}/result.json'))
    m = SpinVAE(L, True, spur=spur, dmi='interfacial', global_pool=True, n_down=2)
    m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval()
    kl = np.array(r['kl_per_dim_val']); used = int(kl.argmax()); is_vec = used >= m.enc.K0
    with torch.no_grad():
        mu1, mu0, lv0, lv1 = m.encode(S[va])
        d = dict(name=name, used_channel=used, l=1 if is_vec else 0, KL_used=float(kl[used]), KL_total=float(kl.sum()))
        if is_vec:
            k = used - m.enc.K0; mu = mu1[:, k, :, 0, 0]; s2 = lv1[:, k, 0, 0].exp()
            var = mu.var(0)                                   # variance over samples of each component of the posterior mean
            d['z_fraction_of_mean_variance'] = float(var[2]/var.sum()); d['mean_var_xyz'] = var.tolist()
            waste = 0.5*((mu[:, :2]**2).sum(1) + 2*(s2 - 1 - s2.log()))
            d['wasted_transverse_KL'] = float(waste.mean()); d['posterior_var_sigma2_mean'] = float(s2.mean())
        R2 = r['B_probe_R2_heldout_sampled_z']['per_scalar_channel'][used] if not is_vec else r['B_probe_R2_heldout_sampled_z']['all']
        d['R2_sample'] = R2; d['I_identity'] = float(-0.5*np.log(1 - R2))
        # decoder Fisher information: curvature of the mean reconstruction loss under h -> h + beta dB z_hat (posterior-mean decode)
        h = m.decode(mu1, mu0); Jx, Jy = m.bonds(S.dtype)
        def loss(dB):
            hh = h + torch.tensor([0, 0, beta*dB]); return TL.coupled_pl_loss(S[va], hh, Jx, Jy).mean().item()
        dB = 0.005; I_dec = (loss(dB) + loss(-dB) - 2*loss(0.0))/dB**2
        d['I_dec'] = I_dec; d['I_dec_over_I_PL'] = I_dec/I_PL
        # rate-distortion optimum from the measured I_dec: l=0: t(1+t)=1/(I VarB); l=1: t^2 + t/3 = 1/(I VarB)
        q = 1/(I_dec*varB)
        t = (-1 + np.sqrt(1 + 4*q))/2 if not is_vec else (-1/3 + np.sqrt(1/9 + 4*q))/2
        d['t_opt'] = float(t); d['R2_pred_from_Idec'] = float(1/(1 + t))
        d['KL_pred_from_Idec'] = float(0.5*np.log(1 + 1/t) if not is_vec else 1.5*np.log(1 + 1/(3*t)))
        for tag, f in (('I_dec=I_PL', 1.0), ('I_dec=I_PL/2', 0.5)):
            q2 = 1/(f*I_PL*varB); t2 = (-1 + np.sqrt(1 + 4*q2))/2 if not is_vec else (-1/3 + np.sqrt(1/9 + 4*q2))/2
            d[f'R2_pred[{tag}]'] = float(1/(1 + t2)); d[f'KL_pred[{tag}]'] = float(0.5*np.log(1 + 1/t2) if not is_vec else 1.5*np.log(1 + 1/(3*t2)))
    table.append(d)
res['runs'] = table
json.dump(res, open(f'{out}/stage2b/E3_stage2c_analysis.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
