"""Stage 2e-C: effective Fisher information of the E3 operating point,
 I_eff/I_PL = { E[rec(sampled z)] - E[rec(true B)] } / { 1/2 E[ I_PL_i (B_hat_i - B_i)^2 ] },  B_hat = decoded h_z/beta."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import torch_losses as TL      # noqa
out = 'results'; z = np.load(f'{out}/stage2/E3.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; cid = z['chain_id']; va = np.where(cid % 5 == 0)[0]
B = torch.tensor(z['B'][va], dtype=torch.float32); beta = float(z['beta']); IPL = torch.tensor(1/z['sigma_B'][va]**2, dtype=torch.float32)
res = {}
for name in ['E3_conv_pl_s0', 'E3_conv_pl_s1', 'E3_conv_sm_alt_s0']:
    m = SpinVAE(L, True, spur=[[0, 0, 1.], [1., 0, 0], [0, 1., 0]], dmi='interfacial', global_pool=True, n_down=2)
    m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval(); Jx, Jy = m.bonds(torch.float32)
    num, den = [], []
    with torch.no_grad():
        mu1, mu0, lv0, lv1 = m.encode(S[va])
        rec_true = TL.coupled_pl_loss(S[va], torch.stack([torch.zeros_like(B), torch.zeros_like(B), beta*B], -1)[:, None, None, :].expand(-1, L, L, -1), Jx, Jy)
        for rep in range(5):
            torch.manual_seed(100 + rep); z1, z0 = m.sample(mu1, mu0, lv0, lv1, True); h = m.decode(z1, z0)
            rec = TL.coupled_pl_loss(S[va], h, Jx, Jy); Bh = h[..., 2].mean((1, 2))/beta
            num.append((rec - rec_true).mean().item()); den.append((0.5*IPL*(Bh - B)**2).mean().item())
        zm = m.decode(mu1, mu0); rec_mean = TL.coupled_pl_loss(S[va], zm, Jx, Jy); Bhm = zm[..., 2].mean((1, 2))/beta
    res[name] = dict(num=float(np.mean(num)), den=float(np.mean(den)), I_eff_over_I_PL=float(np.mean(num)/np.mean(den)), num_reps=num, den_reps=den,
                     mean_z=dict(num=float((rec_mean - rec_true).mean()), den=float((0.5*IPL*(Bhm - B)**2).mean()), ratio=float((rec_mean - rec_true).mean()/(0.5*IPL*(Bhm - B)**2).mean())),
                     rec_true=float(rec_true.mean()))
    print(name, json.dumps({k: v for k, v in res[name].items() if 'reps' not in k}, default=float), flush=True)
json.dump(res, open(f'{out}/stage2e/C_ieff.json', 'w'), indent=1, default=float)
