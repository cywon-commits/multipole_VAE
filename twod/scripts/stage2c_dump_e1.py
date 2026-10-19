"""Stage 2c-A: dump the E1 coupled-PL VAE's decoded field directions in the frame where the true field is along z,
in the format of stage2c_e1_bias.py --vae. usage: python stage2c_dump_e1.py <result dir name under results/stage2b> <out.npz>"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE

name, out = sys.argv[1], sys.argv[2]
z = np.load('results/stage2/E1.npz'); S = torch.tensor(z['spins']); cid = z['chain_id']; L = S.shape[1]
va = np.arange(len(S)) if (len(sys.argv) > 3 and sys.argv[3] == 'all') else np.where(cid % 5 == 0)[0]; n = z['field_dir'][va].astype(float)
m = SpinVAE(L, True, K1=2, K0=2, C=8, spur=None, dmi=None, global_pool=True, n_down=0)
m.load_state_dict(torch.load(f'results/stage2b/{name}/model.pt')); m.eval()
torch.manual_seed(0)
with torch.no_grad():
    mu1, mu0, lv0, lv1 = m.encode(S[va])
    kl = m.kl_per_dim(mu1, mu0, lv0, lv1)
    dirs = {}
    for tag, st in (('sample', True), ('mean', False)):
        z1, z0 = m.sample(mu1, mu0, lv0, lv1, stochastic=st)
        h = m.decode(z1, z0).mean((1, 2)).double().numpy()
        dirs[tag] = h/np.linalg.norm(h, axis=1, keepdims=True)
    hn = m.decode(mu1, mu0).norm(dim=-1).mean().item()
# rotation R' with R' z = n (any z-rotation is a symmetry of the isotropic model with a z field)
def frame(nv):
    a = np.array([1.0, 0, 0]) if abs(nv[2]) > 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(a, nv); e1 /= np.linalg.norm(e1); e2 = np.cross(nv, e1)
    return np.stack([e1, e2, nv], 1)
Rs = np.stack([frame(v) for v in n])                      # columns: e1, e2, n  => R z = n
back = lambda X: np.einsum('nba,n...b->n...a', Rs, X)      # R^T X
spins0 = back(S[va].numpy().astype(float))
np.savez_compressed(out, spins=spins0.astype(np.float32), dir_sample=back(dirs['sample']), dir_mean=back(dirs['mean']),
                    J_vae=m.J0.item(), kappa_vae=hn, KL=float(kl.sum()))
print(name, 'J', m.J0.item(), '|h|', hn, 'KL', float(kl.sum()), 'n_val', len(va))
