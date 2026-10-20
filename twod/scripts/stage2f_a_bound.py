"""Stage 2f-A/figures: tangent-construction lower bound beta_lb = max over learned states (D_c - D)/R, states = continued E2 independent
solutions (beta 0.01..31.7, seeds 0, 1). D = expected reconstruction with posterior-sampled z (8 draws) and with the posterior mean;
D_c = reconstruction of the collapsed solution on the same validation samples. Figures: results/fig/stage2f_a.png (KL(beta)), stage2f_b.png (R, D plane)."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2spin.vae import SpinVAE             # noqa
from s2spin import torch_losses as TL      # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

out = 'results'; z = np.load(f'{out}/stage2/E2.npz'); S = torch.tensor(z['spins']); L = S.shape[1]; va = np.where(z['chain_id'] % 5 == 0)[0]
spur = [[0, 0, 1.], [1., 0, 0], [0, 1., 0]]
def state(name):
    r = json.load(open(f'{out}/stage2b/{name}/result.json')); m = SpinVAE(L, False, spur=spur, dmi='interfacial', global_pool=False, n_down=3)
    m.load_state_dict(torch.load(f'{out}/stage2b/{name}/model.pt')); m.eval()
    with torch.no_grad():
        mu1, mu0, lv0, lv1 = m.encode(S[va]); R = float(m.kl_per_dim(mu1, mu0, lv0, lv1).sum())
        Dm = TL.vmf_nll(S[va], m.decode(mu1, mu0)).sum((1, 2)).mean().item(); Ds = []
        for rep in range(8):
            torch.manual_seed(500 + rep); z1, z0 = m.sample(mu1, mu0, lv0, lv1, True); Ds.append(TL.vmf_nll(S[va], m.decode(z1, z0)).sum((1, 2)).mean().item())
    return dict(name=name, beta=r['args']['beta'] if not r['args'].get('beta_ramp') else r['hist'][-1]['beta'], R=R, D_mean=Dm, D_samp=float(np.mean(Ds)))
src = lambda b, s: {0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}', 10: f'E2_A2_b10_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}')
res = dict(states={}, Dc={})
for s in (0, 1):
    res['Dc'][s] = state(f'E2_B2s_b1_s{s}')
    rows = [state(src(b, s)) for b in (0.01, 0.03, 0.1, 0.3, 1, 3, 10)]
    rows += [state(os.path.basename(f.rsplit('/', 1)[0])) for pat in ('E2_F_b*', 'E2_FC_b*', 'E2_FB2_b*') for f in sorted(glob.glob(f'{out}/stage2b/{pat}_s{s}/result.json'), key=lambda x: float(x.split('_b')[1].split('_')[0]))
             if json.load(open(f))['kl_total'] > 0.5]
    res['states'][s] = rows
for s in (0, 1):
    Dc_s, Dc_m = res['Dc'][s]['D_samp'], res['Dc'][s]['D_mean']
    print(f'--- seed {s}: D_c (collapsed, same val samples) = {Dc_m:.1f} (mean z), {Dc_s:.1f} (sampled z)')
    best = {}
    for r in res['states'][s]:
        r['ratio_mean'] = (Dc_m - r['D_mean'])/r['R']; r['ratio_samp'] = (Dc_s - r['D_samp'])/r['R']
        print('  beta %-7.4g R %7.2f  D_mean %8.1f D_samp %8.1f  (Dc-D)/R: mean-z %6.2f  sampled-z %6.2f   [%s]' % (r['beta'], r['R'], r['D_mean'], r['D_samp'], r['ratio_mean'], r['ratio_samp'], r['name']))
    res[f'beta_lb_mean_z_s{s}'] = max(r['ratio_mean'] for r in res['states'][s]); res[f'beta_lb_sampled_z_s{s}'] = max(r['ratio_samp'] for r in res['states'][s])
    print(f'  beta_lb: mean-z {res[f"beta_lb_mean_z_s{s}"]:.1f}, sampled-z {res[f"beta_lb_sampled_z_s{s}"]:.1f}')
json.dump(res, open(f'{out}/stage2f_A_bound.json', 'w'), indent=1, default=float)
# ---- figures
os.makedirs(f'{out}/fig', exist_ok=True)
R_ = lambda n: json.load(open(f'{out}/stage2b/{n}/result.json'))
fig, ax = plt.subplots(figsize=(5.6, 4))
for s, c in ((0, 'C0'), (1, 'C1')):
    sc = [(b, R_(f'E2_B2s_b{b}_s{s}')['kl_total']) for b in (0.01, 0.03, 0.1, 0.3, 1, 3)]
    ct = [(b, R_(src(b, s))['kl_total']) for b in (0.01, 0.03, 0.1, 0.3, 1, 3, 10)] + [(b, R_(f'E2_A2_b{b}_s{s}')['kl_total']) for b in (31.6, 100, 316, 1000)]
    fn = [(float(f.split('_b')[1].split('_')[0]), json.load(open(f))['kl_total']) for f in glob.glob(f'{out}/stage2b/E2_F_b*_s{s}/result.json')]
    ax.semilogx(*zip(*sc), 'o', c='C3', ms=4, label='from scratch' if s == 0 else None); ax.semilogx(*zip(*ct), 's-', c=c, ms=4, lw=0.8, label=f'continued s{s}')
    if fn: ax.semilogx(*zip(*sorted(fn)), '^', c='k', ms=4, label='fine grid' if s == 0 else None)
    ax.axvline(res[f'beta_lb_sampled_z_s{s}'], c=c, ls=':', lw=1)
ax.set_xlabel('beta'); ax.set_ylabel('KL [nat]'); ax.set_yscale('symlog', linthresh=1); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{out}/fig/stage2f_a.png', dpi=130)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, s in enumerate((0, 1)):
    st = res['states'][s]; Dc = res['Dc'][s]['D_samp']
    ax[k].plot([r['R'] for r in st], [r['D_samp'] for r in st], 'o-', ms=4); ax[k].plot([0], [Dc], 'r*', ms=10, label='collapsed (0, D_c)')
    for r in st: ax[k].annotate(f"{r['beta']:.3g}", (r['R'], r['D_samp']), fontsize=6)
    bst = max(st, key=lambda r: r['ratio_samp']); xs = np.array([0, 110.0])
    ax[k].plot(xs, Dc - bst['ratio_samp']*xs, 'k--', lw=0.8, label=f"tangent line, slope -beta_lb={bst['ratio_samp']:.1f}")
    ax[k].set_xlim(0, 110); ax[k].set_ylim(3000, 5300); ax[k].set_xlabel('R = KL [nat] (zoom to R < 110)'); ax[k].set_ylabel('D (sampled-z reconstruction)'); ax[k].set_title(f'seed {s}'); ax[k].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{out}/fig/stage2f_b.png', dpi=130)
