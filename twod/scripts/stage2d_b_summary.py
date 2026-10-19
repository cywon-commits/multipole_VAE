"""Stage 2d-B1/B2 summary: hysteresis ramp curves and KL(beta) for from-scratch vs continued training."""
import glob, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = lambda n: json.load(open(f'results/stage2b/{n}/result.json'))
ex = lambda n: len(glob.glob(f'results/stage2b/{n}/result.json')) > 0
COLL = 5155.2
out = dict(B1={}, B2={})
for tag, h0, rp in (('indep', 'E2_h0.01_s{}', 'E2_ramp_s{}'), ('K+M1', 'E2_K1_h0.01_s{}', 'E2_K1_ramp_s{}'), ('K+M1-hm', 'E2_hmK_h0.01_s{}', 'E2_hmK_ramp_s{}')):
    for s in (0, 1):
        if not (ex(h0.format(s)) and ex(rp.format(s))): continue
        a, b = R(h0.format(s)), R(rp.format(s)); hb = b['hist']
        out['B1'][f'{tag}_s{s}'] = dict(start_KL=a['kl_total'], start_obj=a['hist'][-1]['obj_val'], start_epochs=a['epochs_run'], start_converged=a['converged_at'],
            end_KL=b['kl_total'], end_obj=hb[-1]['obj_val'], end_rec=hb[-1]['val_rec'], end_epochs=b['epochs_run'], end_converged=b['converged_at'])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for s, c in ((0, 'C0'), (1, 'C1')):
    if ex(f'E2_ramp_s{s}'):
        hb = R(f'E2_ramp_s{s}')['hist']; ax[0].semilogx([h['beta'] for h in hb], [h['kl_val'] for h in hb], c, label=f'indep ramp s{s}')
    if ex(f'E2_K1_ramp_s{s}'):
        hb = R(f'E2_K1_ramp_s{s}')['hist']; ax[0].semilogx([h['beta'] for h in hb], [h['kl_val'] for h in hb], c + '--', label=f'K+M1 ramp s{s}')
ax[0].set_xlabel('beta (ramp, then hold at 1)'); ax[0].set_ylabel('KL [nat]'); ax[0].legend(fontsize=6); ax[0].set_title('B1 hysteresis')
bs = [0.01, 0.03, 0.1, 0.3, 1, 3]; sc, ct = {}, {}
for b in bs:
    vals = [R(f'E2_B2s_b{b}_s{s}')['kl_total'] for s in (0, 1) if ex(f'E2_B2s_b{b}_s{s}')]; sc[b] = vals
    cn = []
    for s in (0, 1):
        n = {0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}')
        if ex(n): cn.append(R(n)['kl_total'])
    ct[b] = cn
    ax[1].semilogx([b]*len(sc[b]), sc[b], 'o', c='C3', label='from scratch (20-epoch warm-up)' if b == bs[0] else None)
    ax[1].semilogx([b]*len(cn), cn, 's', c='C2', label='continued from beta=0.01 path' if b == bs[0] else None)
ax[1].set_xlabel('beta'); ax[1].set_ylabel('converged KL [nat]'); ax[1].legend(fontsize=6); ax[1].set_title('B2 KL(beta)')
mc = np.array([[b, np.mean(ct[b])] for b in bs if ct[b]]); 
if len(mc) > 2:
    sl = np.polyfit(np.log(mc[mc[:, 0] >= 0.1, 0]), mc[mc[:, 0] >= 0.1, 1], 1)[0]; out['B2_continued_slope_dKL_dlnbeta_0.1_to_3'] = sl
    out['B2_active_modes_from_slope'] = -2*sl
out['B2'] = dict(scratch={str(b): sc[b] for b in bs}, continued={str(b): ct[b] for b in bs})
for b in bs:
    for s in (0, 1):
        n = {0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}')
        m = f'E2_B2s_b{b}_s{s}'
        if ex(n) and ex(m): out['B2'][f'obj_{b}_s{s}'] = dict(continued=R(n)['hist'][-1]['obj_val'], scratch=R(m)['hist'][-1]['obj_val'])
ax[2].semilogx(bs, [np.mean([R(f'E2_B2s_b{b}_s{s}')['hist'][-1]['obj_val'] for s in (0, 1) if ex(f'E2_B2s_b{b}_s{s}')]) for b in bs], 'o-', c='C3', label='scratch')
ax[2].semilogx([b for b in bs if ct[b]], [np.mean([R({0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}'))['hist'][-1]['obj_val'] for s in (0, 1) if ex({0.01: f'E2_h0.01_s{s}', 1: f'E2_ramp_s{s}'}.get(b, f'E2_B2c_b{b}_s{s}'))]) for b in bs if ct[b]], 's-', c='C2', label='continued')
ax[2].axhline(COLL, c='k', ls=':'); ax[2].set_xlabel('beta'); ax[2].set_ylabel('val objective (rec + beta KL)'); ax[2].legend(fontsize=6); ax[2].set_title('objective vs collapsed 5155.2')
fig.tight_layout(); fig.savefig('results/fig/stage2d_B.png', dpi=130)
json.dump(out, open('results/stage2b/stage2d_B_summary.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
