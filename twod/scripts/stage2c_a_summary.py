"""Stage 2c-A summary: beta scaling (L=32) and size scaling (L=64) of the E1 coupled-PL bias vs the field-noise theory."""
import glob, json, numpy as np
R = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(f'results/stage2b/{pat}/result.json'))]
th = lambda L, J, k: (lambda q: (lambda gam: (lambda Sq: (lambda S2: dict(Sigma2=S2, cJ=4*(1 - 4*J/(k + 4*J))/((J/(k + 4*J))*S2) - (k + 4*J)*(1 - 4*J/(k + 4*J))**2,
        cK=-(k + 4*J)*(1 - 4*J/(k + 4*J))**2 - 16/S2, P0=k + 4*J))((gam**2*Sq).mean()))(1/(k + J*(4 - gam))))(2*np.cos(q)[:, None] + 2*np.cos(q)[None, :]))(2*np.pi*np.arange(L)/L)
out = {}
for L, J, k in ((32, 5., 10.), (64, 5., 10.)):
    co = th(L, J, k); N = L*L; keff = k*k/co['P0']
    out[f'theory_L{L}'] = {b: dict(KL=1.5*np.log(2*N*keff/(9*b)), dJ_c1=100*co['cJ']*(3*b/(2*N*keff) - 1/(N*k)), dK_c1=100*co['cK']*(3*b/(2*N*keff) - 1/(N*k))) for b in (0.25, 0.5, 1.0)}
rows = []
for beta, pat in ((0.25, 'E1beta0.25_pl_s*'), (0.5, 'E1beta0.5_pl_s*'), (1.0, 'E1_pl_s[01]')):
    for r in R(pat): rows.append((beta, 100*(r['J0']/5 - 1), 100*(r['h_norm_mean']/10 - 1), r['kl_total'], r['epochs_run']))
rows = np.array(rows); b, dj = rows[:, 0], rows[:, 1]
slope, icpt = np.polyfit(b, dj, 1)
co = th(32, 5., 10.); N = 1024; keff = 100/co['P0']
c_fit = -icpt/(100*co['cJ']/(N*10))             # intercept = -cJ * c/(N kappa0)
out.update(beta_rows=rows.tolist(), fit_dJ_vs_beta=dict(slope=slope, intercept=icpt, c_from_intercept=c_fit, slope_pred=100*co['cJ']*3/(2*N*keff)))
out['L64'] = [dict(name=r['name'], dJ=100*(r['J0']/5 - 1), dK=100*(r['h_norm_mean']/10 - 1), KL=r['kl_total'], ep=r['epochs_run']) for r in R('E1L64_pl_s*')]
json.dump(out, open('results/stage2b/stage2c_A_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
