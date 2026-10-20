"""Fig. 2 (two-column): estimator validation. (a) 1D exact chain correlations, (b) 2D Gaussian regime C(r) and <|d|^2>(kappa) inset,
(c) kappa_ind/kappa vs 4J/kappa, (d) DMI recovery (Stage 1b) with Moriya-angle inset.
Raw data: data/1d_*.csv (rerun_1d.py), results/stage1a/summary.json, results/stage1b/summary.json."""
from common import *
sys.path.insert(0, str(TWOD/'scripts'))
from s2spin import analytic
# ---- CSVs ----
c1 = pd.read_csv(DATA/'1d_exact_corr.csv'); k1 = pd.read_csv(DATA/'1d_kappa_ind.csv')
a = jload(RES/'stage1a'/'summary.json'); b = jload(RES/'stage1b'/'summary.json')
rows = []
for key, v in a.items():
    if not key.startswith('J='): continue
    J = float(key[2:]); C, Cp = v['C'], v['Cpred']
    for r in range(len(C)): rows.append(dict(J=J, r=r, C_mc=C[r], C_gauss=Cp[r]))
c2 = save_csv('fig2b_corr2d.csv', pd.DataFrame(rows)); c2 = pd.read_csv(c2)
mw = pd.DataFrame([dict(kappa=float(k), delta2_mc=v[0], two_G0=v[1], disk=analytic.delta2_disk(float(k), 5.0)) for k, v in a['mermin_wagner'].items()]); save_csv('fig2b_mw.csv', mw)
kk = []
for key, v in a.items():
    if key.startswith('J='):
        J = float(key[2:]); d2 = v['C'][0]; G0 = 1/v['kappa_ind_pred']
        kk.append(dict(J=J, kappa=10.0, x=4*J/10.0, ratio_mc=v['kappa_ind']/10.0, ratio_leading=v['kappa_ind_pred']/10.0, ratio_nlo=(1/G0)*(1 - G0)*2*G0/d2/10.0))
k2 = pd.DataFrame(kk); save_csv('fig2c_kappa_ind2d.csv', k2)
L = 32; xs = np.linspace(0.2, 9, 60); lead2 = [analytic.kappa_ind_pred(L, 10.0, 10.0*x/4)/10.0 for x in xs]; lead1 = np.sqrt(1 + xs)
save_csv('fig2c_curves.csv', pd.DataFrame(dict(x=xs, lead_2d_L32=lead2, lead_1d=lead1)))
cases = list(b.keys())
from s2spin import NAMES
# SM standard errors are stored for the 21 fitted parameters (first 21 entries of NAMES order)
drows = []
for key in cases:
    kind = key.split()[0]; Dv = float(key.split('D=')[1].split()[0]); beta = float(key.split('beta=')[1]); v = b[key]
    for q in ['J0_x', 'J0_y', 'Dy_x', 'Dx_y', 'Dx_x', 'Dy_y', 'hz']:
        i = NAMES.index(q); tr = v['true'][i]
        if abs(tr) > 0 and i < len(v['se']): drows.append(dict(kind=kind, D=Dv, beta=beta, param=q, true=tr, SM=v['theta'][i], SM_se=v['se'][i], PL=v['pl'][i]))
    if v['moriya_deg']: pass
dd = pd.DataFrame(drows); save_csv('fig2d_dmi.csv', dd)
mo = pd.DataFrame([dict(kind=k.split()[0], D=float(k.split('D=')[1].split()[0]), beta=float(k.split('beta=')[1]), ang_x=v['moriya_deg']['x'], ang_y=v['moriya_deg']['y']) for k, v in b.items() if v['moriya_deg']]); save_csv('fig2d_moriya.csv', mo)
# ---- figure ----
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.4)); (A, Bx), (Cx, Dx) = axs
cols = {0.5: EXTRA[1], 2.0: PL['color'], 5.0: EXTRA[0]}
for J, g in c1.groupby('J'):
    A.semilogy(g.r, g.corr_exact, c=cols[J], lw=0.9, label=f'$J={J:g}$ exact $\\mathcal{{L}}(J)^r$'); gg = g[g.corr_exact > 2e-3]; A.semilogy(gg.r, gg.corr_mc, 'o', c=cols[J], ms=3.5, mfc='none')
A.set_xlabel('$r$'); A.set_ylabel(r'$\langle \mathbf{S}_0\cdot\mathbf{S}_r\rangle$'); A.set_ylim(1e-5, 1.5); A.legend(fontsize=6, loc='lower left'); plabel(A, '(a)')
for J, g in c2.groupby('J'):
    Bx.semilogy(g.r, g.C_gauss, c=cols.get(J, EXTRA[2]) if J != 20 else EXTRA[2], lw=0.9, label=f'$J={J:g}$: $2G(r)$'); gg = g[g.r <= 4]; Bx.semilogy(gg.r, gg.C_mc, 'o', c=cols.get(J, EXTRA[2]) if J != 20 else EXTRA[2], ms=3.5, mfc='none')
Bx.set_xlabel('$r$'); Bx.set_ylabel(r'$C(r)$ (transverse)'); Bx.legend(fontsize=6, loc='upper right'); plabel(Bx, '(b)')
ins = Bx.inset_axes([0.2, 0.1, 0.32, 0.32]); ins.loglog(mw.kappa, mw.delta2_mc, 'o', c='k', ms=2.5); ins.loglog(mw.kappa, mw.two_G0, **PRED); ins.set_xlabel('$\\kappa$', fontsize=6, labelpad=0); ins.set_ylabel('$\\langle|\\delta|^2\\rangle$', fontsize=6, labelpad=-2); ins.minorticks_off(); ins.tick_params(labelsize=5)
Cx.plot(xs, lead1, **TRUTH | dict(ls='-', color='#bbbbbb'), label='1D: $\\sqrt{1+4J/\\kappa}$'); Cx.plot(xs, lead2, **PRED, label='2D ($L{=}32$): $1/(\\kappa G(0))$')
Cx.plot(k1.x_4J_over_kappa, k1.kappa_ind_mc/k1.kappa, 'o', c='k', mfc='none', label='1D MC'); Cx.plot(k2.x, k2.ratio_mc, 's', c='k', label='2D MC')
Cx.plot(k2.x, k2.ratio_nlo, 'x', c=EXTRA[0], ms=5, label='2D, next order'); Cx.plot(k1.x_4J_over_kappa, k1.kappa_ind_pred/k1.kappa*0 + (k1.kappa_ind_pred_nlo/k1.kappa), '+', c=EXTRA[0], ms=5)
Cx.set_xlabel('$4J/\\kappa$'); Cx.set_ylabel('$\\kappa_{\\rm ind}/\\kappa$'); Cx.legend(fontsize=6, loc='upper left'); plabel(Cx, '(c)')
labs = ['J0_x', 'J0_y', 'Dy_x', 'Dx_y', 'Dx_x', 'Dy_y', 'hz']; grp = {'J0_x': 0, 'J0_y': 0, 'Dy_x': 1, 'Dx_y': 1, 'Dx_x': 1, 'Dy_y': 1, 'hz': 2}
bmk = {0.5: 'o', 1.0: 's', 2.0: '^'}
for _, r in dd.iterrows():
    x = grp[r.param] + (0.12*list(bmk).index(r.beta) - 0.12) + (0.0 if r.kind == 'interfacial' else 0.0)
    Dx.errorbar(x + (0.30 if r.kind == 'bulk' else 0.0), r.SM/r.true, yerr=r.SM_se/abs(r.true), fmt=bmk[r.beta], color=SM['color'] if r.kind == 'interfacial' else EXTRA[1], mfc='none' if r.kind == 'bulk' else None, ms=3, lw=0.6, capsize=1)
Dx.axhline(1, **TRUTH); Dx.set_xticks([0.15, 1.15, 2.1]); Dx.set_xticklabels(['$J_\\mu$', 'Moriya $D$', '$h_z$']); Dx.set_ylabel('estimate / true'); Dx.set_ylim(0.85, 1.15); plabel(Dx, '(d)')
for k_, c_ in (('interfacial', SM['color']), ('bulk', EXTRA[1])): Dx.plot([], [], 'o', c=c_, mfc='none' if k_ == 'bulk' else c_, label=k_)
for bt, m_ in bmk.items(): Dx.plot([], [], m_, c='gray', label=f'$\\beta={bt:g}$')
Dx.legend(fontsize=6, ncol=2, loc='upper right')
ins = Dx.inset_axes([0.2, 0.08, 0.3, 0.26]); ins.plot(mo.beta, mo[['ang_x', 'ang_y']].max(axis=1), 'k.', ms=3); ins.set_ylabel('Moriya angle\nerror [°]', fontsize=5.5, labelpad=0); ins.set_xlabel('$\\beta$', fontsize=6, labelpad=0); ins.tick_params(labelsize=5)
fig.tight_layout(); savefig(fig, 'fig2')
rel = (c1.corr_mc - c1.corr_exact).abs(); e1 = pd.read_csv(DATA/'1d_exact_est.csv'); e1 = e1[e1.fit == 'J0_only']
caption('fig2', f"""**Figure 2. Estimator validation.** (a) 1D open chain ($N=100$, 4000 samples, one seed; MC points where the exact value is below $2\\times10^{-3}$ are noise-limited and omitted): correlations (open circles) agree with the Fisher exact result $\\mathcal{{L}}(J)^r$ (lines) for $J=0.5,2,5$ (max abs. difference over $r\\le 8$: {rel.max():.4f}); score-matching $J$ estimates {', '.join(f'{r.SM_J0:.4f}$\\pm${r.SM_J0_se:.4f}' for r in e1.itertuples())} (chain-group jackknife; truth {', '.join(f'{r.true_J0:g}' for r in e1.itertuples())}). (b) 2D Gaussian regime ($L=32$, $\\kappa=10$; MC with 100 chains, one seed; points at $r>4$ are noise-limited and omitted; no error bars shown, statistical errors are below the symbol size): transverse correlation $C(r)$ vs lattice Gaussian $2G(r)$ for $J=2,5,20$; ratios MC/Gaussian at $r=1$: {', '.join(f'{r.C_mc/r.C_gauss:.3f}' for r in c2[c2.r==1].itertuples())}. Inset: $\\langle|\\delta|^2\\rangle$ vs $\\kappa$ ($J=5$) against the lattice value $2G(0)$ (MC is always below, the Mermin--Wagner log trend is reproduced). (c) Independent-decoder overestimate $\\kappa_{{\\rm ind}}/\\kappa$ vs $4J/\\kappa$: 1D ($\\sqrt{{1+4J/\\kappa}}$, grey line) and 2D ($1/(\\kappa G(0))$ at $L=32$, black line); markers are MC values; crosses use the next-order correction $(1-G)\\,2G/\\langle|\\delta|^2\\rangle$ (max deviation from MC: 1D {abs(k1.kappa_ind_pred_nlo-k1.kappa_ind_mc).max()/10*100:.2f}\\%$\\kappa$, 2D {abs(k2.ratio_nlo-k2.ratio_mc).max()*100:.2f}\\%$\\kappa$, versus the leading order {abs(k2.ratio_leading-k2.ratio_mc).max()*100:.2f}\\%$\\kappa$ in 2D). (d) 2D DMI recovery (Stage 1b, $L=32$, one seed, score matching): estimate/true for non-zero parameters; error bars: chain-group jackknife standard errors; interfacial: filled, bulk: open; marker shape: $\\beta$. Maximum $|z|$ over all non-zero parameters {((dd.SM-dd.true)/dd.SM_se).abs().max():.2f}. Inset: Moriya-vector angle error (max over $x,y$ bonds; maximum {mo[['ang_x','ang_y']].max().max():.2f}$^\\circ$).""")
print('fig2 done')
