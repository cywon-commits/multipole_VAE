"""Fig. 6: collapse. (a) KL(beta) log-log, (b) beta 8-40 linear with a linear fit of the seed-1 fine grid, (c) tangent construction (sampled-z D), (d) periodic-phase toy.
Data: data/fig6_*.csv built from results/stage2b/E2_* and results/stage2f_A_bound.json (stage2f_a_bound.py); toy from scripts/stage2f_phase_toy.py logic."""
from common import *
from scipy.optimize import minimize_scalar
R = RES/'stage2b'; ex = lambda n: (R/n/'result.json').exists(); kl = lambda n: jload(R/n/'result.json')['kl_total']
rows = []
for s in (0, 1):
    for b in (0.01, 0.03, 0.1, 0.3, 1, 3): rows.append(dict(series='scratch', seed=s, beta=b, KL=kl(f'E2_B2s_b{b}_s{s}')))
    for b, n in ((0.01, f'E2_h0.01_s{s}'), (0.03, f'E2_B2c_b0.03_s{s}'), (0.1, f'E2_B2c_b0.1_s{s}'), (0.3, f'E2_B2c_b0.3_s{s}'), (1, f'E2_ramp_s{s}'), (3, f'E2_B2c_b3_s{s}'), (10, f'E2_A2_b10_s{s}')):
        rows.append(dict(series='continued', seed=s, beta=b, KL=kl(n)))
    for b in (31.6, 100, 316, 1000, 3162, 10000): rows.append(dict(series='continued (x3.16 steps)', seed=s, beta=b, KL=kl(f'E2_A2_b{b}_s{s}')))
    for b in (12.6, 15.9, 20.0, 25.2, 31.8): rows.append(dict(series='fine grid', seed=s, beta=10*1.26**[12.6, 15.9, 20.0, 25.2, 31.8].index(b)*0 + {12.6: 12.6, 15.9: 15.876, 20.0: 20.0, 25.2: 25.2, 31.8: 31.75}[b], KL=kl(f'E2_F_b{b}_s{s}')))
    rows.append(dict(series='flat start (lr/3)', seed=s, beta=31.75, KL=kl(f'E2_FC_b31.8_s{s}')))
rows.append(dict(series='B-2 (fixed beta, lr/3)', seed=1, beta=31.75, KL=kl('E2_FB2_b31.8_s1')))
d = pd.DataFrame(rows); save_csv('fig6a_kl.csv', d)
bd = jload(RES/'stage2f_A_bound.json'); st = []
for s in ('0', '1'):
    for r in bd['states'][s]: st.append(dict(seed=int(s), name=r['name'], beta=r['beta'], R=r['R'], D_samp=r['D_samp'], D_mean=r['D_mean'], ratio_samp=r['ratio_samp'], Dc=bd['Dc'][s]['D_samp'],
                                         kind='chain' if (r['name'].startswith('E2_F_b') or r['name'].startswith('E2_A2') or r['name'].startswith('E2_B2c') or r['name'].startswith('E2_h') or r['name'].startswith('E2_ramp')) else ('flat' if 'FC' in r['name'] else 'B2')))
st = pd.DataFrame(st); save_csv('fig6c_states.csv', st); lb = {s: bd[f'beta_lb_sampled_z_s{s}'] for s in (0, 1)}
# linear fit, seed 1 fine grid + start state
f1 = d[(d.seed == 1) & (d.series == 'fine grid')].sort_values('beta'); f1 = f1[f1.KL > 0.5]
slope, icpt = np.polyfit(f1.beta, f1.KL, 1); b0 = -icpt/slope; save_csv('fig6b_fit.csv', pd.DataFrame([dict(seed=1, slope=slope, intercept=icpt, beta_zero=b0, n_points=len(f1))]))
# toy
xm = np.pi**2/3; gc = 1 - np.exp(-xm)
def interior(b):
    r = minimize_scalar(lambda lx: 1 - np.exp(-np.exp(lx)) + b*np.log(xm/np.exp(lx)), bounds=(-12, 0), method='bounded'); return np.exp(r.x), r.fun
bs = np.linspace(0.004, 0.36, 400); toy = []
for b in bs:
    x, f = interior(b); toy.append(dict(beta_over_snr=b/xm, KL=0.5*np.log(xm/x), obj=f))
toy = pd.DataFrame(toy); bin_ = toy[toy.obj > gc].beta_over_snr.min() if (toy.obj > gc).any() else np.nan
toy['stable'] = toy.beta_over_snr <= bin_; toy['KL_gauss'] = np.clip(0.5*np.log(1/toy.beta_over_snr), 0, None); save_csv('fig6d_toy.csv', toy)
fig = plt.figure(figsize=(W2, 6.2)); gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28); A = fig.add_subplot(gs[0, 0]); Bx = fig.add_subplot(gs[0, 1]); C0 = fig.add_subplot(gs[1, 0]); Dx = fig.add_subplot(gs[1, 1])
seedc = {0: '#E69F00', 1: '#A05A00'}; FLOOR = 0.3
def kval(v): return max(v, FLOOR)
for (series, seed), g in d.groupby(['series', 'seed']):
    col = seedc[seed]; mk = 's' if seed == 0 else '^'; y = [kval(v) for v in g.KL]; z = g.KL.values < 0.05
    if series == 'scratch': A.plot(g.beta[~z], g.KL[~z], mk, c=col, mfc='none', ms=4); A.plot(g.beta[z], [FLOOR]*z.sum(), 'v', c=col, mfc='none', ms=4)
    elif series == 'continued': A.plot(g.beta, g.KL, mk + '-', c=col, ms=3.5, lw=0.7)
    elif series == 'continued (x3.16 steps)': A.plot(g.beta[~z], g.KL[~z], mk, c=col, ms=3.5); A.plot(g.beta[z], [FLOOR]*z.sum(), 'v', c=col, ms=4, mfc=col)
    elif series == 'fine grid': A.plot(g.beta[~z], g.KL[~z], mk, c='k', ms=3.5, mfc=col, mew=0.8); A.plot(g.beta[z], [FLOOR]*z.sum(), 'v', c='k', mfc=col, ms=4)
    elif series.startswith('flat'): A.plot(g.beta[~z], g.KL[~z], 'x', c=col, ms=5); A.plot(g.beta[z], [FLOOR]*z.sum(), 'x', c=col, ms=5)
    else: A.plot(g.beta, g.KL, '*', c=col, ms=7)
A.set_xscale('log'); A.set_yscale('log'); A.set_ylim(0.2, 2500); A.axhline(FLOOR, c='#cccccc', lw=0.5); A.text(150, 0.33, 'KL $=0$ (floor)', fontsize=5.5, va='bottom')
for s in (0, 1): A.axvline(lb[s], c=seedc[s], ls=':', lw=1.0)
A.set_xlabel('$\\beta$'); A.set_ylabel('KL [nat]'); plabel(A, '(a)')
hand = [A.plot([], [], 's', c=seedc[0], mfc='none', label='from scratch')[0], A.plot([], [], 's-', c=seedc[0], label='continued, seed 0 ($\\blacktriangle$ seed 1)')[0], A.plot([], [], 's', c='k', mfc=seedc[0], label='fine grid ($\\times1.26$)')[0], A.plot([], [], 'x', c='gray', label='flat start')[0], A.plot([], [], '*', c='gray', label='B-2 (fixed $\\beta$)')[0], A.plot([], [], ':', c='gray', label='$\\beta_{\\rm lb}$ (seed 0, 1)')[0]]
A.legend(handles=hand, fontsize=5.3, loc='center left', bbox_to_anchor=(0.0, 0.40))
sel = d[(d.beta >= 8) & (d.beta <= 40)]
for (series, seed), g in sel.groupby(['series', 'seed']):
    col = seedc[seed]; mk = 's' if seed == 0 else '^'
    if series in ('continued', 'fine grid', 'continued (x3.16 steps)'): Bx.plot(g.beta, g.KL, mk, c=col if series != 'fine grid' else 'k', mfc=col, ms=4, mew=0.8)
    elif series.startswith('flat'): Bx.plot(g.beta, g.KL, 'x', c=col, ms=5)
    elif series.startswith('B-2'): Bx.plot(g.beta, g.KL, '*', c=col, ms=7)
Bx.axhline(0, **TRUTH); b2p = d[d.series.str.startswith('B-2')].iloc[0]
Bx.plot([b2p.beta], [b2p.KL], '*', c=seedc[1], mec='k', mew=0.8, ms=13, zorder=5); Bx.annotate('KL = 3.8 at $\\beta$ = 31.75', (b2p.beta, b2p.KL), xytext=(b2p.beta - 17, 24), fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5))
for s in (0, 1): Bx.axvline(lb[s], c=seedc[s], ls=':', lw=1.0)
Bx.set_xlim(8, 40); Bx.set_ylim(-3, 80); Bx.set_xlabel('$\\beta$'); Bx.set_ylabel('KL [nat]'); plabel(Bx, '(b)')
# (c) tangent: two seeds side by side inside one panel
C0.axis('off'); plabel(C0, '(c)', x=-0.08, y=1.05); sub = gs[1, 0].subgridspec(1, 2, wspace=0.45); cax = [fig.add_subplot(sub[0, k]) for k in (0, 1)]
for k, s in enumerate((0, 1)):
    ax = cax[k]; g = st[st.seed == s]; Dc = g.Dc.iloc[0]; ch = g[g.kind == 'chain'].sort_values('beta'); ch = ch[ch.beta >= 10]
    ax.plot(ch.R, ch.D_samp, 'o-', c=seedc[s], ms=3, lw=0.8, mfc='w' if False else seedc[s]); ax.plot([0], [Dc], '*', c='r', ms=8)
    for r in ch.itertuples(): ax.annotate(f'{r.ratio_samp:.0f}', (r.R, r.D_samp), fontsize=5, xytext=(2, 3), textcoords='offset points')
    fl = g[g.kind == 'flat']; b2 = g[g.kind == 'B2']
    ax.plot(fl.R, fl.D_samp, 'x', c='k', ms=5); ax.plot(b2.R, b2.D_samp, '*', c='k', ms=6)
    top = g.loc[g.ratio_samp.idxmax()]; ax.plot([top.R], [top.D_samp], 'D', mfc='none', mec='r', mew=1.2, ms=8, zorder=6)
    xs = np.array([0, 110.0]); ax.plot(xs, Dc - lb[s]*xs, **APPROX); ax.set_xlim(0, 100); ax.set_ylim(3300, 5300 if s == 0 else 5300); ax.set_xlabel('$R$ = KL [nat]'); ax.set_title(f'seed {s}: $\\beta_{{\\rm lb}}={lb[s]:.1f}$', fontsize=6.5)
    if k == 0: ax.set_ylabel('$D$ (sampled-$z$ reconstruction)')
    ins = ax.inset_axes([0.6, 0.5, 0.36, 0.38]); gg = g.sort_values('beta'); ins.plot(gg.R, gg.D_samp, '.-', c=seedc[s], ms=3, lw=0.6); ins.plot([0], [Dc], '*', c='r', ms=6); ins.tick_params(labelsize=4.5); ins.set_title('full range', fontsize=5, pad=1)
cax[0].plot([], [], 'x', c='k', label='flat start'); cax[0].plot([], [], '*', c='k', label='B-2'); cax[0].plot([], [], '*', c='r', label='collapsed $(0,D_c)$'); cax[0].plot([], [], **APPROX, label='tangent'); cax[0].plot([], [], 'D', mfc='none', mec='r', label='state giving $\\beta_{\\rm lb}$'); cax[1].legend(handles=[h for h in cax[0].lines[-5:]], fontsize=5, loc='lower right')
Dx.semilogx(toy.beta_over_snr[toy.stable], toy.KL[toy.stable], c='k', label='periodic phase mode'); Dx.semilogx(toy.beta_over_snr[~toy.stable], toy.KL[~toy.stable], c='k', ls=':', lw=0.8)
Dx.semilogx(toy.beta_over_snr, toy.KL_gauss, c='#888888', label='Gaussian water-filling'); Dx.axvline(bin_, c='k', lw=0.4, ls='--'); klb = float(toy.KL[toy.stable].iloc[-1]); Dx.plot([bin_, bin_], [klb, 0], c='k', lw=1.0); Dx.plot([bin_, 1.2], [0, 0], c='k', lw=1.0); Dx.set_xlim(4e-3, 1.2); Dx.set_ylim(0, 3.2)
Dx.set_xlabel('$\\beta/\\mathrm{SNR}$'); Dx.set_ylabel('KL per mode [nat]'); Dx.legend(fontsize=6, loc='upper right'); plabel(Dx, '(d)'); Dx.text(0.0125, 0.25, 'first-order only in this restricted\nparametrisation (Gaussian posterior)', fontsize=5.3)
fig.tight_layout(); savefig(fig, 'fig6')
caption('fig6', f"""**Figure 6. Collapse of the latent.** (a) KL versus $\\beta$ for the E2 independent decoder (two seeds: squares/light, triangles/dark; log--log axes; KL $=0$ points are drawn on the floor, marked by down-triangles): training from scratch with a 20-epoch warm-up (open) collapses for $\\beta\\ge0.3$ (and for seed 1 at $\\beta=0.1$), while continued training from the $\\beta=0.01$ solution keeps KL $>0$ up to $\\beta=10$ (seed 0/1: {d[(d.series=='continued')&(d.seed==0)&(d.beta==10)].KL.iloc[0]:.1f}/{d[(d.series=='continued')&(d.seed==1)&(d.beta==10)].KL.iloc[0]:.1f} nat). A single $\\times3.16$ step to $\\beta=31.6$ collapses both seeds; the fine grid ($\\times1.26$ steps, black-edged) is smooth. Dotted verticals: tangent lower bounds $\\beta_{{\\rm lb}}={lb[0]:.1f}$ (seed 0), ${lb[1]:.1f}$ (seed 1). Stars/crosses: B-2 (fixed $\\beta=31.75$, learning rate/3 from the $\\beta=25.2$ state) and flat start from $\\beta=10$. (b) Linear zoom, $\\beta=8$--40: fine-grid, continued and flat-start states of both seeds, with the B-2 state (large star: KL $=3.8$ nat at $\\beta=31.75$) still carrying information at the last grid point; seed 0 is at KL $={d[(d.series=='fine grid')&(d.seed==0)&(d.beta>31)].KL.iloc[0]:.1f}$ nat at $\\beta=31.75$. The curve is convex near the end, so a linear extrapolation of the fine grid is not a transition estimate (its zero depends on the fit window; see README). (c) Tangent construction in the $(R,D)$ plane (expected reconstruction with posterior-sampled $z$; $D_c={st.Dc.iloc[0]:.1f}$ of the collapsed solution, star at $R=0$): the fine-grid chain (line; numbers: $(D_c-D)/R$) ends on the tangent of slope $-\\beta_{{\\rm lb}}$ (dotted); the state through which the tangent passes (red diamond, maximal $(D_c-D)/R$) is, for seed 0, the chain state at $\\beta=31.76$ ({st[(st.seed==0)].ratio_samp.max():.2f}; the flat-start state at the same $\\beta$ gives {st[(st.seed==0)&(st.kind=='flat')].ratio_samp.iloc[0]:.2f}, so it is not the flat start) and, for seed 1, the B-2 state at $\\beta=31.75$ ({st[(st.seed==1)].ratio_samp.max():.2f}; the last chain state gives {st[(st.seed==1)&(st.kind=='chain')].ratio_samp.max():.2f}); flat-start (crosses) and B-2 (stars) states are different solutions at the same $\\beta$ and are not connected. Insets: full range. (d) Periodic-phase toy model (one mode, uniform prior over a period, bounded gain): first-order transition with KL jumping from 0.95 nat to zero at $\\beta/{{\\rm SNR}}={bin_:.3f}$ (dashed; dotted: metastable branch) versus continuous Gaussian water-filling; this first-order behaviour holds only in this restricted (Gaussian-posterior) parametrisation. Error bars: none; seed values are shown individually.""")
print('beta_zero', b0, 'slope', slope, 'lb', lb, 'bin', bin_)
