"""Fig. 3: identifiability, collinear vs non-collinear. (a) normalised cond(M) (26 parameters) vs B with <Q> (Stage 1c);
(b) collinear flat direction from the all-open fits: 1D (replicated fits, principal axis of (dJ, dkappa, dG2)) and 2D (Stage 1a)."""
from common import *
c = jload(RES/'stage1c'/'summary.json'); a = jload(RES/'stage1a'/'summary.json')
ca = pd.DataFrame([dict(B=float(k[2:]), cond=v['cond'], Q_mean=v['Q_mean'], Q_std=v['Q_std'], mz=v['mz']) for k, v in c.items()]); save_csv('fig3a_cond.csv', ca)
f = pd.read_csv(DATA/'1d_flat.csv'); ki = pd.read_csv(DATA/'1d_kappa_ind.csv').set_index('J'); rows = []
for J, g in f.groupby('J'):
    X = g[['dJ', 'dkappa', 'dG2']].values; X = X - X.mean(0); w, V = np.linalg.eigh(np.cov(X.T)); v = V[:, -1]; v = v/v[0]
    rows.append(dict(system='1D', J=J, dkappa_over_dJ=v[1], dG2_over_dJ=v[2], pred_kappa=-6.0, pred_G2=-np.sqrt(6), n_rep=len(g), cond=np.nan, delta2=ki.delta2_mc.loc[J]))
for k, v in a.items():
    if k.startswith('J='): rows.append(dict(system='2D', J=float(k[2:]), dkappa_over_dJ=v['flat_ratios'][0]*(-6.0)/(-6.0)*(-6.0) if False else v['flat_ratios'][0], dG2_over_dJ=-np.sqrt(6)*np.mean(v['flat_ratios'][1:]), pred_kappa=-6.0, pred_G2=-np.sqrt(6), n_rep=1, cond=v['cond_free'], delta2=v['C'][0]))
fb = pd.DataFrame(rows); save_csv('fig3b_flat.csv', fb)
fig, (A, B_) = plt.subplots(1, 2, figsize=(W2, 2.9))
A.semilogy(ca.B[:3], ca.cond[:3], 'o-', c=PL['color'], label='non-collinear SkX'); A.semilogy(ca.B[3:], ca.cond[3:], 's', c=EXTRA[0], label='polarised (collinear)')
A.set_xlabel('$B$'); A.set_ylabel('normalised cond($M$), 26 parameters'); plabel(A, '(a)'); A.set_ylim(30, 1e5)
A2 = A.twinx(); A2.errorbar(ca.B, -ca.Q_mean, yerr=ca.Q_std, fmt='k^', ms=3.5, capsize=1.5, lw=0.6); A2.set_ylabel('$-\\langle Q\\rangle$ (triangles; bars: sample s.d.)'); A2.spines['right'].set_visible(True)
A.legend(loc='center left', fontsize=6.5, bbox_to_anchor=(0.12, 0.62))
x = np.arange(len(fb)); lab = [f"{r.system}\n$J{{=}}{r.J:g}$" for r in fb.itertuples()]
B_.axhline(1, **TRUTH); B_.plot(x - 0.1, fb.dkappa_over_dJ/fb.pred_kappa, 'o', c=PL['color'], label='$\\Delta\\kappa/\\Delta J$ (pred. $-6$)'); B_.plot(x + 0.1, fb.dG2_over_dJ/fb.pred_G2, 's', c=EXTRA[0], label='$\\Delta G_2/\\Delta J$ (pred. $-\\sqrt{6}$)')
B_.set_xticks(x); B_.set_xticklabels(lab, fontsize=6.5); B_.set_ylabel('measured / predicted flat-direction ratio'); B_.set_ylim(0.6, 1.3); B_.legend(fontsize=6.5, loc='lower right'); plabel(B_, '(b)')
fig.tight_layout(); savefig(fig, 'fig3')
r5 = fb[(fb.system == '1D')]; r2 = fb[fb.system == '2D']
caption('fig3', f"""**Figure 3. Identifiability: collinear vs non-collinear order.** (a) Column-normalised condition number of the 26-parameter score-matching matrix $M$ (Stage 1c, $L=48$, $J=D=1$ interfacial DMI, $\\beta=6$, one seed, 60 chains) versus $B$: {ca.cond[0]:.0f}, {ca.cond[1]:.0f}, {ca.cond[2]:.0f} in the skyrmion lattice ($B=0.2,0.4,0.6$; $\\langle Q\\rangle=${', '.join(f'{q:.1f}' for q in ca.Q_mean[:3])}) against {ca.cond[3]:.2e} in the polarised state ($B=1.5$, $Q=0$), a ratio of {ca.cond[3]/ca.cond[1]:.0f} (vs $B=0.4$). (b) Collinear flat direction of the all-open fits, as measured/predicted ratio of the flat-direction coefficients ($\\Delta\\kappa/\\Delta J$ predicted $-6$, $\\Delta G_2/\\Delta J$ predicted $-\\sqrt6$). 1D: principal axis of $(\\Delta J,\\Delta\\kappa,\\Delta G_2)$ over {int(r5.n_rep.iloc[0])} independent chain-ensemble replicates at $J=5,20$, $\\kappa=10$ (coefficients: $\\Delta\\kappa/\\Delta J={', '.join(f'{v:.2f}' for v in r5.dkappa_over_dJ)}$, $\\Delta G_2/\\Delta J={', '.join(f'{v:.2f}' for v in r5.dG2_over_dJ)}$). 2D: Stage 1a all-21-parameter fit ($L=32$, one seed); $\\Delta\\kappa/(\\Delta J_x+\\Delta J_y)=${', '.join(f'{v:.2f}' for v in r2.dkappa_over_dJ)} and mean $G_2/(-\\sqrt6\\,\\Delta J_\\mu)=${', '.join(f'{-v/np.sqrt(6):.2f}' for v in r2.dG2_over_dJ)} for $J=2,5,20$; cond$(M)$ of the 21-parameter fit {', '.join(f'{v:.1e}' for v in r2.cond)}. The 1D replicate axis deviates from the predicted $-6$ by {abs(r5.dkappa_over_dJ/(-6)-1).max()*100:.0f}\\% at most (see README). The predicted direction $(1,-6,-\\sqrt6)$ is exactly flat only at harmonic order; it is fixed by the anharmonic terms alone, so the principal axis of the fit scatter deviates from the harmonic direction by $O(\\langle|\\delta|^2\\rangle)$. $\\langle|\\delta|^2\\rangle$ (CSV column \\texttt{{delta2}}) is {', '.join(f'{v:.3f}' for v in r5.delta2)} for 1D $J=5,20$ ($\\kappa=10$, MC) and {', '.join(f'{v:.3f}' for v in r2.delta2)} for 2D $J=2,5,20$ (Stage 1a $C(0)$): it is smaller in 2D than in 1D at $J=5$ and $J=20$ but not at $J=2$, where it is comparable to the 1D $J=5$ value. No error bars (see text for the replicate counts).""")
print(fb.round(3).to_string())
