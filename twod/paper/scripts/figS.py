"""Supplementary figures S1-S5.
S1: Stage 2e A1/A2 per-dimension KL spectra and the pre-registered water-filling prediction vs observation (failed prediction shown as is).
S2: Stage 2d A1 field profiles h_z(d). S3: E3 encoder sigma(B) vs the toy-VAE nonlinear encoder. S4: 1D DMI pitch identity and 1D field-chain correlation.
S5: E2 independent continued solution: decoded h_z next to the data S_z, with phase locking R(k)."""
from common import *
import re
# ================= S1 =================
pre = jload(RES/'stage2e'/'A2_prediction_preregistered.json'); obs = jload(RES/'stage2e'/'A2_observed.json')['rows']
rows = []
for s in (0, 1):
    for b in (0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        sp = np.load(RES/'stage2e'/f'spec_b{b if b != 1 else 1}_s{s}.npy') if (RES/'stage2e'/f'spec_b{b if b != 1 else 1}_s{s}.npy').exists() else None
        if sp is not None: rows.append(dict(seed=s, beta=b, KL_total=sp.sum(), median_dim=np.median(sp), max_dim=sp.max(), n_active=int((sp > 0.02).sum())))
S1a = pd.DataFrame(rows); save_csv('figS1_spectra_summary.csv', S1a)
pr = pd.DataFrame([dict(seed=s, kind=k, beta=b, KL_pred=v) for s in (0, 1) for k in ('spec', 'pca') for b, v in zip(pre['betas'], pre[f'{k}_s{s}']['KL_pred'])]); save_csv('figS1_prediction.csv', pr)
ob = pd.DataFrame([dict(seed=r[0], beta=r[1], KL_obs=r[4]) for r in obs]); save_csv('figS1_observed.csv', ob)
fig, (A, B_) = plt.subplots(1, 2, figsize=(W2, 3.0))
for b, c in zip((0.01, 0.1, 1, 3), ('#999999', EXTRA[1], IND['color'], EXTRA[0])):
    sp = np.sort(np.load(RES/'stage2e'/f'spec_b{b}_s0.npy'))[::-1]; A.plot(sp, c=c, lw=0.9, label=f'$\\beta={b:g}$')
A.set_yscale('log'); A.set_xlabel('latent dimension (sorted)'); A.set_ylabel('per-dimension KL [nat]'); A.legend(fontsize=6); plabel(A, '(a)')
kk = pd.concat([pd.DataFrame(dict(seed=S1a.seed, beta=S1a.beta, KL_obs=S1a.KL_total)), ob]).drop_duplicates(['seed', 'beta'], keep='last')
for s, mk in ((0, 's'), (1, '^')):
    g = kk[(kk.seed == s) & (kk.KL_obs > 0.2)].sort_values('beta'); B_.plot(g.beta, g.KL_obs, mk + '-', c=IND['color'], ms=3.5, lw=0.7, label=f'observed (seed {s})')
    z = kk[(kk.seed == s) & (kk.KL_obs <= 0.2)]; B_.plot(z.beta, [0.3]*len(z), 'v', c=IND['color'], ms=3.5)
    for k_, ls in (('spec', '-'), ('pca', ':')):
        h = pr[(pr.seed == s) & (pr.kind == k_)]; hh = h[h.KL_pred > 0.05]; B_.plot(hh.beta, hh.KL_pred, c='k', ls=ls, lw=0.8, alpha=1.0 if s == 0 else 0.5)
B_.plot([], [], c='k', label='prediction, diagonal spectrum'); B_.plot([], [], c='k', ls=':', label='prediction, PCA spectrum'); B_.set_xscale('log'); B_.set_yscale('log'); B_.set_xlabel('$\\beta$'); B_.set_ylabel('total KL [nat]'); B_.set_ylim(0.2, 2000); B_.legend(fontsize=5.5, loc='lower left'); plabel(B_, '(b)')
fig.tight_layout(); savefig(fig, 'figS1')
caption('S1', f"""**Figure S1. Reverse water-filling prediction versus observation (Stage 2e).** (a) Sorted per-dimension KL spectra (288 latent dimensions, seed 0, validation samples) of the continued solutions at $\\beta=0.01,0.1,1,3$; all dimensions are active ($>0.02$ nat) throughout $\\beta\\le3$. (b) Total KL versus $\\beta$ (log--log; down-triangles: KL $\\approx0$ drawn at the floor). Lines: the pre-registered predictions made from the $\\beta_0=1$ spectrum (diagonal posterior spectrum: solid; PCA information spectrum: dotted; seed 1 lighter), computed before the continuation $\\beta=3\\to10^4$ was run (file `A2_prediction_preregistered.json`); markers: observed (seed 0 squares, seed 1 triangles). The prediction fails: at $\\beta=10$ the observed KL is {ob[(ob.seed==0)&(ob.beta==10)].KL_obs.iloc[0]:.1f}/{ob[(ob.seed==1)&(ob.beta==10)].KL_obs.iloc[0]:.1f} nat (seeds 0/1) vs {pre['spec_s0']['KL_pred'][1]:.0f}/{pre['spec_s1']['KL_pred'][1]:.0f} predicted (diagonal), and the KL is already zero at $\\beta=31.6$ (a single $\\times3.16$ step, later shown to be an optimisation trap, see Fig. 6). Two seeds.""")
# ================= S2 =================
A1 = jload(RES/'stage2b'/'stage2d_A1_profiles.json'); rows = []
for r in A1:
    if r['mask_l'] == 8 and r['rho'] in (0.25, 0.5) and r['name'].endswith('s0'):
        rows.append(dict(run=r['name'], mode=r['mode'], rho=r['rho'], d=0, h_z=r['h_masked'], h_m_star=r['h_m_star']))
        for d, v in r['h_profile'].items(): rows.append(dict(run=r['name'], mode=r['mode'], rho=r['rho'], d=int(d), h_z=v, h_m_star=r['h_m_star']))
p2 = pd.DataFrame(rows); save_csv('figS2_profiles.csv', p2)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True); cmap = {'shared': EXTRA[0], 'm1': PL['color'], 'm2': SM['color']}
for k, rho in enumerate((0.25, 0.5)):
    for run, g in p2[p2.rho == rho].groupby('run'): ax[k].plot(g.d, g.h_z, 'o-', c=cmap[g['mode'].iloc[0]], ms=3, lw=0.8, label={'shared': 'shared $J$ (2b)', 'm1': 'M1 (2c)', 'm2': 'M2 (2c)'}[g['mode'].iloc[0]] if run.endswith('_s0') else None)
    ax[k].axhline(2.4, **TRUTH); ax[k].axhline(p2[p2.rho == rho].h_m_star.iloc[0], **TRUTH, ls='--'); ax[k].set_xlabel('distance $d$ to nearest masked site (0: masked sites)'); ax[k].set_title(f'$\\ell=8$, $\\rho={rho}$', fontsize=7.5); plabel(ax[k], '(a)' if k == 0 else '(b)')
ax[0].set_ylabel('$\\langle h_z\\rangle$'); h, l = ax[0].get_legend_handles_labels(); ax[0].legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), fontsize=6, loc='lower right'); fig.tight_layout(); savefig(fig, 'figS2')
caption('S2', f"""**Figure S2. Decoder field profiles around masked blocks (Stage 2d A1).** Mean decoded $h_z$ of the unmasked sites as a function of the Chebyshev distance $d$ to the nearest masked site ($d=0$: masked sites) for the shared-$J$ (Stage 2b), M1 and M2 (Stage 2c) decoders ($\\ell=8$, seed 0, 192 validation samples; (a) $\\rho=0.25$, (b) $\\rho=0.5$). Grey solid: $\\beta B=2.4$; grey dashed: $h_m^*=\\mathcal{{L}}^{{-1}}(\\langle S_z\\rangle_{{\\rm masked}})$ ({p2.h_m_star.min():.2f}--{p2.h_m_star.max():.2f}). The unmasked plateau lies below 2.4 and is flat for $d\\ge2$ (decoder receptive-field radius 3): the field deficit is a global level shift, not a boundary effect.""")
# ================= S3 =================
sc = pd.read_csv(DATA/'fig4c_e3_scatter.csv'); src = open(TWOD/'scripts'/'stage2d_toy_vae.py').read().replace('return dict(R2_sample', 'return dict(s_profile=s, B_grid=B, mu_profile=mu, R2_sample')
ns = {}; exec(compile(src.split("if __name__")[0], 'toy', 'exec'), ns); toy = ns['solve']('uniform', 17.07, False, 1.0)
tp = pd.DataFrame(dict(B_std=toy['B_grid'], sigma=toy['s_profile'], mu=toy['mu_profile'])); save_csv('figS3_toy_encoder.csv', tp)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9)); rows = []
for name, g in sc.groupby('run'):
    q = np.quantile(g.B, np.linspace(0, 1, 9)); idx = np.clip(np.digitize(g.B, q[1:-1]), 0, 7); cmean = [g.B[idx == i].mean() for i in range(8)]; smean = [g.sigma_used[idx == i].mean() for i in range(8)]
    kind = PL if '_pl_' in name else SM; ls = '-' if name.endswith('s0') else '--'; ax[0].plot(cmean, smean, ls, c=kind['color'], marker=seed_marker(kind, 0 if name.endswith('s0') else 1), ms=3, lw=0.8, label=name.replace('E3_conv_', ''))
    rows += [dict(run=name, B=b, sigma=s_) for b, s_ in zip(cmean, smean)]
save_csv('figS3_sigma_vs_B.csv', pd.DataFrame(rows)); ax[0].set_xlabel('$B$'); ax[0].set_ylabel('posterior $\\sigma$ of the used channel'); ax[0].legend(fontsize=5.5); plabel(ax[0], '(a)')
ax[1].plot(tp.B_std, tp.sigma, c='k'); ax[1].set_xlabel('standardised $B$'); ax[1].set_ylabel('toy-VAE encoder $\\sigma(B)$'); plabel(ax[1], '(b)'); fig.tight_layout(); savefig(fig, 'figS3')
caption('S3', f"""**Figure S3. Encoder noise profile for E3.** (a) Posterior standard deviation of the used channel (binned means over 8 equal-count $B$ bins, validation samples) for the four converged coupled runs (PL blue, SM green; seed 1 dashed with triangles); (b) the nonlinear encoder of the toy VAE (uniform source, SNR $=17.07$, $l=0$): $\\sigma(B)$ is U-shaped with $\\sigma_{{\\max}}/\\sigma_{{\\min}}={toy['s_ratio']:.2f}$. Observed ratios (bin means): {', '.join(f"{r.run.replace('E3_conv_','')}: {g.sigma.max()/g.sigma.min():.2f}" for r, g in [(pd.Series(dict(run=n)), gg) for n, gg in pd.DataFrame(rows).groupby('run')])}. The $l=1$ run (sm\\_alt\\_s1) is not converged (80 epochs).""")
# ================= S4 =================
pt = pd.read_csv(DATA/'1d_dmi_pitch.csv'); fc = pd.read_csv(DATA/'1d_field_corr.csv')
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9)); lim = [0, 1.7]; ax[0].plot(lim, lim, **TRUTH)
g = pt[pt.kind == 'grid']; ax[0].plot(g.expected, g.measured, 'o', c=SM['color'], mfc='none', label='D along $z$ (grid)'); c_ = pt[pt.kind == 'case']; ax[0].plot(c_.expected, c_.measured, 's', c=SM['color'], label='cases A, B, C')
ax[0].set_xlabel('$D/J$'); ax[0].set_ylabel(r'$\langle e\cdot(S_i\times S_{i+1})\rangle/\langle S_\perp\cdot S_\perp\rangle$'); ax[0].legend(fontsize=6); plabel(ax[0], '(a)')
cc = {1.0: EXTRA[1], 2.0: PL['color'], 5.0: EXTRA[0], 10.0: EXTRA[2], 20.0: '#555555'}
for J, g in fc.groupby('J'): ax[1].semilogy(g.r, g.C_gauss, c=cc[J], lw=0.9, label=f'$J={J:g}$'); gg = g[g.C_mc > 3e-4]; ax[1].semilogy(gg.r, gg.C_mc, 'o', c=cc[J], mfc='none', ms=3.5)
ax[1].set_xlabel('$r$'); ax[1].set_ylabel('$C(r)$ (transverse, 1D)'); ax[1].legend(fontsize=6); plabel(ax[1], '(b)'); fig.tight_layout(); savefig(fig, 'figS4')
caption('S4', f"""**Figure S4. 1D reference checks.** (a) DMI pitch identity: $\\langle\\hat e\\cdot(\\mathbf S_i\\times\\mathbf S_{{i+1}})\\rangle/\\langle\\mathbf S_{{i\\perp}}\\cdot\\mathbf S_{{i+1\\perp}}\\rangle$ ($\\hat e=\\hat D$) against $D/J$ for an open chain ($N=100$, $J=1$; 800 samples per point, one seed); max deviation {abs(pt.measured-pt.expected).max():.3f}. Cases A, B, C: A $J=2$, $\\mathbf D=(0,1,0)$ (cycloid); B $J=2$, $\\mathbf D=(1,0,0)$ (helix); C $J=1$, $\\mathbf D=(0,0.3,0)$ (200 chains, 800 samples, one seed each). (b) Transverse correlation of the external-field chain ($\\kappa=10$, $J=1,2,5,10,20$; 1000 samples, one seed; points with $C<3\\times10^{{-4}}$ omitted) against the lattice Gaussian solution $2G(r)$ (lines). No error bars.""")
# ================= S5 =================
pl = pd.read_csv(DATA/'figS5_phaselock.csv'); fig = plt.figure(figsize=(W2, 3.4)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], wspace=0.15, hspace=0.12)
for j in range(3):
    d = pd.read_csv(DATA/f'figS5_e2_sample{j}.csv'); Sz = d.Sz.values.reshape(48, 48); hz = d.hz.values.reshape(48, 48)
    a0 = fig.add_subplot(gs[0, j]); a0.imshow(Sz.T, origin='lower', cmap='coolwarm', vmin=-1, vmax=1); a0.set_xticks([]); a0.set_yticks([]); a0.set_title(f'data $S_z$, sample {j+1}', fontsize=6.5)
    a1 = fig.add_subplot(gs[1, j]); a1.imshow(hz.T, origin='lower', cmap='coolwarm', vmin=-hz.std()*3 + hz.mean(), vmax=hz.std()*3 + hz.mean()); a1.set_xticks([]); a1.set_yticks([]); a1.set_title('decoded $h_z$ (independent, $\\beta=1$)', fontsize=6)
ax = fig.add_subplot(gs[:, 3]); ax.bar(range(3), pl.R, color=IND['color']); ax.axhline(0.05, **APPROX); ax.set_xticks(range(3)); ax.set_xticklabels([f'({int(r.kx)},{int(r.ky)})' for r in pl.itertuples()], fontsize=6); ax.set_ylabel('phase lock $R(k)$'); ax.set_ylim(0, 1.05); ax.set_title('texture wavevectors', fontsize=6.5)
savefig(fig, 'figS5')
caption('S5', f"""**Figure S5. Independent decoder (continued solution) follows the texture.** Top: three validation samples of the E2 skyrmion lattice (data $S_z$); bottom: the field $h_z$ decoded from the posterior mean of the continued $\\beta=1$ independent decoder (seed 0; colour scale: mean $\\pm3$ s.d.). Right: Fourier-phase lock $R(\\mathbf k)$ between decoded $h_z$ and data $S_z$ at the three dominant texture wavevectors (192 validation samples): {', '.join(f'{r:.3f}' for r in pl.R)} (dotted: chance level $\\approx0.05$).""")
# ================= S6 =================
al = pd.read_csv(DATA/'fig4a_e1_align.csv'); fig, A = plt.subplots(figsize=(W1, 2.6))
for mode, kind in (('indep', IND), ('pl', PL), ('sm_alt', SM)):
    for s_ in (0, 1, 2):
        v = np.sort((1 - al[(al['mode'] == mode) & (al.seed == s_)].abs_cos.values).clip(1e-7)); A.step(v, np.arange(1, len(v) + 1)/len(v), c=kind['color'], lw=0.9, ls=['-', '--', ':'][s_], where='post')
A.set_xscale('log'); A.set_xlabel('$1-|\\cos\\angle(z_{l=1},\\hat n)|$'); A.set_ylabel('cumulative fraction')
for mode, kind, lab in (('indep', IND, 'indep.'), ('pl', PL, 'PL'), ('sm_alt', SM, 'SM')): A.plot([], [], c=kind['color'], label=lab)
A.legend(fontsize=6, loc='lower right'); fig.tight_layout(); savefig(fig, 'figS6')
al_s = al.groupby('mode').abs_cos.agg(['min', 'mean'])
caption('S6', f"""**Figure S6. E1 latent direction versus field direction.** Distribution of $1-|\\cos|$ between the used $l=1$ latent (posterior mean) and the true field direction $\\hat n$ on 800 validation samples for the three decoders (colour; line style: seed 0, 1, 2; $L=32$, $J=5$, $\\kappa=10$). The nine curves coincide: the latent direction equals the direction of each sample's magnetisation, which differs from $\\hat n$ by the finite-size fluctuation ($\\sim10^{{-4}}$ in $1-|\\cos|$), so the curves do not discriminate between the decoders. Mean $|\\cos|$: independent {al_s.loc['indep','mean']:.5f}, PL {al_s.loc['pl','mean']:.5f}, SM-alternating {al_s.loc['sm_alt','mean']:.5f} (minimum {al_s['min'].min():.4f}; the float32 limit is $\\sim10^{{-4}}$).""")
print('S done')
