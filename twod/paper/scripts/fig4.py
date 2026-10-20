"""Fig. 4: what the latent encodes. (a) E1 alignment, (b) E3 used-channel KL vs PL Cramer-Rao bound and toy prediction, (c) decoded h_z/beta vs true B,
(d) R^2 of B (sampled z and posterior-mean z), (e) E2: coupled KL ~ 0 vs independent continued solution."""
from common import *
R = RES/'stage2b'
al = pd.read_csv(DATA/'fig4a_e1_align.csv'); sc = pd.read_csv(DATA/'fig4c_e3_scatter.csv')
runs = [('E3_conv_pl_s0', PL, 0), ('E3_conv_pl_s1', PL, 1), ('E3_conv_sm_alt_s0', SM, 0), ('E3_conv_sm_alt_s1', SM, 1)]
rows = []
for name, kind, seed in runs:
    r = jload(R/name/'result.json'); kl = r['kl_per_dim_val']; used = int(np.argmax(kl))
    rows.append(dict(run=name, decoder='PL' if 'pl' in name else 'SM', seed=seed, l=0 if used < 2 else 1, KL_used=max(kl), R2_sampled=r['B_probe_R2_heldout_sampled_z']['all'], R2_mean=r['B_probe_R2_heldout_mean_z']['all'],
                     epochs=r['epochs_run'], converged=r['converged_at'] is not None))
for s in (0, 1):
    r = jload(R/f'E3_pl_givenB_s{s}'/'result.json'); rows.append(dict(run=f'E3_pl_givenB_s{s}', decoder='control', seed=s, l=-1, KL_used=r['kl_total'], R2_sampled=np.nan, R2_mean=np.nan, epochs=r['epochs_run'], converged=False))
e3 = pd.DataFrame(rows); e3['KL_bound'] = 1.2441535; e3['toy_KL'] = np.where(e3.l == 1, 1.662, 1.373); e3['toy_R2_sampled'] = np.where(e3.l == 1, 0.854, 0.918); e3['toy_R2_mean'] = np.where(e3.l == 1, 0.999, 0.993); e3['R2_max'] = 0.9416489
save_csv('fig4b_e3_kl_r2.csv', e3)
e2 = []
for lab, name in (('coupled (PL) beta=1, E2', 'E2_pl_b1.0_jlr50'), ('coupled (SM) beta=1, E2', 'E2_sm_alt_b1.0_s0')):
    r = jload(RES/'stage2'/name/'result.json') if (RES/'stage2'/name).exists() else jload(R/name/'result.json'); e2.append(dict(label=lab, decoder='PL' if 'PL' in lab else 'SM', seed=0, KL=r['kl_total'] if 'kl_total' in r else sum(r['kl_per_dim_val'])))
for s in (0, 1): e2.append(dict(label='independent, continued, beta=1', decoder='indep', seed=s, KL=jload(R/f'E2_ramp_s{s}'/'result.json')['kl_total']))
e2 = pd.DataFrame(e2); save_csv('fig4e_e2_kl.csv', e2)
fig = plt.figure(figsize=(W2, 5.2)); gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=1.2)
A = fig.add_subplot(gs[0, 0:2]); Bx = fig.add_subplot(gs[0, 2:4]); Cx = fig.add_subplot(gs[0, 4:6]); Dx = fig.add_subplot(gs[1, 0:3]); Ex = fig.add_subplot(gs[1, 3:6])
for mode, kind in (('indep', IND), ('pl', PL), ('sm_alt', SM)):
    for s in (0, 1, 2):
        v = np.sort((1 - al[(al['mode'] == mode) & (al.seed == s)].abs_cos.values).clip(1e-7)); A.step(v, np.arange(1, len(v) + 1)/len(v), c=kind['color'], lw=0.9, ls=['-', '--', ':'][s], where='post')
A.set_xscale('log'); A.set_xlabel('$1-|\\cos\\angle(z_{l=1},\\hat n)|$'); A.set_ylabel('cumulative fraction'); plabel(A, '(a)')
for mode, kind, lab in (('indep', IND, 'indep.'), ('pl', PL, 'PL'), ('sm_alt', SM, 'SM')): A.plot([], [], c=kind['color'], label=lab)
A.legend(fontsize=6, loc='lower right')
xs = np.arange(len(e3)); 
for i, r in enumerate(e3.itertuples()):
    kind = PL if r.decoder in ('PL', 'control') else SM
    Bx.bar(i, max(r.KL_used, 0.0), color='white' if r.decoder == 'control' else kind['color'], edgecolor=kind['color'], lw=0.8, hatch='//' if r.l == 1 else None)
    if r.decoder == 'control': Bx.plot([i], [0.03], 'v', c=kind['color'], ms=4)
Bx.axhline(1.2441535, **APPROX); Bx.axhline(1.373, **PRED, xmin=0.0, xmax=0.5); Bx.axhline(1.662, **PRED, xmin=0.5, xmax=0.69)
Bx.set_xticks(xs); Bx.set_xticklabels([f"{('PL' if r.decoder=='PL' else 'SM' if r.decoder=='SM' else 'ctrl')}\ns{r.seed}" for r in e3.itertuples()], fontsize=6); Bx.set_ylabel('KL of used channel [nat]'); Bx.set_ylim(0, 2.0); plabel(Bx, '(b)')
Bx.text(5.45, 1.17, 'PL Cramer-Rao 1.25 (dotted)', fontsize=5.5, ha='right', va='top'); Bx.text(3.6, 1.69, 'toy $l{=}1$: 1.66', fontsize=5.5, va='bottom'); Bx.text(-0.4, 1.40, 'toy $l{=}0$: 1.37', fontsize=5.5, va='bottom')
sub = sc[sc.run == 'E3_conv_pl_s0']; Cx.plot([0.2, 0.6], [0.2, 0.6], **TRUTH); Cx.plot(sub.B, sub.hz_sampled_z, '.', c='#56B4E9', ms=3, label='sampled $z$'); Cx.plot(sub.B, sub.hz_mean_z, '.', c=PL['color'], ms=3, label='posterior-mean $z$')
Cx.set_xlabel('true $B$'); Cx.set_ylabel('decoded $h_z/\\beta$'); Cx.legend(fontsize=6, markerscale=2.5, loc='upper left'); plabel(Cx, '(c)')
for i, r in enumerate(e3[e3.decoder != 'control'].itertuples()):
    kind = PL if r.decoder == 'PL' else SM; mk = seed_marker(kind, r.seed)
    Dx.plot(i - 0.12, r.R2_sampled, mk, c=kind['color'], mfc='none'); Dx.plot(i + 0.12, r.R2_mean, mk, c=kind['color'])
Dx.axhline(0.918, **PRED, xmin=0.0, xmax=0.7); Dx.axhline(0.993, **PRED, xmin=0.0, xmax=0.7); Dx.axhline(0.854, **PRED, xmin=0.7, xmax=1.0); Dx.axhline(0.9416, **APPROX)
Dx.set_xticks(range(4)); Dx.set_xticklabels([f"{r.decoder}\ns{r.seed}\n$l{{=}}{r.l}$" for r in e3[e3.decoder != 'control'].itertuples()], fontsize=6); Dx.set_ylabel('$R^2$ of $B$'); Dx.set_ylim(0.78, 1.01); plabel(Dx, '(d)')
Dx.plot([], [], 'o', c='gray', mfc='none', label='sampled $z$ (open)'); Dx.plot([], [], 'o', c='gray', label='posterior mean $z$ (filled)'); Dx.plot([], [], **PRED, label='toy VAE'); Dx.plot([], [], **APPROX, label='$R^2_{\\max}$ (PL)'); Dx.legend(fontsize=5.5, loc='center', bbox_to_anchor=(0.5, 0.83), ncol=2, columnspacing=0.8, handlelength=1.2)
for i, r in enumerate(e2.itertuples()):
    kind = PL if r.decoder == 'PL' else SM if r.decoder == 'SM' else IND; mk = seed_marker(kind, r.seed)
    if r.KL < 0.05: Ex.plot(i, 0.1, 'v', c=kind['color'], ms=5)
    else: Ex.plot(i, r.KL, mk, c=kind['color'])
Ex.set_yscale('log'); Ex.set_ylim(0.05, 1500); Ex.axhline(0.1, color='#cccccc', lw=0.4, ls='-'); Ex.set_xticks(range(len(e2))); Ex.set_xticklabels(['coupled\nPL', 'coupled\nSM', 'indep.\ns0', 'indep.\ns1'], fontsize=6)
Ex.set_ylabel('KL [nat]'); Ex.text(0.5, 0.2, 'triangles down: KL $<0.05$, drawn at the floor', fontsize=5.5, ha='center', va='bottom'); plabel(Ex, '(e)')
fig.tight_layout(); savefig(fig, 'fig4')
al_s = al.groupby('mode').abs_cos.agg(['min', 'mean'])
caption('fig4', f"""**Figure 4. What the latent encodes.** (a) E1 ($L=32$, $J=5$, $\\kappa=10$, random field direction): distribution of $1-|\\cos|$ between the used $l=1$ latent (posterior mean) and the true field direction on 800 validation samples for the three decoders (line style: seed 0, 1, 2); the nine curves coincide (the latent direction equals the direction of each sample's magnetisation, which deviates from $\\hat n$ by the finite-size fluctuation $\\sim10^{-4}$ in $1-|\\cos|$); mean $|\\cos|$: independent {al_s.loc['indep','mean']:.5f}, PL {al_s.loc['pl','mean']:.5f}, SM-alternating {al_s.loc['sm_alt','mean']:.5f} (minimum {al_s['min'].min():.4f}; the float32 limit is $\\sim10^{{-4}}$). (b) E3 ($B\\sim U[0.2,0.6]$ per chain): KL of the used latent channel (bars; hatched: $l=1$ channel) vs the pseudo-likelihood Cram\\'er--Rao bound $1.24$ nat (dotted) and the toy-VAE prediction (solid: $l=0$ 1.37, $l=1$ 1.66). Used KL: {', '.join(f'{r.KL_used:.2f}' for r in e3[e3.decoder!='control'].itertuples())}; controls with the true $B$ given to the decoder: {', '.join(f'{r.KL_used:.4f}' for r in e3[e3.decoder=='control'].itertuples())}. Two seeds each; SM-alternating seed 1 was stopped at {int(e3[(e3.decoder=='SM')&(e3.seed==1)].epochs.iloc[0])} epochs without meeting the convergence criterion. (c) Decoded uniform field $h_z/\\beta$ against the true $B$ (PL seed 0, validation samples): posterior-mean $z$ (blue) and a posterior sample (light blue); line: identity. (d) $R^2$ of a ridge regression of $B$ on the latent (held-out chains): sampled $z$ (open) and posterior-mean $z$ (filled) vs the toy-VAE values (solid: 0.918/0.993 for $l=0$, 0.854 for $l=1$) and the pseudo-likelihood $R^2_{{\\max}}=0.942$ (dotted); sampled-$z$ values {', '.join(f'{r.R2_sampled:.3f}' for r in e3[e3.decoder!='control'].itertuples())}, mean-$z$ values {', '.join(f'{r.R2_mean:.3f}' for r in e3[e3.decoder!='control'].itertuples())}. (e) E2 skyrmion lattice: the coupled decoder's KL is $\\approx0$ at $\\beta=1$ (Markov theorem; the collapse is the trivial solution) while the independent decoder's continued solution carries {', '.join(f'{v:.0f}' for v in e2[e2.decoder=='indep'].KL)} nat (seeds 0, 1). Error bars: none (single-chain-ensemble values; seed spread shown by the two seeds).""")
print(e3.round(3).to_string()); print(e2)
