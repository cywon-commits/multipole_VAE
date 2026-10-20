"""Fig. 5: the bias valley. (a) E1 field-direction noise bias, (b) mask bias vs field deficit, (c) separate mask field removes the bias, (d) valley cross-section.
Data: data/fig5a_*.csv, fig5d_valley.csv (make_fig5_data.py); results/stage2b/stage2d_A1_profiles.json, results/stage2b/E2_hm_*, results/stage2e/B_kplus.json."""
from common import *
# theory coefficients (Gaussian order) for L = 32, 64 (same formula as stage2c_e1_bias.py)
def coef(L, J0=5.0, k0=10.0):
    q = 2*np.pi*np.arange(L)/L; gam = 2*np.cos(q)[:, None] + 2*np.cos(q)[None, :]; Sq = 1/(k0 + J0*(4 - gam)); S2 = (gam**2*Sq).mean(); P0 = k0 + 4*J0; a0 = J0/P0; r = 1 - 4*a0
    return 4*r/(a0*S2) - P0*r**2, -P0*r**2 - 16/S2
syn = pd.read_csv(DATA/'fig5a_synthetic.csv'); va = pd.read_csv(DATA/'fig5a_vae.csv'); vl = pd.read_csv(DATA/'fig5d_valley.csv')
A1 = jload(RES/'stage2b'/'stage2d_A1_profiles.json'); Bk = jload(RES/'stage2e'/'B_kplus.json')
mrows = []
for r in A1:
    if r['mode'] == 'shared': var = 'shared J (Stage 2b)'
    elif r['mode'] == 'm1': var = 'M1 (K,U; Stage 2c)'
    else: continue
    mrows.append(dict(variant=var, run=r['name'], deficit=r['deficit'], dJ_pct=r['dJ'], dD_pct=r['dD']))
for k, r in Bk.items():
    if k.startswith('E2_K1_ramp'): mrows.append(dict(variant='K+M1 continued (Stage 2e)', run=k, deficit=2.4 - r['h_unmasked'], dJ_pct=r['dJ'], dD_pct=r['dD']))
mb = pd.DataFrame(mrows); save_csv('fig5b_mask_bias.csv', mb)
hrows = []
for f in sorted((RES/'stage2b').glob('E2_hm_*/result.json')):
    r = jload(f); a = r['args']; hrows.append(dict(variant='M1-hm' if a['mask_mode'] == 'm1hm' else 'M2-hm', run=f.parent.name, dJ_pct=100*(r['J0']/6 - 1), dD_pct=100*(r['D']/6 - 1)))
for r in A1:
    if r['name'].startswith('E2_K_m1_'): hrows.append(dict(variant='M1 (Stage 2c)', run=r['name'], dJ_pct=r['dJ'], dD_pct=r['dD']))
hb = pd.DataFrame(hrows); save_csv('fig5c_hm.csv', hb)
fig, axs = plt.subplots(2, 2, figsize=(W2, 5.6)); (A, B_), (C, D) = axs
s2 = np.linspace(-1.5e-4, 6.2e-4, 50); cJ32, cK32 = coef(32); cJ64, cK64 = coef(64)
A.plot(s2*1e4, 100*cJ32*s2, c='k', lw=1.0); A.plot(s2*1e4, 100*cK32*s2, c='k', lw=1.0, ls='-'); A.plot(s2*1e4, 100*cJ64*s2, c='k', ls=':', lw=0.9); A.plot(s2*1e4, 100*cK64*s2, c='k', ls=':', lw=0.9)
A.axhline(0, **TRUTH); A.axvline(0, **TRUTH)
A.plot(syn.s2_eff*1e4, syn.dJ_pct, 'o', c='k', mfc='w', label='synthetic noise: $\\Delta J/J$'); A.plot(syn.s2_eff*1e4, syn.dK_pct, 's', c='k', mfc='w', label='synthetic: $\\Delta\\kappa/\\kappa$')
cth = 1.0
for r in va.itertuples():
    s2e = r.s2_eff if not np.isnan(r.s2_eff) else 3*r.beta/(2*(r.L**2)*(100/(10 + 20))) - cth/(r.L**2*10.0)   # theory s2* for points without a dump
    x = s2e*1e4; mk = PL['marker'] if r.seed == 0 else '^'; fc = PL['color'] if r.L == 32 else EXTRA[0]
    A.plot(x, r.dJ_pct, mk, c=fc, ms=4); A.plot(x, r.dK_pct, mk, c=fc, mfc='w', ms=4)
A.set_xlabel('$s^2_{\\rm eff}\\ [10^{-4}]$'); A.set_ylabel('bias [%]'); plabel(A, '(a)'); A.text(5.3, 100*cJ32*5.3e-4 + 0.1, '$\\Delta J/J$', fontsize=6); A.text(5.0, 100*cK32*5.0e-4 - 0.6, '$\\Delta\\kappa/\\kappa$', fontsize=6)
A.plot([], [], 'o', c=PL['color'], label='VAE $L{=}32$ (filled: $J$; open: $\\kappa$)'); A.plot([], [], 'o', c=EXTRA[0], label='VAE $L{=}64$'); A.plot([], [], c='k', label='theory $L{=}32$'); A.plot([], [], c='k', ls=':', label='theory $L{=}64$'); A.legend(fontsize=5.3, loc='lower left')
cm = {'shared J (Stage 2b)': ('o', 'none'), 'M1 (K,U; Stage 2c)': ('o', 'full'), 'K+M1 continued (Stage 2e)': ('^', 'full')}
xd = np.linspace(0, 0.7, 20); B_.plot(xd, 4.7*xd, c='k', lw=1.0); B_.plot(xd, -13.3*xd, c='k', lw=1.0); B_.axhline(0, **TRUTH)
for v, g in mb.groupby('variant'):
    m, fs = cm[v]; B_.plot(g.deficit, g.dJ_pct, m, c=PL['color'], mfc='none' if fs == 'none' else PL['color'], ms=3.5); B_.plot(g.deficit, g.dD_pct, m, c=EXTRA[0], mfc='none' if fs == 'none' else EXTRA[0], ms=3.5)
B_.set_xlabel('field deficit $\\delta=2.4-\\langle h_z\\rangle_{\\rm unmasked}$'); B_.set_ylabel('bias [%]'); plabel(B_, '(b)'); B_.text(0.45, 2.5, '$+4.7\\%\\,\\delta$ ($J$)', fontsize=6); B_.text(0.3, -6.7, '$-13.3\\%\\,\\delta$ ($D$)', fontsize=6)
for v, (m, fs) in cm.items(): B_.plot([], [], m, c='gray', mfc='none' if fs == 'none' else 'gray', label=v)
B_.plot([], [], 'o', c=PL['color'], label='$J$'); B_.plot([], [], 'o', c=EXTRA[0], label='$D$'); B_.legend(fontsize=5.3, loc='lower left')
order = ['M1 (Stage 2c)', 'M1-hm', 'M2-hm']; rng = np.random.default_rng(1)
C.axhspan(-0.5, 0.5, color='#e6e6e6', zorder=0); C.axhline(0, **TRUTH)
for i, v in enumerate(order):
    g = hb[hb.variant == v]; jit = rng.uniform(-0.15, 0.15, len(g)); C.plot(i - 0.15 + jit*0.3, g.dJ_pct, 'o', c=PL['color'], ms=3, mfc='none' if v == 'M1 (Stage 2c)' else PL['color']); C.plot(i + 0.15 + jit*0.3, g.dD_pct, 's', c=EXTRA[0], ms=3, mfc='none' if v == 'M1 (Stage 2c)' else EXTRA[0])
C.set_xticks(range(3)); C.set_xticklabels(['M1\n(shared field)', 'M1-hm\n(own field)', 'M2-hm\n(own field+bonds)'], fontsize=6.5); C.set_ylabel('bias [%]'); C.set_yscale('symlog', linthresh=1.0) if False else None; plabel(C, '(c)')
C.plot([], [], 'o', c=PL['color'], label='$\\Delta J/J$'); C.plot([], [], 's', c=EXTRA[0], label='$\\Delta D/D$'); C.legend(fontsize=6, loc='lower right'); C.text(2.45, 0.6, '$\\pm0.5\\%$', fontsize=6, ha='right')
D.plot(vl.deficit, vl.cost_fixed, 'o-', c='k', ms=3.5, label='$J,D$ fixed'); D.plot(vl.deficit, vl.cost_adapt, 's-', c=PL['color'], ms=3.5, label='$J,D$ re-fitted')
D.set_xlabel('field deficit $\\delta$'); D.set_ylabel('PL cost [nat/sample]'); D.set_yscale('log'); D.legend(fontsize=6.5, loc='upper left'); plabel(D, '(d)')
ins = D.inset_axes([0.58, 0.12, 0.38, 0.38]); ins.plot(vl.deficit, vl.dJ_per_unit_pct, 'o-', c=PL['color'], ms=3); ins.plot(vl.deficit, vl.dD_per_unit_pct, 's-', c=EXTRA[0], ms=3); ins.set_ylabel('%/unit $\\delta$', fontsize=6, labelpad=0); ins.set_xlabel('$\\delta$', fontsize=6, labelpad=0); ins.tick_params(labelsize=5)
fig.tight_layout(); savefig(fig, 'fig5')
from itertools import product
v64 = va[va.L == 64]; v32b1 = va[(va.L == 32) & (va.beta == 1.0)]
caption('fig5', f"""**Figure 5. The bias valley.** (a) E1 field-direction-noise bias of the coupled decoder: $\\Delta J/J$ and $\\Delta\\kappa/\\kappa$ versus the effective noise $s^2_{{\\rm eff}}$ (per transverse component, $10^{{-4}}$). Solid lines: theory $+{cJ32:.1f}\\,s^2$ and ${cK32:.1f}\\,s^2$ ($L=32$); dotted: $L=64$ ($+{cJ64:.1f}\\,s^2$, ${cK64:.1f}\\,s^2$; the two lines nearly coincide, the $L$ dependence enters through $s^2_{{\\rm eff}}$). Open black symbols: synthetic tests with independent noise ($10^{{-4}}$, $3\\times10^{{-4}}$) and the $\\hat M$ direction ($c=1$). Coloured symbols: VAE solutions (blue $L=32$; orange $L=64$; circles seed 0, triangles seed 1/2; filled $J$, open $\\kappa$); for the three $\\beta=1$, $L=32$ seeds $s^2_{{\\rm eff}}$ is measured on 4000 samples ($s^2_{{\\rm eff}}=${', '.join(f'{v*1e4:.2f}' for v in v32b1.s2_eff)}$\\times10^{{-4}}$), for the $\\beta$-scan and $L=64$ points it is the theory value $3\\beta/(2N\\kappa_{{\\rm eff}})-1/(N\\kappa_0)$ ($\\beta=0.25,0.5,1$). VAE $\\Delta J/J$ at $\\beta=1$: {', '.join(f'{v:+.2f}' for v in v32b1.dJ_pct)}\\%, at $L=64$: {', '.join(f'{v:+.2f}' for v in v64.dJ_pct)}\\%. (b) Mask bias versus the field deficit $\\delta=2.4-\\langle h_z\\rangle_{{\\rm unmasked}}$ ($J$ blue, $D$ orange): shared-$J$ decoders (Stage 2b, open), M1 decoders (Stage 2c, filled circles) and continued K+M1 solutions (Stage 2e, triangles) against the $L=48$ valley lines $+4.7\\%\\,\\delta$ and $-13.3\\%\\,\\delta$. The shared-$J$ points lie above the line because a second mechanism acts in addition to the valley: masked sites that keep only part of their neighbours push $J$ up directly (the misspecified factorised masked likelihood). (c) Separate field for the masked sites removes the bias: biases of the 16 runs each of M1-hm and M2-hm (2 seeds; $\\ell=2,8$; $\\rho=0.25,0.5$; max $|\\Delta J/J|=${hb[hb.variant.isin(['M1-hm','M2-hm'])].dJ_pct.abs().max():.2f}\\%, max $|\\Delta D/D|=${hb[hb.variant.isin(['M1-hm','M2-hm'])].dD_pct.abs().max():.2f}\\%) against 16 shared-field M1 runs (Stage 2c); grey band: $\\pm0.5\\%$. (d) Cross-section of the unmasked pseudo-likelihood along the valley ($L=48$, 320 samples, one seed): cost of a field deficit $\\delta$ with $J,D$ re-fitted or fixed; inset: valley slopes per unit deficit ({vl.dJ_per_unit_pct.min():.2f}--{vl.dJ_per_unit_pct.max():.2f}\\% for $J$, {vl.dD_per_unit_pct.min():.2f}--{vl.dD_per_unit_pct.max():.2f}\\% for $D$). No error bars (seed values are shown individually).""")
print('ok')
