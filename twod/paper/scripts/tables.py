"""Tables 1 and 2 (CSV + LaTeX booktabs). Table 1: estimator recovery (1D chains from data/1d_*.csv, 2D Stage 1a/1b/1c from results/stage1*/summary.json).
Table 2: information summary (E1, E3, E2, collapse) from results/stage2b, results/stage2e, results/stage2f_A_bound.json."""
from common import *
from s2spin import NAMES
rows = []
e = pd.read_csv(DATA/'1d_exact_est.csv'); a12 = e[e.fit == 'all12'].set_index('J'); j0 = e[e.fit == 'J0_only'].set_index('J')
for J in (0.5, 2.0, 5.0): rows.append(dict(system='1D exact chain', case=f'J={J:g}', quantity='J', truth=J, SM=j0.loc[J].SM_J0, SM_se=j0.loc[J].SM_J0_se, PL=j0.loc[J].PL_J0, cond=a12.loc[J].cond, note=f"max|z| zero params {a12.loc[J].maxabs_z_zero_params:.2f} (12-param fit)"))
dm = pd.read_csv(DATA/'1d_dmi_est.csv'); pt = pd.read_csv(DATA/'1d_dmi_pitch.csv').set_index('case')
for cs, g in dm.groupby('case'):
    z0 = ((g[g.true == 0].SM/g[g.true == 0].SM_se).abs()).max()
    for r in g[g.true != 0].itertuples(): rows.append(dict(system='1D DMI chain', case=cs, quantity=r.param, truth=r.true, SM=r.SM, SM_se=r.SM_se, PL=r.PL, cond=pt.loc[cs].cond if r.param == 'J0' else np.nan, note=f'max|z| zero params {z0:.2f}' if r.param == 'J0' else ''))
a = jload(RES/'stage1a'/'summary.json')
for k, v in a.items():
    if k.startswith('J='): rows.append(dict(system='2D Gaussian (1a)', case=f'J={k[2:]}, kappa=10', quantity='J', truth=float(k[2:]), SM=v['sm_tied'][0], SM_se=v['sm_tied_se'][0], PL=v['pl_tied'][0], cond=v['cond_free'], note='tied (J,h_z); cond: 21 parameters open'))
b = jload(RES/'stage1b'/'summary.json'); tab = []
for key, v in b.items():
    tr = np.array(v['true'][:21]); th = np.array(v['theta'][:21]); se = np.array(v['se']); pl = np.array(v['pl'][:21]); nz = tr != 0
    tab.append(dict(key=key, maxz_nonzero=np.max(np.abs((th - tr)[nz]/se[nz])), maxz_zero=np.max(np.abs(th[~nz]/se[~nz])), max_SM_PL=np.max(np.abs(th - pl)), cond=v['cond']))
tb = pd.DataFrame(tab); save_csv('table1_stage1b_cases.csv', tb)
rows.append(dict(system='2D DMI (1b)', case='15 cases (none/interfacial/bulk, D, beta=0.5,1,2)', quantity='all 21 parameters', truth=np.nan, SM=np.nan, SM_se=np.nan, PL=np.nan, cond=np.nan,
                 note=f"max|z| non-zero {tb.maxz_nonzero.max():.2f}; max|z| zero {tb.maxz_zero.max():.2f}; max|SM-PL| {tb.max_SM_PL.max():.4f}; cond {tb.cond.min():.1f}-{tb.cond.max():.0f}"))
c = jload(RES/'stage1c'/'summary.json')
for k, v in c.items():
    tr = np.array(v['true']); th = np.array(v['theta']); se = np.array(v['se']); nz = tr != 0; z0 = np.max(np.abs(th[~nz]/se[~nz])); zn = np.max(np.abs((th - tr)[nz]/se[nz]))
    rows.append(dict(system='2D SkX (1c)', case=k, quantity='26 parameters', truth=np.nan, SM=np.nan, SM_se=np.nan, PL=np.nan, cond=v['cond'], note=f"<Q>={v['Q_mean']:.1f}; max|z| non-zero {zn:.2f}; max|z| zero (incl. Gamma, Q) {z0:.2f}"))
t1 = pd.DataFrame(rows); save_csv('table1_estimators.csv', t1)
def tex(df, cols, heads, fmt, name, caption_, label):
    lines = ['\\begin{table}[t]', '\\centering', '\\small', f'\\caption{{{caption_}}}', f'\\label{{{label}}}', '\\begin{tabular}{' + 'l'*len(cols) + '}', '\\toprule', ' & '.join(heads) + ' \\\\', '\\midrule']
    for _, r in df.iterrows(): lines.append(' & '.join(f(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else '--' for c, f in zip(cols, fmt)) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']; (DATA/name).write_text('\n'.join(lines) + '\n')
s = lambda x: str(x).replace('_', '\\_').replace('|', '$|$').replace('<', '$<$').replace('>', '$>$')
tex(t1, ['system', 'case', 'quantity', 'truth', 'SM', 'SM_se', 'PL', 'cond', 'note'], ['system', 'case', 'param.', 'truth', 'SM', 'jack.\\ err.', 'PL', 'cond($M$)', 'notes'],
    [s, s, s, lambda x: f'{x:.3g}', lambda x: f'{x:.4f}', lambda x: f'{x:.4f}', lambda x: f'{x:.4f}', lambda x: f'{x:.3g}', s], 'table1_estimators.tex', 'Estimator recovery (SM: score matching with chain-group jackknife errors; PL: pseudo-likelihood). 1D chains were re-run with fixed seeds; 2D from Stage 1.', 'tab:estimators')
# ---------------- Table 2 ----------------
R = RES/'stage2b'; rows = []
for mode in ('pl', 'indep'):
    ks = [jload(R/f'E1_{mode}_s{s}'/'result.json')['kl_total'] for s in range(3)]
    rows.append(dict(block='E1', item=f'{mode} KL [nat]', observed=f'{np.mean(ks):.2f} ({min(ks):.2f}-{max(ks):.2f}, 3 seeds)', prediction='9.95' if mode == 'pl' else '13.01'))
for name in ('E3_conv_pl_s0', 'E3_conv_pl_s1', 'E3_conv_sm_alt_s0', 'E3_conv_sm_alt_s1'):
    r = jload(R/name/'result.json'); kl = max(r['kl_per_dim_val']); ie = jload(RES/'stage2e'/'C_ieff.json').get(name, {}).get('I_eff_over_I_PL', np.nan)
    rows.append(dict(block='E3', item=f'{name}: KL, R2 sampled/mean, I_eff/I_PL', observed=f"{kl:.2f}, {r['B_probe_R2_heldout_sampled_z']['all']:.3f}/{r['B_probe_R2_heldout_mean_z']['all']:.3f}, " + (f'{ie:.2f}' if ie == ie else 'n/a'), prediction=('KL>=1.24 (PL CR bound), toy 1.66; R2 0.854 (l=1, SM s1)' if name.endswith('sm_alt_s1') else 'KL>=1.24 (PL CR bound), toy 1.37; R2 0.918/0.993 (l=0); I_eff/I_PL ' + ('0.8 (PL)' if '_pl_' in name else '1.0 (SM, predicted)'))))
rows.append(dict(block='E3', item='control (true B given): KL', observed='0.000, 0.000', prediction='0'))
rows.append(dict(block='E2', item='coupled decoder KL (beta=1)', observed=f"{sum(jload(RES/'stage2'/'E2_pl_b1.0_jlr50'/'result.json')['kl_per_dim_val']):.1e} (collapsed, Markov theorem)", prediction='0'))
rows.append(dict(block='E2', item='independent continued KL (beta=1), 2 seeds', observed=', '.join(f"{jload(R/f'E2_ramp_s{s}'/'result.json')['kl_total']:.0f}" for s in (0, 1)), prediction='> 0 (collapse is a trap)'))
bd = jload(RES/'stage2f_A_bound.json')
rows.append(dict(block='collapse', item='tangent lower bound beta_lb (sampled z), seeds 0/1', observed=f"{bd['beta_lb_sampled_z_s0']:.1f} / {bd['beta_lb_sampled_z_s1']:.1f}", prediction='>= 24.3 (from beta<=10 states, mean z)'))
rows.append(dict(block='collapse', item='transition location', observed='beta_c > 31.75 (seed 0, 1; B-2); fine-grid linear zero 28.6 (window dependent 26-34)', prediction='first-order jump (toy), not observed'))
t2 = pd.DataFrame(rows); save_csv('table2_information.csv', t2)
tex(t2, ['block', 'item', 'observed', 'prediction'], ['block', 'quantity', 'observed', 'prediction'], [s, s, s, s], 'table2_information.tex', 'Information content of the latent and collapse (seeds as stated).', 'tab:information')
print(t1.round(4).to_string()[:3000]); print(t2.to_string())
