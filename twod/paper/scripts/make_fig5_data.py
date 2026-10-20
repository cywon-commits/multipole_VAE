"""Data for Fig. 5: (a) synthetic field-noise tests (stage2c_e1_bias.py --synthetic, re-run), measured s2_eff of the E1 VAE solutions
(stage2c_dump_e1.py dumps re-generated into a temp dir, 4000 samples), VAE dJ/J, dkappa/kappa from results (beta scan, L=32/64);
(d) the (h_z, J, D) valley (stage2e_valley.py full run, re-run)."""
import re, subprocess, tempfile, os
from common import *
SC = TWOD/'scripts'; env = dict(os.environ, OMP_NUM_THREADS='1')
def run(cmd, **kw): return subprocess.run(cmd, cwd=SC, capture_output=True, text=True, env=env, **kw).stdout
# ---- synthetic ----
out = run(['python', 'stage2c_e1_bias.py', '--synthetic']); rows = []
N, k0 = 1024, 10.0
for line in out.splitlines():
    m = re.match(r'(.+?)\s*: dJ/J=([+-][\d.]+)% \(pred ([+-][\d.]+)%\)\s+dk/k=([+-][\d.]+)% \(pred ([+-][\d.]+)%\)', line)
    if m:
        lab = m.group(1).strip(); s2e = {'noise 1e-4': 1e-4, 'noise 3e-4': 3e-4, 'M_hat (c=1)': -1/(N*k0)}[lab]
        rows.append(dict(label=lab, s2_eff=s2e, dJ_pct=float(m.group(2)), dJ_pred_pct=float(m.group(3)), dK_pct=float(m.group(4)), dK_pred_pct=float(m.group(5))))
save_csv('fig5a_synthetic.csv', pd.DataFrame(rows)); print(rows)
# ---- measured s2_eff of the VAE solutions (beta=1, L=32, 3 seeds) ----
tmp = tempfile.mkdtemp(); vr = []
for s in (0, 1, 2):
    p = f'{tmp}/pl_s{s}.npz'; run(['python', 'stage2c_dump_e1.py', f'E1_pl_s{s}', p, 'all'], cwd=None) if False else subprocess.run(['python', 'stage2c_dump_e1.py', f'E1_pl_s{s}', p, 'all'], cwd=SC, capture_output=True, env=env)
    d = np.load(p); Sx = d['dir_sample'][:, :2]; Mx = d['dir_mean'][:, :2]; dbar = d['spins'].astype(float).reshape(len(Sx), -1, 3).mean(1)[:, :2]
    s2 = ((Sx - Mx)**2).sum(1).mean()/2; c = float((Mx*dbar).sum()/(dbar*dbar).sum()); resid = ((Mx - c*dbar)**2).sum(1).mean()/2
    vr.append(dict(run=f'E1_pl_s{s}', beta=1.0, L=32, s2_noise=s2, c=c, resid=resid, s2_eff=s2 + resid + (c*c - 2*c)/(N*k0)))
vae = pd.DataFrame(vr)
rr = []
for name, beta, L in [(f'E1_pl_s{s}', 1.0, 32) for s in (0, 1, 2)] + [(f'E1beta0.5_pl_s{s}', 0.5, 32) for s in (0, 1)] + [(f'E1beta0.25_pl_s{s}', 0.25, 32) for s in (0, 1)] + [(f'E1L64_pl_s{s}', 1.0, 64) for s in (0, 1)]:
    r = jload(RES/'stage2b'/name/'result.json'); rr.append(dict(run=name, beta=beta, L=L, seed=int(name[-1]), dJ_pct=100*(r['J0']/5 - 1), dK_pct=100*(r['h_norm_mean']/10 - 1), KL=r['kl_total']))
rr = pd.DataFrame(rr); rr = rr.merge(vae[['run', 's2_eff', 'c', 's2_noise']], on='run', how='left'); save_csv('fig5a_vae.csv', rr)
# ---- valley ----
out = run(['python', 'stage2e_valley.py']); vrow = []
for line in out.splitlines():
    m = re.match(r'deficit ([\d.]+): dJ/J per unit ([+-][\d.]+)%\s+dD/D per unit ([+-][\d.]+)%\s+cost \(J,D adapt\) ([\d.]+)\s+cost \(J,D fixed\) ([\d.]+)', line)
    if m: vrow.append(dict(deficit=float(m.group(1)), dJ_per_unit_pct=float(m.group(2)), dD_per_unit_pct=float(m.group(3)), cost_adapt=float(m.group(4)), cost_fixed=float(m.group(5))))
save_csv('fig5d_valley.csv', pd.DataFrame(vrow)); print(vrow)
