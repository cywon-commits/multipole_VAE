"""Stage 1b: Heisenberg + DMI (interfacial / bulk / none) in the disordered-to-helical regime.
Checks: (i) recovery of all 21 parameters (SM with jackknife errors, PL); (ii) Moriya directions per bond;
(iii) null components (Gamma, D when D=0, transverse h) consistent with zero; (iv) linear scaling theta ∝ beta."""
from _common import *

a = parse(__doc__)
L = 12 if a.quick else 32
ch, ns, gap, th = (40, 6, 10, 100) if a.quick else (100, 16, 20, 400)
cases = [(None, 0.0), ('interfacial', 0.6), ('bulk', 0.6)] if a.quick else \
        [(None, 0.0), ('interfacial', 0.3), ('interfacial', 0.6), ('bulk', 0.3), ('bulk', 0.6)]
betas = [1.0] if a.quick else [0.5, 1.0, 2.0]
md = [f"# Stage 1b — DMI recovery (L={L}, J=1, B=0.4 z)", ""]; summary = {}
for kind, D in cases:
    for beta in betas:
        H = Hamiltonian.heisenberg_dmi(1.0, D, kind, B=(0, 0, 0.4)).scaled(beta)
        tau = tau_energy(H, L, chains=8, sweeps=100 if a.quick else 400, seed=a.seed)
        S, cid = run_mc(H, L, chains=ch, therm=th, nsamp=ns, gap=gap, seed=a.seed)
        sm = score_matching(S, list(range(21)), groups=cid); pl = pseudo_likelihood(S)
        tr = H.theta(); t = sm['theta_full']
        key = f"{kind or 'none'} D={D} beta={beta}"
        md += [f"## {key}  (tau_int ≈ {tau:.1f}, cond(M) = {sm['cond']:.1f})", ""] + param_rows(tr, sm)
        line = f"- max |z| of zero-valued params: {zero_param_maxz(tr, sm):.2f};  max |SM − PL| = {np.abs(t[:21] - pl['theta_full'][:21]).max():.4f}"
        ang = {}
        if kind:
            for mu, sl in (('x', slice(1, 4)), ('y', slice(10, 13))):
                e = analytic.moriya_expected(kind)[mu]; d = t[sl]
                ang[mu] = float(np.degrees(np.arccos(np.clip(d @ e/np.linalg.norm(d), -1, 1))))
            line += f";  Moriya angle error x/y = {ang['x']:.2f}° / {ang['y']:.2f}°"
        md += ["", line, ""]
        summary[key] = dict(theta=t, se=sm['se'], true=tr, pl=pl['theta_full'], cond=sm['cond'], tau=tau, moriya_deg=ang)
if len(betas) > 1:
    md += ["## beta scaling (theta/beta should be constant)", "", "| case | " + " | ".join(f"J0_x/β (β={b})" for b in betas) + " |",
           "|---|" + "---|"*len(betas)]
    for kind, D in cases:
        md.append(f"| {kind or 'none'} D={D} | " + " | ".join(f"{summary[(kind or 'none') + f' D={D} beta={b}']['theta'][0]/b:.4f}" for b in betas) + " |")
save(a.out, 'stage1b', summary, md)
