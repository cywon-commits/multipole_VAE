"""Stage 1c: interfacial DMI skyrmion lattice (J = D = 1, beta = 6) prepared by annealing.
Checks: (i) topological charge per sample; (ii) recovery of all 26 parameters incl. Gamma and Q (SM only);
(iii) identifiability: cond(M) in the non-collinear SkX vs the collinear field-polarized state (Gamma flat direction)."""
from _common import *

a = parse(__doc__)
L = 24 if a.quick else 48
ch, ns, gap, th = (20, 6, 20, 100) if a.quick else (60, 16, 40, 400)
beta = 6.0; betas = np.linspace(0.5, beta, 12)
md = [f"# Stage 1c — skyrmion lattice (L={L}, J=D=1 interfacial, beta={beta})", ""]; summary = {}
for B in ([0.4, 1.5] if a.quick else [0.2, 0.4, 0.6, 1.5]):
    make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, B)).scaled(b)
    S0 = anneal(make, L, ch, betas, 40 if a.quick else 100, seed=a.seed)
    H = make(beta)
    S, cid = run_mc(H, L, chains=ch, therm=th, nsamp=ns, gap=gap, seed=a.seed, init=S0)
    Q = observables.topological_charge(S)
    sm = score_matching(S, groups=cid)                       # all 26 (Q via score matching only)
    tr = H.theta()
    md += [f"## B = {B}: <Q> = {Q.mean():.2f} (std {Q.std():.2f}), m_z = {S[..., 2].mean():.3f}, cond(M, 26 params) = {sm['cond']:.1f}", ""]
    md += param_rows(tr, sm) + ["", f"- max |z| of zero-valued params (incl. Gamma, Q): {zero_param_maxz(tr, sm):.2f}", ""]
    summary[f"B={B}"] = dict(Q_mean=Q.mean(), Q_std=Q.std(), mz=S[..., 2].mean(), cond=sm['cond'], theta=sm['theta_full'], se=sm['se'], true=tr)
save(a.out, 'stage1c', summary, md)
