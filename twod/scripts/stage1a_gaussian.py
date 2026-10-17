"""Stage 1a: isotropic Heisenberg + field (collinear, large kappa).
Checks: (i) transverse correlations vs lattice Gaussian 2G(r); (ii) independent-decoder kappa_ind vs 1/G(0);
(iii) SM/PL recovery of (J, kappa) with x/y tied; (iv) collinear flat direction when Gamma is free;
(v) Mermin-Wagner trend of <|delta|^2> vs kappa."""
from _common import *
from scipy.optimize import brentq

a = parse(__doc__)
L = 16 if a.quick else 32
ch, ns, gap, th = (60, 8, 10, 100) if a.quick else (100, 20, 20, 400)
kap = 10.0
md = [f"# Stage 1a — Gaussian regime (L={L}, kappa={kap})", ""]
summary = {}
for J in ([2.0, 5.0] if a.quick else [2.0, 5.0, 20.0]):
    H = Hamiltonian.heisenberg_dmi(J, 0, None, B=(0, 0, kap))
    tau = tau_energy(H, L, chains=8, sweeps=100 if a.quick else 400, seed=a.seed, init=np.broadcast_to([0, 0, 1.0], (8, L, L, 3)).copy())
    S, cid = run_mc(H, L, chains=ch, therm=th, nsamp=ns, gap=gap, seed=a.seed, init='field')
    C = observables.transverse_corr(S, [0, 0, 1], 6); Cp = analytic.transverse_corr_pred(L, kap, J, 6)
    cb = S[..., 2].mean(); k_ind = brentq(lambda k: langevin(k) - cb, 1e-3, 1e5)
    iso = [0, 9, 20]; T = tie_matrix(iso, [['J0_x', 'J0_y']])
    sm = score_matching(S, iso, T=T, groups=cid); pl = pseudo_likelihood(S, iso, T=T)
    fr = score_matching(S, list(range(21)), groups=cid); t = fr['theta_full']
    ratio = [(t[20] - kap)/((t[0] - J) + (t[9] - J)), t[5]/(-np.sqrt(6)*(t[0] - J)), t[14]/(-np.sqrt(6)*(t[9] - J))]
    md += [f"## J = {J}  (tau_int ≈ {tau:.1f} sweeps; gap = {gap})", "",
           "| r | C(r) MC | 2G(r) Gaussian | ratio |", "|---|---|---|---|"]
    md += [f"| {r} | {C[r]:.5f} | {Cp[r]:.5f} | {C[r]/Cp[r]:.3f} |" for r in range(4)]
    md += ["", f"- kappa_ind (vMF MLE, J ignored) = {k_ind:.2f}; prediction 1/G(0) = {analytic.kappa_ind_pred(L, kap, J):.2f}; true local kappa = {kap}",
           f"- SM tied (J, hz): J = {sm['theta'][0]:.4f} ± {sm['se'][0]:.4f}, hz = {sm['theta'][2]:.4f} ± {sm['se'][2]:.4f}, cond = {sm['cond']:.1f}",
           f"- PL tied (J, hz): J = {pl['theta'][0]:.4f}, hz = {pl['theta'][2]:.4f}",
           f"- SM 21 free: cond = {fr['cond']:.2e}; dkappa/(dJx+dJy) = {ratio[0]:.2f} (pred −6); G2_mu/(−√6 dJ_mu) = {ratio[1]:.2f}, {ratio[2]:.2f} (pred 1; noisy when dJ_mu ≈ 0)", ""]
    summary[f"J={J}"] = dict(tau=tau, C=C, Cpred=Cp, kappa_ind=k_ind, kappa_ind_pred=analytic.kappa_ind_pred(L, kap, J),
                             sm_tied=sm['theta'], sm_tied_se=sm['se'], pl_tied=pl['theta'], cond_free=fr['cond'], flat_ratios=ratio)
# Mermin-Wagner trend
J = 5.0; md += ["## <|delta|^2> vs kappa (J = 5)", "", "| kappa | MC | lattice Gaussian 2G(0) | disk formula |", "|---|---|---|---|"]
mw = {}
for k in ([1.0, 10.0] if a.quick else [0.5, 1.0, 2.0, 5.0, 10.0]):
    H = Hamiltonian.heisenberg_dmi(J, 0, None, B=(0, 0, k))
    S, _ = run_mc(H, L, chains=ch//2, therm=th, nsamp=ns//2, gap=gap, seed=a.seed + 1, init='field')
    d2 = (1 - S[..., 2]**2).mean()
    p = 2*analytic.gaussian_G(L, k, J)[0, 0]
    md.append(f"| {k} | {d2:.4f} | {p:.4f} | {analytic.delta2_disk(k, J):.4f} |"); mw[k] = [d2, p]
summary['mermin_wagner'] = mw
save(a.out, 'stage1a', summary, md)
