"""Re-run of the 1D reference chains (reference_1d/) with fixed seeds -> CSVs for Fig. 2a/2c, Fig. 3b, Fig. S4 and Table 1.
The 1D results of the earlier chat were not stored in the repository, so they are regenerated here:
  exact chain (open, N=100, J = 0.5, 2, 5): correlation <S_i.S_{i+r}> vs the Fisher exact L(J)^r; SM/PL estimates with jackknife errors.
  DMI chain (N=100, open, no field, bond direction x): cases A: J=2, D=(0,1,0); B: J=2, D=(1,0,0); C: J=1, D=(0,0.3,0) (definitions of fix-1 request),
      plus a pitch-identity grid (D along z, J=1) and 8 independent MC runs of case A; SM/PL estimates (12 parameters) with jackknife errors.
  external-field chain (N=100, kappa=10, J = 1, 2, 5, 10, 20): transverse correlation vs lattice Gaussian, kappa_ind vs sqrt(kappa(kappa+4J)),
      and replicated 12-parameter fits for the collinear flat direction."""
import sys, time
from common import *
sys.path.insert(0, str(TWOD/'reference_1d'))
import heisenberg_chain_estimators as H1
from dmi_chain_mc import run as run_dmi
from field_chain_mc import run as run_field
from scipy.optimize import brentq

def cond12(S, idx):
    g = H1.site_grads(S)[idx]; t = np.cross(S[None], g); Mm = np.einsum('kmia,lmia->kl', t, t)/S.shape[0]
    d = np.sqrt(np.diag(Mm)); return float(np.linalg.cond(Mm/np.outer(d, d)))
def jack(S, idx, fn, G=20):
    """group-jackknife (groups of chains, i.e. of independent samples) standard error of fn(S) -> vector."""
    M = len(S); gr = np.array_split(np.arange(M), G); est = np.array([fn(S[np.setdiff1d(np.arange(M), g)]) for g in gr])
    return np.sqrt((G - 1)/G*((est - est.mean(0))**2).sum(0))
t0 = time.time(); allp = list(range(H1.P)); sm = lambda idx: (lambda S: H1.score_matching(S, idx))
# ---------------- exact chain ----------------
rows_c, rows_e = [], []
for J in (0.5, 2.0, 5.0):
    rng = np.random.default_rng(10); S = H1.chain(J, 100, 4000, rng); Lg = H1.langevin(J)
    for r in range(0, 9):
        rows_c.append(dict(J=J, r=r, corr_mc=float((S[:, :100-r]*S[:, r:]).sum(-1).mean()), corr_exact=float(Lg**r)))
    for tag, idx in (('J0_only', [0]), ('all12', allp)):
        th = H1.score_matching(S, idx); se = jack(S, idx, sm(idx)); pl = H1.pseudo_lik(S, idx)
        rows_e.append(dict(J=J, fit=tag, SM_J0=float(th[0]), SM_J0_se=float(se[0]), PL_J0=float(pl[0]), true_J0=J, cond=cond12(S, idx), maxabs_z_zero_params=float(np.max(np.abs(th[1:]/se[1:]))) if tag == 'all12' else np.nan))
    print('exact', J, time.time() - t0, flush=True)
save_csv('1d_exact_corr.csv', pd.DataFrame(rows_c)); save_csv('1d_exact_est.csv', pd.DataFrame(rows_e))
# ---------------- DMI chain ----------------
cases = {'A': (2.0, np.array([0, 1.0, 0])), 'B': (2.0, np.array([1.0, 0, 0])), 'C': (1.0, np.array([0, 0.3, 0]))}
rows_d, rows_p = [], []
def pitch(S, D):
    e = D/np.linalg.norm(D); Sp = S - (S@e)[..., None]*e
    num = (np.cross(S[:, :-1], S[:, 1:])@e).mean(); den = (Sp[:, :-1]*Sp[:, 1:]).sum(-1).mean(); return float(num), float(den)
for name, (J, D) in cases.items():
    S = run_dmi(J, D, N=100, chains=200, therm=300, nsamp=4, gap=50, seed=3)
    th = H1.score_matching(S, allp); se = jack(S, allp, sm(allp)); pl = H1.pseudo_lik(S, allp)
    true = np.zeros(H1.P); true[0] = J; true[1:4] = D
    for k, nm in enumerate(H1.names):
        rows_d.append(dict(case=name, param=nm, true=true[k], SM=float(th[k]), SM_se=float(se[k]), PL=float(pl[k])))
    num, den = pitch(S, D); rows_p.append(dict(case=name, J=J, D=float(np.linalg.norm(D)), measured=num/den, expected=float(np.linalg.norm(D)/J), kind='case'))
    rows_p[-1]['cond'] = cond12(S, allp); print('dmi', name, time.time() - t0, flush=True)
for Dm in (0.1, 0.3, 0.6, 1.0, 1.5):                   # pitch identity grid (D along z)
    S = run_dmi(1.0, np.array([0, 0, Dm]), N=100, chains=200, therm=300, nsamp=4, gap=50, seed=4); num, den = pitch(S, np.array([0, 0, Dm]))
    rows_p.append(dict(case=f'grid', J=1.0, D=Dm, measured=num/den, expected=Dm, kind='grid', cond=np.nan))
rows_8 = []
for rep in range(8):                                       # 8 independent MC runs of case A (D_y estimate, pitch identity)
    S = run_dmi(2.0, np.array([0, 1.0, 0]), N=100, chains=200, therm=300, nsamp=4, gap=50, seed=200 + rep); th = H1.score_matching(S, allp); num, den = pitch(S, np.array([0, 1.0, 0]))
    rows_8.append(dict(rep=rep, J0=float(th[0]), D_dir=float(th[2]), pitch=num/den))
save_csv('1d_dmi_caseA_8mc.csv', pd.DataFrame(rows_8))
save_csv('1d_dmi_est.csv', pd.DataFrame(rows_d)); save_csv('1d_dmi_pitch.csv', pd.DataFrame(rows_p))
# ---------------- external-field chain ----------------
kap = 10.0; rows_f, rows_k, rows_fl = [], [], []
for J in (1.0, 2.0, 5.0, 10.0, 20.0):
    S = run_field(J, kap, N=100, C=200, therm=200, nsamp=5, gap=10, seed=5); sz = S[..., 2].mean()
    kind = brentq(lambda k: H1.langevin(k) - sz, 1e-3, 1e5); lam = ((kap + 2*J) - np.sqrt(kap*(kap + 4*J)))/(2*J); G0 = 1/np.sqrt(kap*(kap + 4*J))
    d = S - S[..., 2:3]*np.array([0, 0, 1.0])
    for r in range(0, 9):
        rows_f.append(dict(J=J, r=r, C_mc=float((d[:, :100-r]*d[:, r:]).sum(-1).mean()), C_gauss=float(2*G0*lam**r)))
    d2 = float((d**2).sum(-1).mean())
    rows_k.append(dict(J=J, kappa=kap, x_4J_over_kappa=4*J/kap, kappa_ind_mc=float(kind), kappa_ind_pred=float(1/G0), delta2_mc=d2,
                       kappa_ind_pred_nlo=float((1/G0)*(1 - G0)*2*G0/d2)))
    print('field', J, time.time() - t0, flush=True)
    if J in (5.0, 20.0):                                  # replicated 12-parameter fits: collinear flat direction (J0, G2, hz free + the rest)
        for rep in range(20):
            S = run_field(J, kap, N=100, C=100, therm=100, nsamp=4, gap=10, seed=100 + rep)
            idx = [0, 5, 9, 10, 11]                           # J0, G2, hx, hy, hz (axis = z)
            th = H1.score_matching(S, idx)
            rows_fl.append(dict(J=J, rep=rep, dJ=float(th[0] - J), dG2=float(th[1]), dkappa=float(th[4] - kap)))
save_csv('1d_field_corr.csv', pd.DataFrame(rows_f)); save_csv('1d_kappa_ind.csv', pd.DataFrame(rows_k)); save_csv('1d_flat.csv', pd.DataFrame(rows_fl))
print('done', time.time() - t0)
