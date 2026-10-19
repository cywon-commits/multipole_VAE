"""Stage 2c-A: field-direction-noise bias of the coupled decoder (E1).

Theory (Gaussian order, per transverse component; P = kappa + 4J, a = J/P, r = kappa0/P0):
  dJ/J     = [4 r/(a0 Sigma2) - P0 r^2] * s2_eff
  dkap/kap = -[P0 r^2 + 16/Sigma2]      * s2_eff,      Sigma2 = (1/N) sum_q gamma_q^2 S(q)
  s2_eff   = s2_noise - c/(N kappa0)     (c = regression of the decoded-direction error on the q=0 mode)
  VAE optimum (beta, l=1 isotropic latent): s2* = 3 beta / (2 N kappa_eff), KL* = 1.5 ln(2 N kappa_eff/(9 beta)),
  kappa_eff = kappa0^2/P0 (coupled), kappa_ind (independent).

Modes:
  python stage2c_e1_bias.py --predict [--L 32 --J 5 --kappa 10 --beta 1]      analytic predictions only
  python stage2c_e1_bias.py --synthetic                                        MC check of the bias formulas
  python stage2c_e1_bias.py --vae dump.npz                                     analyse a VAE dump
      dump.npz keys: spins (n,L,L,3) in the frame where the TRUE field is along z (rotate back with R^T),
                     dir_sample (n,3) decoded field direction from a sampled z (same frame),
                     dir_mean (n,3) decoded direction from the posterior mean (same frame),
                     J_vae, kappa_vae (floats, learned values), KL (float, used-channel KL in nat)
"""
from _common import *
import argparse
from scipy.optimize import minimize

def coefficients(L, J0, k0):
    q = 2*np.pi*np.arange(L)/L
    gam = 2*np.cos(q)[:, None] + 2*np.cos(q)[None, :]
    Sq = 1/(k0 + J0*(4 - gam))
    Sig2 = (gam**2*Sq).mean()
    P0 = k0 + 4*J0; a0 = J0/P0; r = 1 - 4*a0
    return dict(Sigma2=Sig2, cJ=4*r/(a0*Sig2) - P0*r**2, cK=-P0*r**2 - 16/Sig2, P0=P0)

def predict(L, J0, k0, beta=1.0, kappa_ind=None, c=1.0):
    N = L*L; co = coefficients(L, J0, k0)
    keff = k0**2/co['P0']
    s2 = 3*beta/(2*N*keff); s2e = s2 - c/(N*k0)
    out = dict(**co, s2_star=s2, KL_star=1.5*np.log(2*N*keff/(9*beta)), s2_eff=s2e,
               dJ=co['cJ']*s2e, dK=co['cK']*s2e)
    if kappa_ind:
        out['KL_star_indep'] = 1.5*np.log(2*N*kappa_ind/(9*beta)); out['s2_star_indep'] = 3*beta/(2*N*kappa_ind)
    return out

def tied_fit(S, nprime, J0, k0):
    nb = local_field(S, Hamiltonian(np.eye(3), np.eye(3), np.zeros(3)))     # sum of neighbours
    def f(p):
        J, k = p
        h = J*nb + k*nprime[:, None, None, :]
        kk = np.linalg.norm(h, axis=-1)
        R = (langevin(kk)/kk)[..., None]*h - S
        loss = (log_A(kk) - (h*S).sum(-1)).sum()/len(S)
        return loss, np.array([(R*nb).sum(), (R*nprime[:, None, None, :]).sum()])/len(S)
    return minimize(f, [J0, k0], jac=True, method='L-BFGS-B').x

def transverse(d):
    """x,y components of directions near z."""
    return d[:, :2]

ap = argparse.ArgumentParser()
ap.add_argument('--predict', action='store_true'); ap.add_argument('--synthetic', action='store_true')
ap.add_argument('--vae'); ap.add_argument('--L', type=int, default=32)
ap.add_argument('--J', type=float, default=5.0); ap.add_argument('--kappa', type=float, default=10.0)
ap.add_argument('--beta', type=float, default=1.0); ap.add_argument('--kappa_ind', type=float, default=25.7)
ap.add_argument('--chains', type=int, default=100); ap.add_argument('--nsamp', type=int, default=10)
a = ap.parse_args()
L, J0, k0 = a.L, a.J, a.kappa

if a.predict or not (a.synthetic or a.vae):
    for c in (1.0, 0.0):
        p = predict(L, J0, k0, a.beta, a.kappa_ind, c)
        print(f"L={L} J={J0} kappa={k0} beta={a.beta} c={c}: Sigma2={p['Sigma2']:.4f} cJ={p['cJ']:.1f} cK={p['cK']:.1f} "
              f"ratio={p['cK']/p['cJ']:.2f} | s2*={p['s2_star']:.2e} KL*={p['KL_star']:.2f} (indep {p['KL_star_indep']:.2f}) | "
              f"pred dJ/J={100*p['dJ']:+.2f}% dk/k={100*p['dK']:+.2f}%")

if a.synthetic:
    H = Hamiltonian.heisenberg_dmi(J0, 0, None, B=(0, 0, k0))
    S, _ = run_mc(H, L, chains=a.chains, therm=100, nsamp=a.nsamp, gap=10, seed=11, init='field')
    co = coefficients(L, J0, k0); N = L*L
    z = np.tile([0, 0, 1.0], (len(S), 1)); base = tied_fit(S, z, J0, k0)
    rng = np.random.default_rng(5)
    M = S.reshape(len(S), -1, 3).sum(1); nM = M/np.linalg.norm(M, axis=1, keepdims=True)
    print(f"true direction: J={base[0]:.4f} kappa={base[1]:.4f}")
    for lab, nd, s2e in [("noise 1e-4", None, 1e-4), ("noise 3e-4", None, 3e-4), ("M_hat (c=1)", nM, -1/(N*k0))]:
        if nd is None:
            e = rng.normal(scale=np.sqrt(s2e), size=(len(S), 2)); nd = z.copy(); nd[:, :2] += e
            nd /= np.linalg.norm(nd, axis=1, keepdims=True)
        J, k = tied_fit(S, nd, J0, k0)
        print(f"{lab:12s}: dJ/J={100*(J/base[0]-1):+.3f}% (pred {100*co['cJ']*s2e:+.3f}%)  "
              f"dk/k={100*(k/base[1]-1):+.3f}% (pred {100*co['cK']*s2e:+.3f}%)")

if a.vae:
    d = np.load(a.vae); S = d['spins'].astype(float); N = L*L
    co = coefficients(L, J0, k0)
    ds, dm = transverse(d['dir_sample']), transverse(d['dir_mean'])
    dbar = S.reshape(len(S), -1, 3).mean(1)[:, :2]
    s2_sample = ((ds - dm)**2).sum(1).mean()/2            # per-component variance of the sampling noise
    c = float((dm*dbar).sum()/(dbar*dbar).sum())
    resid = ((dm - c*dbar)**2).sum(1).mean()/2
    # prediction for the sampled direction: s2_eff = s2_sample + resid + (c^2 - 2c)/(N kappa0)
    s2e = s2_sample + resid + (c*c - 2*c)/(N*k0)
    print(f"formula prediction from measured noise: dJ/J={100*co['cJ']*s2e:+.3f}%  dk/k={100*co['cK']*s2e:+.3f}%")
    z = np.tile([0, 0, 1.0], (len(S), 1)); base = tied_fit(S, z, J0, k0)
    fits = {k_: tied_fit(S, v/np.linalg.norm(v, axis=1, keepdims=True), J0, k0)
            for k_, v in [('sample', d['dir_sample']), ('mean', d['dir_mean'])]}
    p = predict(L, J0, k0, a.beta)
    print(f"measured sampling-noise variance per component s2 = {s2_sample:.2e} (VAE optimum {p['s2_star']:.2e})")
    print(f"regression of posterior-mean error on q=0 mode: c = {c:.3f}; residual variance {resid:.2e}")
    print(f"refit with true direction: J={base[0]:.4f} kappa={base[1]:.4f}")
    for k_, (J, kk) in fits.items():
        print(f"refit with decoded {k_:6s} direction: dJ/J={100*(J/base[0]-1):+.3f}%  dk/k={100*(kk/base[1]-1):+.3f}%")
    if 'J_vae' in d:
        print(f"VAE learned: dJ/J={100*(float(d['J_vae'])/J0-1):+.3f}%  dk/k={100*(float(d['kappa_vae'])/k0-1):+.3f}%;  "
              f"KL={float(d['KL']):.2f} (pred {p['KL_star']:.2f}) -> implied s2 = {np.exp(-2*float(d['KL'])/3)/3:.2e}")
