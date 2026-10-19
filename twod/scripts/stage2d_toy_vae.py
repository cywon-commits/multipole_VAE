"""Stage 2d-C: 1D toy VAE with a Bayes-optimal decoder (reference values for E3).

Rate     = E_B KL( N(mu(B), s(B)^2) || N(0,1) )   (+ E_B[s^2 - 1 - ln s^2] for an l=1 channel: two unused
           transverse components sharing the scalar log-variance)
Distort. = (SNR/2) E[Var(B | z)],  B standardized (Var B = 1),  SNR = I_dec Var(B)
Objective = beta*Rate + Distortion, minimized over free monotone mu(B) and free log s(B) (piecewise linear, K knots).
Gaussian source, l=0 has the closed form R^2 = 1 - beta/SNR, KL = 0.5 ln(SNR/beta) (collapse if SNR < beta).

usage: python stage2d_toy_vae.py --source uniform --snr 17.33 [--l1] [--beta 1]
E3 numbers: sigma_B^PL = 0.0277, Var B = 0.01333 -> SNR_PL = 17.33; measured I_dec/I_PL = 0.98-0.99.
Reference (SNR = 17.33, beta = 1):  uniform l=0: R2_sample 0.919, R2_mean 0.993, KL 1.38, MI 1.37
                                    uniform l=1: R2_sample 0.855, R2_mean 0.999, KL 1.68, MI 0.96
"""
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def solve(source='uniform', snr=17.33, l1=False, beta=1.0, K=8, nB=81, nz=481):
    if source == 'uniform':
        B = np.linspace(-np.sqrt(3), np.sqrt(3), nB); pB = np.ones(nB)
    else:
        B = np.linspace(-4.5, 4.5, nB); pB = norm.pdf(B)
    pB /= pB.sum()
    z = np.linspace(-6, 6, nz); dz = z[1] - z[0]
    knots = np.linspace(B[0], B[-1], K)
    def unpack(p):
        mk = np.concatenate([[p[K-1]], p[K-1] + np.cumsum(np.exp(p[:K-1]))])
        return np.interp(B, knots, mk), np.exp(np.interp(B, knots, p[K:2*K]))
    def parts(p):
        mu, s = unpack(p)
        joint = pB[:, None]*norm.pdf(z[None, :], mu[:, None], s[:, None])
        pz = joint.sum(0); post = joint/np.maximum(pz, 1e-300)
        Em = (post*B[:, None]).sum(0); Em2 = (post*B[:, None]**2).sum(0)
        D = (pz*(Em2 - Em**2)).sum()*dz
        KL = (pB*0.5*(mu**2 + s**2 - 1 - np.log(s**2))).sum()
        if l1: KL += (pB*(s**2 - 1 - np.log(s**2))).sum()
        return mu, s, D, KL, joint, pz
    obj = lambda p: beta*parts(p)[3] + 0.5*snr*parts(p)[2]
    p0 = np.concatenate([np.full(K-1, np.log(2*np.sqrt(3)/(K-1)*0.8)), [-1.0], np.full(K, np.log(0.3))])
    r = minimize(obj, p0, method='L-BFGS-B', options=dict(maxiter=400))
    mu, s, D, KL, joint, pz = parts(r.x)
    EB = (pB*B).sum(); VB = (pB*(B - EB)**2).sum(); Em = (pB*mu).sum()
    Vm = (pB*(mu - Em)**2).sum(); C = (pB*(B - EB)*(mu - Em)).sum()
    lik = joint/pB[:, None]
    MI = (joint*np.log(np.maximum(lik, 1e-300)/np.maximum(pz[None, :], 1e-300))).sum()*dz
    return dict(R2_sample=C**2/(VB*(Vm + (pB*s**2).sum())), R2_mean=C**2/(VB*Vm), KL=KL, MI=MI, D=D,
                gauss_formula_I=-0.5*np.log(1 - C**2/(VB*(Vm + (pB*s**2).sum()))), s_ratio=s.max()/s.min())

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='uniform'); ap.add_argument('--snr', type=float, default=17.33)
    ap.add_argument('--l1', action='store_true'); ap.add_argument('--beta', type=float, default=1.0)
    a = ap.parse_args()
    o = solve(a.source, a.snr, a.l1, a.beta)
    print(f"{a.source} {'l=1' if a.l1 else 'l=0'} SNR={a.snr} beta={a.beta}: R2_sample={o['R2_sample']:.3f} "
          f"R2_mean={o['R2_mean']:.3f} KL={o['KL']:.3f} MI={o['MI']:.3f} -0.5ln(1-R2)={o['gauss_formula_I']:.3f} "
          f"s_max/s_min={o['s_ratio']:.2f}")
