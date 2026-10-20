"""Stage 2f: periodic-phase rate-distortion toy. One phase mode, prior uniform over a period, bounded gain A.
Outputs binodal/spinodal beta/SNR (0.092 / 0.112), rate just before collapse (~0.95 nat), and dKL/dln(beta) near collapse."""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
# one periodic phase mode: prior uniform over period p, posterior var v; x = q^2 v, q = 2pi/p, xmax = q^2 p^2/12 = pi^2/3
# objective/A: g(x) = 1 - e^{-x} + b ln(xmax/x),  b = beta/(2A); collapse value g_c = 1 - e^{-xmax}
xm = np.pi**2/3; gc = 1-np.exp(-xm)
def interior(b):
    r = minimize_scalar(lambda lx: 1-np.exp(-np.exp(lx)) + b*np.log(xm/np.exp(lx)), bounds=(-12, np.log(1.0)), method='bounded')
    x = np.exp(r.x); return x, r.fun
# Gaussian-limit SNR = I*V with I = 2 A q^2  -> SNR = 2A*xm ; beta/SNR = b/xm
bs = np.linspace(1e-3, 0.4, 4000)
bin_ = None
for b in bs:
    x, f = interior(b)
    if f > gc and bin_ is None: bin_ = b
print(f"spinodal b=1/e={1/np.e:.3f} -> beta/SNR={1/np.e/xm:.4f};  binodal b={bin_:.4f} -> beta/SNR={bin_/xm:.4f}  (SNR/beta_c={xm/bin_:.1f})")
x,_ = interior(bin_); print(f"rate just before collapse: {0.5*np.log(xm/x):.3f} nat")
# KL vs ln beta slope compared with Gaussian 1/2
for b in [0.02,0.05,0.1,0.15,bin_*0.98]:
    x1,_=interior(b); x2,_=interior(b*1.05)
    s = (0.5*np.log(xm/x2)-0.5*np.log(xm/x1))/np.log(1.05)
    print(f"beta/SNR={b/xm:.4f}: KL={0.5*np.log(xm/x1):.3f}  dKL/dlnbeta={s:.3f}")
