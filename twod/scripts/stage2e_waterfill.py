"""Stage 2e-A: reverse water-filling predictions from a per-latent-dimension KL spectrum.

For Gaussian modes with a Bayes decoder, each active latent dimension k obeys
    KL_k(beta) = 1/2 ln(SNR_k / beta)   while SNR_k > beta,   0 otherwise
so one spectrum {KL_k(beta0)} at a single beta0 predicts the whole curve (no free parameter):
    SNR_k = beta0 * exp(2 KL_k(beta0)),     KL(beta) = sum_k [KL_k(beta0) - 1/2 ln(beta/beta0)]_+ ,
    n_active(beta) = #{k : SNR_k > beta},  dKL/dln(beta) = -n_active/2,  beta_c = max_k SNR_k.
Dimensions with KL_k < kl_floor (default 0.02 nat) are treated as inactive.

usage: python stage2e_waterfill.py spectrum.npy --beta0 1 [--betas 1 3 10 30 100 300 1000 3000 10000]
       python stage2e_waterfill.py --check spec_b1.npy spec_b3.npy --beta0 1 --beta1 3
         (checks the per-dimension law KL_k(b1) = KL_k(b0) - 1/2 ln(b1/b0) on matched dimensions)
spectrum.npy: 1D array of per-dimension KL (nat), averaged over validation samples.
"""
import argparse
import numpy as np

def predict(kl, beta0, betas, floor=0.02):
    kl = np.asarray(kl)[np.asarray(kl) > floor]
    snr = beta0*np.exp(2*kl)
    out = []
    for b in betas:
        act = snr > b
        out.append((b, float(np.sum(0.5*np.log(snr[act]/b))), int(act.sum())))
    return snr, out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('spectrum', nargs='?'); ap.add_argument('--beta0', type=float, default=1.0)
    ap.add_argument('--betas', type=float, nargs='*', default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    ap.add_argument('--check', nargs=2); ap.add_argument('--beta1', type=float)
    ap.add_argument('--floor', type=float, default=0.02)
    a = ap.parse_args()
    if a.check:
        k0, k1 = np.load(a.check[0]), np.load(a.check[1])
        act = (k0 > 0.5*np.log(a.beta1/a.beta0) + a.floor)
        pred = k0[act] - 0.5*np.log(a.beta1/a.beta0)
        print(f"dims active at both betas (predicted): {act.sum()}")
        print(f"per-dim KL at beta1: observed mean {k1[act].mean():.3f}, predicted mean {pred.mean():.3f}, "
              f"RMS diff {np.sqrt(np.mean((k1[act]-pred)**2)):.3f}, corr {np.corrcoef(k1[act], pred)[0,1]:.3f}")
        print(f"dims predicted to switch off: {(~act & (k0 > a.floor)).sum()}, of which observed off (<floor): "
              f"{((~act) & (k0 > a.floor) & (k1 < a.floor)).sum()}")
    else:
        kl = np.load(a.spectrum)
        snr, out = predict(kl, a.beta0, a.betas, a.floor)
        print(f"active dims at beta0={a.beta0}: {len(snr)}, total KL {kl[kl > a.floor].sum():.1f}; "
              f"SNR quantiles (10/50/90/max): {np.percentile(snr,10):.1f} {np.median(snr):.1f} {np.percentile(snr,90):.1f} {snr.max():.1f}")
        print("beta        KL_pred   n_active")
        for b, K, n in out:
            print(f"{b:<10g}  {K:8.1f}  {n:6d}")
