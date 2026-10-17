# Stage 1a — Gaussian regime (L=32, kappa=10.0)

## J = 2.0  (tau_int ≈ 1.2 sweeps; gap = 20)

| r | C(r) MC | 2G(r) Gaussian | ratio |
|---|---|---|---|
| 0 | 0.11258 | 0.11731 | 0.960 |
| 1 | 0.01293 | 0.01394 | 0.928 |
| 2 | 0.00158 | 0.00170 | 0.929 |
| 3 | 0.00020 | 0.00021 | 0.925 |

- kappa_ind (vMF MLE, J ignored) = 16.70; prediction 1/G(0) = 17.05; true local kappa = 10.0
- SM tied (J, hz): J = 1.9978 ± 0.0056, hz = 10.0075 ± 0.0199, cond = 13.5
- PL tied (J, hz): J = 1.9978, hz = 10.0066
- SM 21 free: cond = 5.18e+03; dkappa/(dJx+dJy) = -6.26 (pred −6); G2_mu/(−√6 dJ_mu) = 1.07, 1.11 (pred 1; noisy when dJ_mu ≈ 0)

## J = 5.0  (tau_int ≈ 1.2 sweeps; gap = 20)

| r | C(r) MC | 2G(r) Gaussian | ratio |
|---|---|---|---|
| 0 | 0.07511 | 0.07680 | 0.978 |
| 1 | 0.01466 | 0.01521 | 0.964 |
| 2 | 0.00307 | 0.00320 | 0.957 |
| 3 | 0.00068 | 0.00071 | 0.952 |

- kappa_ind (vMF MLE, J ignored) = 25.59; prediction 1/G(0) = 26.04; true local kappa = 10.0
- SM tied (J, hz): J = 4.9981 ± 0.0101, hz = 10.0047 ± 0.0293, cond = 11.6
- PL tied (J, hz): J = 4.9981, hz = 10.0042
- SM 21 free: cond = 1.22e+04; dkappa/(dJx+dJy) = -5.93 (pred −6); G2_mu/(−√6 dJ_mu) = 0.89, 1.07 (pred 1; noisy when dJ_mu ≈ 0)

## J = 20.0  (tau_int ≈ 1.3 sweeps; gap = 20)

| r | C(r) MC | 2G(r) Gaussian | ratio |
|---|---|---|---|
| 0 | 0.03137 | 0.03162 | 0.992 |
| 1 | 0.01046 | 0.01058 | 0.989 |
| 2 | 0.00394 | 0.00400 | 0.986 |
| 3 | 0.00160 | 0.00164 | 0.974 |

- kappa_ind (vMF MLE, J ignored) = 62.75; prediction 1/G(0) = 63.24; true local kappa = 10.0
- SM tied (J, hz): J = 19.9856 ± 0.0230, hz = 10.0187 ± 0.0474, cond = 8.5
- PL tied (J, hz): J = 19.9854, hz = 10.0193
- SM 21 free: cond = 6.47e+04; dkappa/(dJx+dJy) = -6.03 (pred −6); G2_mu/(−√6 dJ_mu) = 1.00, 1.05 (pred 1; noisy when dJ_mu ≈ 0)

## <|delta|^2> vs kappa (J = 5)

| kappa | MC | lattice Gaussian 2G(0) | disk formula |
|---|---|---|---|
| 0.5 | 0.1721 | 0.1817 | 0.1541 |
| 1.0 | 0.1515 | 0.1584 | 0.1323 |
| 2.0 | 0.1297 | 0.1344 | 0.1107 |
| 5.0 | 0.0990 | 0.1016 | 0.0830 |
| 10.0 | 0.0752 | 0.0768 | 0.0632 |
