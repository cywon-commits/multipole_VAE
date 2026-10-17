# Stage 1a — Gaussian regime (L=16, kappa=10.0)

## J = 2.0  (tau_int ≈ 0.8 sweeps; gap = 10)

| r | C(r) MC | 2G(r) Gaussian | ratio |
|---|---|---|---|
| 0 | 0.11282 | 0.11731 | 0.962 |
| 1 | 0.01296 | 0.01394 | 0.930 |
| 2 | 0.00101 | 0.00170 | 0.597 |
| 3 | 0.00029 | 0.00021 | 1.358 |

- kappa_ind (vMF MLE, J ignored) = 16.67; prediction 1/G(0) = 17.05; true local kappa = 10.0
- SM tied (J, hz): J = 2.0108 ± 0.0254, hz = 9.9330 ± 0.0920, cond = 13.5
- PL tied (J, hz): J = 2.0139, hz = 9.9235
- SM 21 free: cond = 5.26e+03; dkappa/(dJx+dJy) = -6.03 (pred −6); G2_mu/(−√6 dJ_mu) = 1.02, 0.82 (pred 1; noisy when dJ_mu ≈ 0)

## J = 5.0  (tau_int ≈ 0.8 sweeps; gap = 10)

| r | C(r) MC | 2G(r) Gaussian | ratio |
|---|---|---|---|
| 0 | 0.07530 | 0.07680 | 0.980 |
| 1 | 0.01466 | 0.01521 | 0.964 |
| 2 | 0.00276 | 0.00320 | 0.862 |
| 3 | 0.00072 | 0.00071 | 1.018 |

- kappa_ind (vMF MLE, J ignored) = 25.52; prediction 1/G(0) = 26.04; true local kappa = 10.0
- SM tied (J, hz): J = 4.9958 ± 0.0367, hz = 9.9656 ± 0.1186, cond = 11.5
- PL tied (J, hz): J = 4.9988, hz = 9.9575
- SM 21 free: cond = 1.21e+04; dkappa/(dJx+dJy) = -5.74 (pred −6); G2_mu/(−√6 dJ_mu) = 0.99, 0.97 (pred 1; noisy when dJ_mu ≈ 0)

## <|delta|^2> vs kappa (J = 5)

| kappa | MC | lattice Gaussian 2G(0) | disk formula |
|---|---|---|---|
| 1.0 | 0.1504 | 0.1585 | 0.1323 |
| 10.0 | 0.0752 | 0.0768 | 0.0632 |
