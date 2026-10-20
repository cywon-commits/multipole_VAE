# Paper figures and data: 2D coupled-decoder section

Everything here is built by `scripts/run_all.sh` (run from `paper/scripts`). Each `figN.py` creates its CSVs in `data/` from the raw results in
`twod/scripts/results/` (or from the re-runs below), draws the figure from the CSVs only, and writes a caption fragment with numbers computed
from the CSVs into `captions/`; `build_captions.py` assembles `CAPTIONS.md` (caption drafts; the numbers are not typed by hand).
Figures: `fig/figN.pdf` (vector) and `fig/figN.png` (300 dpi); `fig/fig1.svg` is the editable source of the schematic panels (text kept as text).
Format: one column 3.4 in / two columns 7.0 in (all main figures are two-column), DejaVu Sans 8 pt (ticks 7 pt), bold panel labels.
Colours/markers: independent decoder orange squares, coupled PL blue circles, coupled SM green diamonds, theory black solid, approximation/bound black dotted,
truth grey thin; seed 1 = triangle in the colour of its object (Okabe--Ito palette). No symlog axes; KL = 0 is drawn on a floor line with a down-triangle.

## Sources and re-runs per item

| item | raw data / script | re-run? |
|---|---|---|
| Fig. 1a,b | schematic (matplotlib, `fig1.py`) | -- |
| Fig. 1c | `data/fig1c_*.csv` from `gen_textures.py`: Stage 1c (B = 0.4, 1.5, L = 48, 2 chains, seed 0) regenerated; E1 sample from `results/stage2/E1.npz` | **re-run** (no stored samples) |
| Fig. 2a, S4 | `rerun_1d.py` (`reference_1d/`: exact chain N = 100, J = 0.5, 2, 5; DMI chain; field chain) | **re-run** (1D results were not in the repository) |
| Fig. 2b | `results/stage1a/summary.json` | no |
| Fig. 2c | 1D: `data/1d_kappa_ind.csv` (re-run); 2D: Stage 1a summary + analytic `kappa_ind_pred` | 1D re-run |
| Fig. 2d | `results/stage1b/summary.json` | no |
| Fig. 3a | `results/stage1c/summary.json` | no |
| Fig. 3b | 1D: `data/1d_flat.csv` (20 replicated fits for J = 5, 20, re-run); 2D: Stage 1a `flat_ratios`, `cond_free` | 1D re-run |
| Fig. 4a | `infer_latent.py` (inference only, E1 checkpoints) | inference |
| Fig. 4b,d | `results/stage2b/E3_conv_*`, `E3_pl_givenB_*` (+ toy-VAE numbers, PL bound from `stage2b_make_E3.py`) | no |
| Fig. 4c | `infer_latent.py` (inference only, E3 checkpoints) | inference |
| Fig. 4e | `results/stage2/E2_pl_b1.0_jlr50`, `E2_sm_alt_b1.0_s0`, `results/stage2b/E2_ramp_s*` | no |
| Fig. 5a | `make_fig5_data.py`: `stage2c_e1_bias.py --synthetic` (re-run), E1 dumps via `stage2c_dump_e1.py` (re-run), VAE biases from `results/stage2b/E1*` | synthetic + dump re-run |
| Fig. 5b | `results/stage2b/stage2d_A1_profiles.json`, `results/stage2e/B_kplus.json` | no |
| Fig. 5c | `results/stage2b/E2_hm_*`, `stage2d_A1_profiles.json` | no |
| Fig. 5d | `stage2e_valley.py` full run (L = 48), parsed by `make_fig5_data.py` | **re-run** (printed output only) |
| Fig. 6a--c | `results/stage2b/E2_*` (B2s, B2c, h0.01, ramp, A2, F, FC, FB2), `results/stage2f_A_bound.json` | no |
| Fig. 6d | `stage2f_phase_toy.py` logic (`fig6.py`) | recomputed |
| Table 1 (CSV + LaTeX) | `data/table1_*.csv/.tex`: 1D chains (re-run), Stage 1a/1b/1c summaries | 1D re-run |
| Table 2 (CSV + LaTeX) | `data/table2_information.csv/.tex`: `results/stage2b`, `stage2e/C_ieff.json`, `stage2f_A_bound.json` | no |
| S1 | `results/stage2e/A2_prediction_preregistered.json` (written before the A2 runs), `A2_observed.json`, spectra `.npy` | no |
| S2 | `stage2d_A1_profiles.json` | no |
| S3 | `data/fig4c_e3_scatter.csv` (inference) + toy VAE (`stage2d_toy_vae.py`) | inference |
| S5 | `infer_latent.py` (inference only, E2 continued model) | inference |

Caption drafts: `CAPTIONS.md`. Seeds are stated in every caption.

## Definitions I had to choose (not in the repository)

- 1D DMI cases (open chain N = 100, no field, bond direction $\hat x$; `dmi_chain_mc.run` defaults: 200 chains, 300 thermalisation sweeps, 4 samples/chain, gap 50):

  | case | J | D | character |
  |---|---|---|---|
  | A | 2 | (0, 1, 0) | perpendicular to the bond, cycloid |
  | B | 2 | (1, 0, 0) | parallel to the bond, helix |
  | C | 1 | (0, 0.3, 0) | weak perpendicular |

  **Resolved: the former definition (A: J=1, D=0.3 z; B: 0.6 z; C: 0.6(1,1,1)/sqrt3) was my guess and is replaced by this table.**

  Re-run (`rerun_1d.py`, one seed per case, errors = chain-group jackknife; the chat reference used a chain bootstrap), compared with the chat values of the fix brief:

  | case | quantity | this re-run SM (PL) | chat reference |
  |---|---|---|---|
  | A | J | 1.994 ± 0.012 (1.999) | 1.999 ± 0.011 |
  | A | D_y | 1.038 ± 0.017 (1.022) | 0.970 ± 0.008 |
  | A | pitch | 0.510 (exp. 0.5) | 0.491 |
  | B | J | 1.991 ± 0.014 (1.994) | 1.991 ± 0.012 |
  | B | D_x | 1.011 ± 0.014 (1.017) | 1.005 ± 0.012 |
  | B | pitch | 0.506 (exp. 0.5) | 0.498 |
  | C | J | 0.986 ± 0.009 (0.990) | 0.994 ± 0.007 |
  | C | D_y | 0.299 ± 0.009 (0.298) | 0.312 ± 0.007 |
  | C | pitch | 0.301 (exp. 0.3) | 0.315 |

  Case A: the chat value 0.970 was a 3.7σ deviation; this re-run gives 1.038 ± 0.017 (+2.2σ), i.e. a deviation of the opposite sign, and 8 independent MC runs of case A (seeds 200–207, `data/1d_dmi_caseA_8mc.csv`, same score-matching estimator as the table) give D_y = 1.001 ± 0.005 (brief: 0.995 ± 0.003), J = 1.996 ± 0.002, pitch 0.502 ± 0.002. The spread of the 8 independent runs (s.d. 0.013 in D_y) is larger than the quoted single-run errors of the chat run (0.008) and comparable to this run's (0.017), so the 0.970 of the chat run and the 1.038 here are both consistent with a single-run scatter of ~0.013–0.017 around D_y ≈ 1; the 3.7σ was an underestimate of the error. Values are as measured; nothing was tuned. All other entries agree with the chat reference within 1–2σ (C: D_y −1.4σ, pitch 0.301 vs 0.315).
- 1D "cond(M)" in Table 1 is the column-normalised condition number of the 12-parameter score-matching matrix, computed in `rerun_1d.py`.

## Mismatches between raw data and the report / brief (the figures follow the raw data)

1. **[Resolved: Fig. 6b no longer shows the fit line; REPORT item 4 and a footnote on the Stage 2f sentence were rewritten (convex tail, window-dependent zero 26.2–34.2).]** **Fine-grid linear zero (Fig. 6b).** The report (Stage 2f) says the linear extrapolation of the seed-1 fine grid reaches zero at beta ~ 32; the brief repeats "zero ~ 32". A straight-line fit in beta of the four fine-grid states (12.6--25.2) gives **28.6**; the zero is window dependent (26.2 including the beta = 10 state, 34.2 with the last states plus the B-2 state). The curve is slightly convex and the B-2 state still has KL = 3.8 nat at beta = 31.75. The new sentence in the report's synthesis (item 4, "KL approaches zero linearly in beta_c - beta", "convex R--D curve, beta_c = -D'(0+)") is therefore supported only approximately; `beta_c ~ 35--40` is a lower-bound statement (beta_lb = 35.0 / 38.5), not a measured transition point.
2. **1D flat direction (Fig. 3b).** The principal axis of the replicated 1D fits is $(1,-5.1,-2.25)$ ($J=5$) and $(1,-5.3,-2.30)$ ($J=20$) versus the predicted $(1,-6,-\sqrt6)$: 12--15 % (kappa) and 6--8 % (G2) below the prediction. The 2D values agree with prediction within 6 %.
3. **[Resolved: Fig. 4a is now the E1 KL comparison (9.95 / 13.01 nat predictions, 3.06 arrow); the alignment curves are Fig. S6.]** **Old Fig. 4a.** The curves of 1-|cos| of all nine E1 models coincide: the latent direction equals each sample's magnetisation direction (finite-size deviation from the label $\hat n$), so the panel does not discriminate between decoders.
4. **[Resolved: Fig. 5a is now observed vs predicted bias; points using the theory s² are hollow with a grey edge.]** **Fig. 5a.** The beta-scan and L = 64 VAE points use the theory $s^2_{\rm eff}$ on the x axis (no per-sample dumps were made for them); only the three beta = 1, L = 32 seeds use the measured $s^2_{\rm eff}$.
5. **Fig. 5b.** The continued K+M1 solutions (triangles) lie off the valley line ($\Delta D/D=-2.5,-3.1\%$ at $\delta\approx-0.05,0.02$): their masked-region field differs strongly (mean $h_z\approx3.7$), so the single-line relation of the valley is not expected to hold for them (already noted in the Stage 2e report).
6. **Fig. 4b,d.** The paper uses the converged Stage 2d runs (`E3_conv_*`); the Stage 2b values (KL 1.32, 1.26, 1.34, R^2 0.906, 0.893, 0.909) differ slightly. SM-alternating seed 1 was stopped at 80 epochs (not converged) and is the only $l=1$ run.
7. **Fig. 2d.** The $h_z$ estimates at $\beta=2$ have large jackknife errors (up to 10 %); they are consistent with the truth within these errors.
8. The report's Stage 2f figures `stage2f_a.png`, `stage2f_b.png` are superseded by Fig. 6 (kept in `twod/scripts/results/fig/`).

## New mismatches found in this revision

9. **1D ⟨|δ|²⟩ (Fig. 3b).** My MC gives ⟨|δ|²⟩ = 0.111 (J = 5) and 0.066 (J = 20); the brief quotes 0.115 and 0.067. The caption uses my values (CSV column `delta2`). In 2D (Stage 1a C(0)) the values are 0.113, 0.075, 0.031 for J = 2, 5, 20: smaller than 1D at J = 5, 20 but not at J = 2 (0.113 vs 0.111), so "smaller in 2D" holds only for J ≥ 5; the caption says so.
10. **Case A D-component (§2 above):** 1.038 ± 0.017 in this re-run vs 0.970 ± 0.008 (chat), and the pitch 0.510 / 0.506 / 0.301 vs 0.491 / 0.498 / 0.315.
11. **Fig. 5a observed vs predicted.** The three β = 1, L = 32 VAE points lie above the first-order prediction for J (observed +1.49 / +1.49 / +1.49 % vs predicted +1.22 / +1.14 / +1.83 %), i.e. the formula is within ~±25 % for these, not exact.
12. **Fig. 6c:** for seed 0 the β_lb state is the chain state E2_F_b31.8_s0 (ratio 38.46), not the flat start (35.23); for seed 1 it is the B-2 state E2_FB2_b31.8_s1 (34.95).

## Change list of this revision (fix 1)

**Figures:** Fig. 1 (a) widened coupled box, compressed height; (c) E1 coloured by S·n̂, title/colour-bar label changed, "(c)" moved, arrows every 2nd/3rd site. Fig. 3(b) new caption sentence. Fig. 4(a) replaced by the E1 KL comparison; old alignment curves → new Fig. S6; Fig. 4(d) grey-edged marker + star for the unconverged SM s1. Fig. 5(a) → observed vs predicted bias. Fig. 6(b) fit line and "zero" label removed, B-2 state emphasised; Fig. 6(c) β_lb states marked. Fig. S4 and Table 1 rebuilt from the new 1D DMI cases.
**Data:** `1d_dmi_*.csv` (new A, B, C), new `1d_dmi_caseA_8mc.csv`, `fig3b_flat.csv` (+`delta2`), new `fig4a_e1_kl.csv`, `fig5a_obs_vs_pred.csv`, other 1D CSVs regenerated with the same seeds.
**Sentences:** REPORT synthesis item 4 ending; footnote on the Stage 2f "β ≈ 32" sentence; captions of Fig. 1(c), 3(b), 4(a), 4(d), 5(a), 6(b), 6(c), S4, S6.
