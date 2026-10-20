#!/bin/sh
# Rebuild all paper figures/tables from the raw results (run from paper/scripts). Re-run steps (see ../README.md) come first.
set -e
python rerun_1d.py          # 1D chains -> data/1d_*.csv
python gen_textures.py      # spin textures -> data/fig1c_*.csv
python infer_latent.py      # inference-only per-sample values -> data/fig4a/4c, figS5
python make_fig5_data.py    # synthetic bias tests, VAE dumps, valley -> data/fig5*.csv
for f in fig1 fig2 fig3 fig4 fig5 fig6 tables figS; do python $f.py; done
python build_captions.py
