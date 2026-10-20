"""Shared style, palette, paths and helpers for the paper figures. Every figN.py builds CSVs in paper/data from the raw results
in twod/scripts/results (or from re-runs listed in paper/README.md), draws the figure from the CSVs only, and writes a caption
fragment with numbers computed from the CSVs to paper/captions/figN.md (assembled into CAPTIONS.md by build_captions.py)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]; TWOD = PAPER.parent
RES = TWOD/'scripts'/'results'; DATA = PAPER/'data'; FIG = PAPER/'fig'; CAPD = PAPER/'captions'
for d in (DATA, FIG, CAPD): d.mkdir(exist_ok=True)
sys.path.insert(0, str(TWOD))
W1, W2 = 3.4, 7.0

def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'lines.markersize': 4,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 100,
        'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
style()

# colour-blind-safe (Okabe-Ito) palette; same object = same colour and marker in every figure
IND = dict(color='#E69F00', marker='s')      # independent decoder
PL = dict(color='#0072B2', marker='o')       # coupled decoder, pseudo-likelihood
SM = dict(color='#009E73', marker='D')       # coupled decoder, score-matching alternating
PRED = dict(color='k', ls='-', lw=1.0)       # analytic prediction
APPROX = dict(color='k', ls=':', lw=1.0)     # approximation / bound
TRUTH = dict(color='#888888', lw=0.6)        # true value, reference line
EXTRA = ['#D55E00', '#56B4E9', '#CC79A7', '#F0E442', '#000000']
def seed_marker(kind, seed):
    """Seed 0: the decoder's own marker, seed 1: triangle (same colour)."""
    return kind['marker'] if seed == 0 else '^'

def plabel(ax, s, x=-0.18, y=1.05):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')

def savefig(fig, name, svg=False):
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})) + ((('svg', {}),) if svg else ()):
        fig.savefig(FIG/f'{name}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)

def save_csv(name, df):
    p = DATA/name; df.to_csv(p, index=False); return p

def caption(name, text):
    (CAPD/f'{name}.md').write_text(text.strip() + '\n')

def jload(path):
    import json
    return json.load(open(path))
