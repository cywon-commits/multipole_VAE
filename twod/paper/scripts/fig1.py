"""Fig. 1 (two-column): model overview. (a) VAE schematic, (b) effective field and DMI bond directions, (c) texture examples.
Data: data/fig1c_*.csv (from gen_textures.py). Editable vector source of (a),(b) is saved as fig/fig1.svg (text kept as text)."""
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from s2spin import observables

def load(tag):
    d = pd.read_csv(DATA/f'fig1c_{tag}.csv'); L = int(round(np.sqrt(len(d)))); S = d[['Sx', 'Sy', 'Sz']].values.reshape(L, L, 3); return d, S
fig = plt.figure(figsize=(W2, 6.0)); gs = fig.add_gridspec(2, 4, height_ratios=[0.62, 1.0], hspace=0.12, wspace=0.12, width_ratios=[1.45, 1, 1, 1])
# ---------------- (a) architecture ----------------
axa = fig.add_subplot(gs[0, :]); axa.set_xlim(0, 14); axa.set_ylim(0.0, 4.5); axa.axis('off'); plabel(axa, '(a)', x=0.0, y=0.98)
def box(ax, x, y, w, h, txt, fc='#f2f2f2', ec='k', fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.7)); ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=fs)
def arr(ax, p, q, **kw): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=0.7, color=kw.get('color', 'k'), ls=kw.get('ls', '-')))
box(axa, 0.2, 1.55, 2.1, 1.4, 'spin field\n$\\{S_i\\}$ on a\nperiodic $L{\\times}L$\nlattice')
box(axa, 3.0, 1.55, 2.5, 1.4, 'encoder\n$SO(3)_S$-equivariant,\ntranslation-\nequivariant', fc='#dbe9f6')
box(axa, 6.2, 1.55, 1.9, 1.4, 'latent $z$\n$l{=}0\\oplus l{=}1$')
box(axa, 8.8, 1.55, 2.2, 1.4, 'decoder\n(equivariant)\n$\\to h_i\\ (l{=}1)$', fc='#dbe9f6')
box(axa, 11.8, 3.05, 2.0, 1.15, 'independent:\n$p(S_i|h_i)$ = vMF', fc='#fdebc8', ec=IND['color'])
box(axa, 11.8, 0.2, 2.0, 1.55, 'coupled:\n$p(S_i|S_{\\partial i};h_i,J,D)$\n(pseudo-likelihood)', fc='#cfe3f5', ec=PL['color'], fs=6.2)
box(axa, 8.8, 0.2, 2.2, 0.85, 'global bonds\n$J,\\ D$ (spurion)', fc='#e6f4ea', ec=SM['color'])
arr(axa, (2.3, 2.25), (3.0, 2.25)); arr(axa, (5.5, 2.25), (6.2, 2.25)); arr(axa, (8.1, 2.25), (8.8, 2.25)); arr(axa, (11.0, 2.7), (11.8, 3.5)); arr(axa, (11.0, 1.9), (11.8, 1.3)); arr(axa, (11.0, 0.62), (11.8, 0.7), color=SM['color'])
axa.text(7.0, 4.35, 'loss: pseudo-ELBO = reconstruction + $\\beta\\,$KL$(q(z|S)\\,\\|\\,\\mathcal{N}(0,1))$', ha='center', fontsize=6.8)
axa.text(0.2, 0.15, 'control variables (field, couplings) $\\to$ $z$;\nsymmetry-broken order is fixed by the neighbours, not by $z$', fontsize=6.2, va='bottom')
# ---------------- (b) local field ----------------
axb = fig.add_subplot(gs[1, 0]); axb.set_xlim(-2.5, 2.5); axb.set_ylim(-2.7, 2.8); axb.set_aspect('equal'); axb.axis('off'); plabel(axb, '(b)', x=0.0, y=1.0)
axb.text(0, 2.3, '$h_i^{\\rm eff}=h_i+\\sum_j\\,(J\\,S_j+S_j\\times D_{ij})$', ha='center', fontsize=6.3)
for (cx, ttl, dvec) in ((-1.2, 'interfacial\n$D_x=D\\hat y$\n$D_y=-D\\hat x$', {'x': (0, 1), 'y': (-1, 0)}), (1.2, 'bulk\n$D_\\mu=D\\hat\\mu$', {'x': (1, 0), 'y': (0, 1)})):
    for s, (dx, dy) in ((+1, (1, 0)), (-1, (-1, 0)), (+1, (0, 1)), (-1, (0, -1))): axb.plot([cx, cx + 0.75*dx], [0, 0.75*dy], c='k', lw=0.6)
    axb.add_patch(Circle((cx, 0), 0.12, fc='k')); axb.text(cx + 0.14, -0.28, '$i$', fontsize=6)
    for s, (dx, dy) in ((+1, (1, 0)), (-1, (-1, 0)), (+1, (0, 1)), (-1, (0, -1))): axb.add_patch(Circle((cx + 0.75*dx, 0.75*dy), 0.07, fc='w', ec='k', lw=0.6))
    vx = dvec['x']; vy = dvec['y']; axb.annotate('', xy=(cx + 0.75 + 0.42*vx[0], 0.38*vx[1] - 0.0), xytext=(cx + 0.75, 0), arrowprops=dict(arrowstyle='-|>', color=SM['color'], lw=1.0))
    axb.annotate('', xy=(cx + 0.42*vy[0], 0.75 + 0.42*vy[1]), xytext=(cx, 0.75), arrowprops=dict(arrowstyle='-|>', color=SM['color'], lw=1.0))
    axb.text(cx, -1.05, ttl, ha='center', va='top', fontsize=6.0)
axb.text(0, -2.6, 'arrows: $D_{ij}$ on the $+x$, $+y$ bonds', ha='center', fontsize=5.6)
# ---------------- (c) textures ----------------
labs = [('polarized', 'polarised, $B=1.5$'), ('skx', 'SkX, $B=0.4$'), ('e1', 'E1 sample (random $\\hat n$)')]
caps = []
for k, (tag, ttl) in enumerate(labs):
    ax = fig.add_subplot(gs[1, k + 1]); d, S = load(tag); L = S.shape[0]
    im = ax.imshow(S[..., 2].T, origin='lower', cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    st = 2 if L > 40 else 1; xs = np.arange(0, L, st)
    X, Y = np.meshgrid(xs, xs, indexing='ij'); ax.quiver(X, Y, S[::st, ::st, 0], S[::st, ::st, 1], color='k', scale=L*0.9, width=0.0035, headwidth=3.5, pivot='mid')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=7.5); plabel(ax, '(c)' if k == 0 else '', x=-0.05, y=1.1)
    Q = float(observables.topological_charge(S[None])[0]); caps.append((tag, L, float(S[..., 2].mean()), Q))
cax = fig.add_axes([0.93, 0.12, 0.012, 0.3]); fig.colorbar(im, cax=cax, label='$S_z$'); cax.tick_params(labelsize=6)
savefig(fig, 'fig1', svg=True)
c = {t: (L, m, Q) for t, L, m, Q in caps}
caption('fig1', f"""**Figure 1. Model overview and test systems.** (a) Effective-field VAE: an SO(3)$_S$- and translation-equivariant encoder maps a spin configuration to a latent $z$ with $l=0$ and $l=1$ channels; the decoder returns the $l=1$ local effective field $h_i$. The independent decoder uses only $h_i$ (vMF observation model); the coupled decoder adds global nearest-neighbour bonds $J$ (and $D$, whose direction pattern is a fixed spurion) and is trained with the pseudo-likelihood. (b) Effective field on site $i$ with the Moriya-vector directions of the interfacial ($D_x=D\\hat y$, $D_y=-D\\hat x$) and bulk ($D_\\mu=D\\hat\\mu$) models. (c) Texture examples (colour: $S_z$; arrows: in-plane components). Polarised state ($L={c['polarized'][0]}$, $\\langle S_z\\rangle={c['polarized'][1]:.3f}$, $Q={c['polarized'][2]:.2f}$), skyrmion lattice ($L={c['skx'][0]}$, $\\langle S_z\\rangle={c['skx'][1]:.3f}$, $Q={c['skx'][2]:.1f}$; interfacial DMI $J=D=1$, $\\beta=6$, annealed, seed 0) and one E1 sample ($L={c['e1'][0]}$, isotropic $J=5$, $\\kappa=10$ along a random direction; $Q={c['e1'][2]:.2f}$). Panels (a),(b) are schematic; no error bars.""")
print('fig1 done', c)
