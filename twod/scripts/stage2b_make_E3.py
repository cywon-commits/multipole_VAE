"""Stage 2b data E3: control-parameter mixture. Interfacial-DMI SkX model (J = D = 1, beta = 6) with the
field strength B drawn uniformly from [B_min, B_max] PER CHAIN (each chain its own Gibbs measure).
Prediction to test: a coupled decoder's l=0 latent encodes B; KL of the used channel >= MI ≈ log(range/sigma_B) - log sqrt(2 pi e), with the
pseudo-likelihood Cramér-Rao width sigma_B = [beta^2 * sum_i Var_cond(S_i.z)]^(-1/2), computed here per sample.
Output: <out>/stage2/E3.npz with spins, B (physical units, beta = 6 not applied), Q, chain_id, sigma_B, theta at B=0."""
from _common import *
from s2spin import probes

a = parse(__doc__)
L = 24 if a.quick else 48
ch = 24 if a.quick else 120
Bmin, Bmax, beta = 0.2, 0.6, 6.0
rng = np.random.default_rng(a.seed)
Bc = rng.uniform(Bmin, Bmax, ch)
hfield = np.zeros((ch, 1, 1, 3)); hfield[..., 2] = Bc[:, None, None]
make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=hfield).scaled(b)   # per-chain field
S0 = anneal(make, L, ch, np.linspace(0.5, beta, 12), 40 if a.quick else 100, seed=a.seed)
H = make(beta)
S, cid = run_mc(H, L, chains=ch, therm=100, nsamp=4 if a.quick else 16, gap=40, seed=a.seed, init=S0)
B = Bc[cid]
Q = observables.topological_charge(S)
# PL Cramér-Rao width for B (field enters W as beta*B*S_z): I = beta^2 sum_i Var_cond(S_i.z)
Hs = Hamiltonian(H.Jx, H.Jy, np.zeros(3))
heff = local_field(S, Hs) + beta*hfield[cid]
I = beta**2*probes.pl_fisher_field(S, heff)
sigma_B = 1/np.sqrt(I)
d = os.path.join(a.out, 'stage2'); os.makedirs(d, exist_ok=True)
np.savez_compressed(os.path.join(d, 'E3.npz'), spins=S.astype(np.float32), B=B, Q=Q, chain_id=cid,
                    sigma_B=sigma_B, beta=beta, theta_B0=Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial').scaled(beta).theta())
corr = np.corrcoef(B, Q)[0, 1]
print(f"E3: {S.shape}, B in [{Bmin},{Bmax}], <Q> = {Q.mean():.2f}, corr(B,Q) = {corr:.2f}, "
      f"median sigma_B = {np.median(sigma_B):.4f}; MI(B; X) under PL ~ log(range/sigma_B) - log sqrt(2 pi e) = "
      f"{np.log((Bmax-Bmin)/np.median(sigma_B)) - 0.5*np.log(2*np.pi*np.e):.2f} nat; "
      f"best linear R^2(B) ~ 1 - sigma_B^2/Var(B) = {1 - np.median(sigma_B)**2/((Bmax-Bmin)**2/12):.3f}")
