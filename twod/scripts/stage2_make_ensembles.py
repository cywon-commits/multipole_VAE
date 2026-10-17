"""Stage 2 data: ensembles with a known generating factor for the VAE experiments (3b).

E1 'field-direction': isotropic J, field kappa along a uniformly random direction n per sample.
   Generated exactly by symmetry: sample with n = z, then apply a random R in SO(3)_S per sample
   (isotropic exchange is SO(3)_S invariant, so R S is an exact sample for field R z). Label: n.
E2 'skx': interfacial-DMI skyrmion lattice; random lattice translation + joint C4 (90° spatial rotation
   combined with the matching 90° spin rotation about z, a symmetry of the C4v interfacial model).
   Labels: shift (dx, dy), c4 index, topological charge Q. The joint-symmetry claim is asserted numerically.
Output: <out>/stage2/E1.npz, E2.npz with spins (n, L, L, 3) float32, labels, and the true Hamiltonian theta."""
from _common import *
from scipy.spatial.transform import Rotation

a = parse(__doc__)
rng = np.random.default_rng(a.seed)
L1, L2 = (16, 24) if a.quick else (32, 48)
d = os.path.join(a.out, 'stage2'); os.makedirs(d, exist_ok=True)

# ---- E1 ----
J, kap = 5.0, 10.0
H1 = Hamiltonian.heisenberg_dmi(J, 0, None, B=(0, 0, kap))
S, cid = run_mc(H1, L1, chains=50 if a.quick else 200, therm=200, nsamp=4 if a.quick else 20, gap=10, seed=a.seed, init='field')
R = Rotation.random(S.shape[0], random_state=a.seed).as_matrix()
S1 = np.einsum('nab,nxyb->nxya', R, S)
n_lab = R[:, :, 2]
# check: the rotated samples obey the rotated model (log-weight identity)
Hr = Hamiltonian(H1.Jx, H1.Jy, kap*n_lab[0])
assert np.isclose(log_weight(S1[:1], Hr)[0], log_weight(S[:1], H1)[0])
np.savez_compressed(os.path.join(d, 'E1.npz'), spins=S1.astype(np.float32), field_dir=n_lab, chain_id=cid,
                    theta_z=H1.theta(), J=J, kappa=kap)
print(f"E1: {S1.shape}, J={J}, kappa={kap}")

# ---- E2 ----
make = lambda b: Hamiltonian.heisenberg_dmi(1.0, 1.0, 'interfacial', B=(0, 0, 0.4)).scaled(b)
H2 = make(6.0)
ch = 20 if a.quick else 60
S0 = anneal(make, L2, ch, np.linspace(0.5, 6.0, 12), 40 if a.quick else 100, seed=a.seed)
S, cid = run_mc(H2, L2, chains=ch, therm=100, nsamp=4 if a.quick else 16, gap=40, seed=a.seed, init=S0)

def c4(S, k):
    """Joint 90°·k rotation: spatial rot90 on (x, y) + spin rotation by 90°·k about z."""
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    out = np.rot90(S, k=k, axes=(-3, -2))
    return out @ np.linalg.matrix_power(Rz, k).T

w0 = log_weight(S[:2], H2)
for k in range(4):
    assert np.allclose(log_weight(c4(S[:2], k), H2), w0), "joint C4 is not a symmetry with this convention"
shift = rng.integers(0, L2, size=(S.shape[0], 2)); kc4 = rng.integers(0, 4, size=S.shape[0])
S2 = np.stack([c4(np.roll(S[i], tuple(shift[i]), axis=(0, 1)), kc4[i]) for i in range(S.shape[0])])
Qc = observables.topological_charge(S2)
np.savez_compressed(os.path.join(d, 'E2.npz'), spins=S2.astype(np.float32), shift=shift, c4=kc4, Q=Qc, chain_id=cid,
                    theta=H2.theta())
print(f"E2: {S2.shape}, <Q> = {Qc.mean():.2f}")
