"""Stage 2c-A: E1 at L=64 (same J=5, kappa=10; field direction randomised exactly by a random R in SO(3)_S).
Output results/stage2/E1L64.npz in the E1 format (spins, field_dir, chain_id)."""
from _common import *
from scipy.spatial.transform import Rotation
a = parse(__doc__); rng = np.random.default_rng(a.seed)
L, J, kap = 64, 5.0, 10.0
H1 = Hamiltonian.heisenberg_dmi(J, 0, None, B=(0, 0, kap))
S, cid = run_mc(H1, L, chains=100, therm=200, nsamp=20, gap=10, seed=a.seed + 3, init='field')
R = Rotation.random(S.shape[0], random_state=a.seed + 3).as_matrix()
S1 = np.einsum('nab,nxyb->nxya', R, S)
np.savez_compressed(os.path.join(a.out, 'stage2', 'E1L64.npz'), spins=S1.astype(np.float32), field_dir=R[:, :, 2], chain_id=cid, J=J, kappa=kap)
print('E1L64', S1.shape)
