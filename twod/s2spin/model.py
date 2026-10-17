"""Translation-invariant square-lattice spin model in log-weight form (beta absorbed):

    W(S) = sum_r [ h.S_r + S_r^T Q S_r ] + sum_r sum_{mu=x,y} S_r^T J_mu S_{r+mu}

J_mu = J0_mu I + [D_mu]_x + Gamma_mu, with S^T [D]_x S' = D.(S x S').  Positive J0 = ferromagnetic.
Lattice axes: array axis -3 = x, axis -2 = y; periodic boundaries.
DMI conventions:  interfacial (C_nv, Neel):  D_mu = D (z_hat x mu_hat)  ->  D_x = D y_hat, D_y = -D x_hat
                  bulk (chiral, Bloch):       D_mu = D mu_hat
"""
from dataclasses import dataclass, field
import numpy as np

EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1

# orthonormal (Frobenius) basis of symmetric traceless 3x3 matrices (l = 2)
_G = [np.diag([1., -1, 0])/np.sqrt(2), np.diag([1., 1, -2])/np.sqrt(6)]
for a, b in [(0, 1), (0, 2), (1, 2)]:
    m = np.zeros((3, 3)); m[a, b] = m[b, a] = 1/np.sqrt(2); _G.append(m)
GAMMA_BASIS = np.array(_G)

def dmi_matrix(D):
    """A with S^T A S' = D.(S x S')  (A_bc = eps_abc D_a)."""
    return np.einsum('abc,a->bc', EPS, np.asarray(D, float))

def decompose(Jmat):
    """3x3 bond tensor -> (J0, D[3], G[5])  (l = 0 + 1 + 2)."""
    J0 = np.trace(Jmat)/3
    D = 0.5*np.einsum('abc,bc->a', EPS, Jmat)
    sym = 0.5*(Jmat + Jmat.T) - J0*np.eye(3)
    G = np.einsum('kbc,bc->k', GAMMA_BASIS, sym)
    return J0, D, G

@dataclass
class Hamiltonian:
    Jx: np.ndarray
    Jy: np.ndarray
    h: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Q: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))   # single-site l=2 (MC requires Q=0)

    @classmethod
    def heisenberg_dmi(cls, J=1.0, D=0.0, kind=None, B=(0, 0, 0)):
        if kind is None or D == 0:
            Dx = Dy = np.zeros(3)
        elif kind == 'interfacial':
            Dx, Dy = D*np.array([0, 1., 0]), D*np.array([-1., 0, 0])
        elif kind == 'bulk':
            Dx, Dy = D*np.array([1., 0, 0]), D*np.array([0, 1., 0])
        else:
            raise ValueError(kind)
        return cls(J*np.eye(3) + dmi_matrix(Dx), J*np.eye(3) + dmi_matrix(Dy), np.asarray(B, float))

    def scaled(self, beta):
        """Same model at inverse temperature beta (all couplings multiplied)."""
        return Hamiltonian(beta*self.Jx, beta*self.Jy, beta*self.h, beta*self.Q)

    def theta(self):
        """Parameter vector in estimators.NAMES order (26 entries)."""
        out = []
        for Jm in (self.Jx, self.Jy):
            J0, D, G = decompose(Jm); out += [J0, *D, *G]
        out += list(self.h)
        out += list(np.einsum('kbc,bc->k', GAMMA_BASIS, self.Q))
        return np.array(out)
