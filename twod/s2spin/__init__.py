"""s2spin: S^2 spin-field toolkit for effective-field / coupled-decoder VAE tests (2D square lattice)."""
from .vmf import langevin, log_A, sample_vmf, random_unit
from .model import Hamiltonian, EPS, GAMMA_BASIS, dmi_matrix, decompose
from .mc import local_field, sweep, run_mc
from .estimators import NAMES, P, score_matching, pseudo_likelihood, pl_loss_grad, theta_to_hamiltonian, tie_matrix
from . import observables, analytic
