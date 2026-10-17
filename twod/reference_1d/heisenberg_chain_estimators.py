import numpy as np
from scipy.optimize import minimize
rng = np.random.default_rng(0)

def sample_vmf(mu, kappa, rng):
    # mu: (M,3) unit axes; returns (M,3) vMF samples with concentration kappa
    M = mu.shape[0]
    u = rng.random(M)
    w = 1 + np.log(u + (1-u)*np.exp(-2*kappa))/kappa          # cos(theta), exact inverse CDF
    phi = 2*np.pi*rng.random(M)
    a = np.where(np.abs(mu[:,[0]])<0.9, [[1,0,0]], [[0,1,0]])
    e1 = np.cross(mu, a); e1 /= np.linalg.norm(e1,axis=1,keepdims=True)
    e2 = np.cross(mu, e1)
    s = np.sqrt(np.clip(1-w**2,0,None))
    return w[:,None]*mu + s[:,None]*(np.cos(phi)[:,None]*e1 + np.sin(phi)[:,None]*e2)

def chain(J, N, M, rng):
    S = np.empty((M,N,3))
    v = rng.normal(size=(M,3)); S[:,0] = v/np.linalg.norm(v,axis=1,keepdims=True)
    for i in range(1,N):
        S[:,i] = sample_vmf(S[:,i-1], J, rng)
    return S

def langevin(k): return 1/np.tanh(k) - 1/k

# ---- features: theta = [J0, Dx,Dy,Dz, G1..G5, hx,hy,hz] ----
E = []  # symmetric traceless basis
E.append(np.diag([1,-1,0])/np.sqrt(2)); E.append(np.diag([1,1,-2])/np.sqrt(6))
for (a,b) in [(0,1),(0,2),(1,2)]:
    m=np.zeros((3,3)); m[a,b]=m[b,a]=1/np.sqrt(2); E.append(m)
eps = np.zeros((3,3,3))
eps[0,1,2]=eps[1,2,0]=eps[2,0,1]=1; eps[0,2,1]=eps[2,1,0]=eps[1,0,2]=-1
# bond matrix for each parameter: S_i^T B S_j
B = [np.eye(3)] + [eps[a] for a in range(3)] + E   # D_a eps_abc S_i^b S_j^c
names = ['J0','Dx','Dy','Dz','G1','G2','G3','G4','G5','hx','hy','hz']
P = len(names)

def site_grads(S):
    # returns g[k] : (M,N,3) ambient gradient of feature k wrt each S_i
    M,N,_ = S.shape
    g = np.zeros((P,M,N,3))
    for k,Bk in enumerate(B):
        g[k,:,:-1] += S[:,1:] @ Bk.T     # d/dS_i of S_i^T B S_{i+1} = B S_{i+1}
        g[k,:,1:]  += S[:,:-1] @ Bk      # d/dS_{i+1} = B^T S_i
    for a in range(3):
        g[9+a,:,:,a] = 1.0
    return g

def features(S):
    f = [np.einsum('mib,bc,mic->m', S[:,:-1], Bk, S[:,1:]) for Bk in B]
    f += [S[:,:,a].sum(1) for a in range(3)]
    return np.array(f)       # (P,M)

def score_matching(S, idx):
    g = site_grads(S)[idx]
    t = np.cross(S[None], g)                       # torque S x g
    Mmat = np.einsum('kmia,lmia->kl', t, t)/S.shape[0]
    f = features(S)[idx].mean(1)
    lap = np.array([4.0 if k<9 else 2.0 for k in idx])   # -Delta Phi = l(l+1) weights
    b = lap*f
    return np.linalg.solve(Mmat, b)

def pseudo_lik(S, idx):
    g = site_grads(S)[idx]                         # (p,M,N,3)
    def loss(th):
        h = np.tensordot(th, g, 1)                 # h_eff (M,N,3)
        k = np.linalg.norm(h,axis=-1)+1e-12
        L = np.log(np.sinh(k)/k) - (h*S).sum(-1)
        Lk = langevin(k)
        grad_h = (Lk/k)[...,None]*h - S
        return L.sum()/S.shape[0], np.einsum('kmia,mia->k', g, grad_h)/S.shape[0]
    th0 = np.zeros(len(idx)); th0[0]=0.5 if idx[0]==0 else 0
    r = minimize(loss, th0, jac=True, method='L-BFGS-B')
    return r.x
