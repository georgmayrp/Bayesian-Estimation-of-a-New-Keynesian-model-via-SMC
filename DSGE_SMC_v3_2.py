#%%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm as norm_dist
from scipy.special import loggamma
from joblib import Parallel, delayed

np.set_printoptions(precision=4, suppress=True)

# -----------------------------
# 1) Load data
# -----------------------------
if not os.path.exists("us.txt"):
    raise FileNotFoundError("Missing us.txt in the current directory.")
data = np.loadtxt("us.txt")
if data.ndim != 2 or data.shape[1] != 3:
    raise ValueError(f"us.txt must be T x 3. Got {data.shape}.")
Tobs = data.shape[0]
print(f"Loaded data: us.txt with T={Tobs}, l=3 observables.")

# -----------------------------
# 2) Embedded priors + bounds
# -----------------------------
param_names = [
    "tau","kappa","psi1","psi2","rA","piA","gammaQ","rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"
]
prior = np.array([
    [2, 2.0, 0.5, 0, 2.83],  # tau
    [5, 0.0, 1.0, 0, 0.78],  # kappa
    [2, 1.5, 0.25,0, 1.80],  # psi1
    [2, 0.5, 0.25,0, 0.63],  # psi2
    [2, 0.5, 0.5, 0, 0.42],  # rA
    [2, 7.0, 2.0, 0, 3.30],  # piA
    [3, 0.4, 0.2, 0, 0.52],  # gammaQ
    [5, 0.0, 1.0, 0, 0.77],  # rho_R
    [5, 0.0, 1.0, 0, 0.98],  # rho_g
    [5, 0.0, 1.0, 0, 0.88],  # rho_z
    [4, 0.4, 4.0, 0, 0.22],  # sigma_R
    [4, 1.0, 4.0, 0, 0.71],  # sigma_g
    [4, 0.5, 4.0, 0, 0.31],  # sigma_z
], dtype=float)
bounds = np.array([
    [1e-20, 1e5],       # tau
    [1e-20, 1.0],       # kappa
    [1e-20, 1e5],       # psi1
    [1e-20, 1e5],       # psi2
    [1e-20, 1e5],       # rA
    [1e-20, 1e5],       # piA
    [1e-20, 1e5],       # gammaQ
    [1e-20, 0.9999999], # rho_R
    [1e-20, 0.9999999], # rho_g
    [1e-20, 0.9999999], # rho_z
    [1e-20, 1e5],       # sigma_R
    [1e-20, 1e5],       # sigma_g
    [1e-20, 1e5],       # sigma_z
], dtype=float)

pshape, p1, p2, pmask, pfix = prior.T
npara = len(param_names)

# -----------------------------
# 3) functions needed in numerics
# -----------------------------
#def nearest_spd(A):
#    B = (A + A.T) / 2
#    U, s, Vt = np.linalg.svd(B)
#    H = (Vt.T * s) @ Vt
#    Ahat = (B + H) / 2
#    Ahat = (Ahat + Ahat.T) / 2
#    k = 0
#    while True:
#        try:
#            np.linalg.cholesky(Ahat)
#            return Ahat
#        except np.linalg.LinAlgError:
#            k += 1
#            mineig = np.min(np.real(np.linalg.eigvals(Ahat)))
#            Ahat = Ahat + (-mineig * (k**2) + np.finfo(float).eps) * np.eye(Ahat.shape[0])

def nearest_spd(A, max_tries=50, fallback_scale=1e2):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("nearest_spd: A must be square.")

    # If already has NaN/Inf, bail out early
    if not np.all(np.isfinite(A)):
        n = A.shape[0]
        return fallback_scale * np.eye(n)

    B = (A + A.T) / 2.0

    # SVD-based symmetrization
    U, s, Vt = np.linalg.svd(B)
    H = (Vt.T * s) @ Vt
    Ahat = (B + H) / 2.0
    Ahat = (Ahat + Ahat.T) / 2.0

    n = Ahat.shape[0]
    I = np.eye(n)

    for k in range(1, max_tries + 1):
        if not np.all(np.isfinite(Ahat)):
            return fallback_scale * I
        try:
            np.linalg.cholesky(Ahat)
            return Ahat
        except np.linalg.LinAlgError:
            # compute min eigenvalue safely
            eigvals = np.linalg.eigvalsh(Ahat)
            if not np.all(np.isfinite(eigvals)):
                return fallback_scale * I
            mineig = np.min(eigvals)
            # shift upward
            shift = (-mineig + np.finfo(float).eps) * (k**2)
            Ahat = Ahat + shift * I

    # final fallback
    return fallback_scale * I

def safe_logdet(M):
    sign, ld = np.linalg.slogdet(M)
    return ld if sign > 0 else -np.inf

def systematic_resampling(w):
    n = len(w)
    positions = (np.random.rand() + np.arange(n)) / n
    cumsum = np.cumsum(w)
    idx = np.zeros(n, dtype=int)
    i = j = 0
    while i < n:
        if positions[i] < cumsum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx

def logsumexp(a):
    a = np.asarray(a)
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(a - m)))

def log_invgamma_std(x, s, nu):
    if x <= 0:
        return -np.inf
    return (np.log(2.0)
            - loggamma(nu/2.0)
            + (nu/2.0)*np.log(nu*(s**2)/2.0)
            - ((nu+1.0)/2.0)*np.log(x**2)
            - nu*(s**2)/(2.0*(x**2)))

def draw_invgamma_std(s, nu, size=1):
    chi2 = np.random.chisquare(df=nu, size=size)
    return np.sqrt((nu*(s**2)) / chi2)

def gamma_ab_from_mean_sd(m, sd):
    b = (sd**2)/m
    a = m/b
    return a, b

# -----------------------------
# 4) Prior logpdf and drawing from prior
# -----------------------------
def log_prior(theta):
    lp = 0.0
    for i in range(npara):
        if int(pmask[i]) == 1:
            if theta[i] != pfix[i]:
                return -np.inf
            continue

        lo, hi = bounds[i]
        if not (lo < theta[i] < hi):
            return -np.inf

        sh = int(pshape[i])
        a1, a2 = p1[i], p2[i]

        if sh == 1:  # beta(mean, sd)
            m, sd = a1, a2
            aa = (1-m) * (m**2) / (sd**2) - m
            bb = aa * (1/m - 1)
            if not (0.0 < theta[i] < 1.0):
                return -np.inf
            lp += beta_dist(aa, bb).logpdf(theta[i])

        elif sh == 2:  # gamma(mean, sd)
            m, sd = a1, a2
            aa, bb = gamma_ab_from_mean_sd(m, sd)
            if theta[i] <= 0:
                return -np.inf
            lp += gamma_dist(aa, scale=bb).logpdf(theta[i])

        elif sh == 3:  # normal(mean, sd)
            lp += norm_dist(a1, a2).logpdf(theta[i])

        elif sh == 4:  # invgamma(s, nu) on std dev
            lp += log_invgamma_std(theta[i], a1, a2)

        elif sh == 5:  # uniform(a,b)
            if not (a1 <= theta[i] <= a2):
                return -np.inf
            lp += -np.log(a2 - a1)

        elif sh == 0:
            lp += 0.0
        else:
            raise ValueError(f"Unsupported prior shape {sh} at i={i}")

    return lp

def draw_one_prior():
    th = np.zeros(npara)
    for i in range(npara):
        lo, hi = bounds[i]
        sh = int(pshape[i])
        a1, a2 = p1[i], p2[i]

        if int(pmask[i]) == 1:
            th[i] = pfix[i]
            continue

        while True:
            if sh == 1:
                m, sd = a1, a2
                aa = (1-m) * (m**2) / (sd**2) - m
                bb = aa * (1/m - 1)
                x = beta_dist(aa, bb).rvs()

            elif sh == 2:
                m, sd = a1, a2
                aa, bb = gamma_ab_from_mean_sd(m, sd)
                x = gamma_dist(aa, scale=bb).rvs()

            elif sh == 3:
                x = norm_dist(a1, a2).rvs()

            elif sh == 4:
                x = draw_invgamma_std(a1, a2, size=1)[0]

            elif sh == 5:
                x = a1 + (a2 - a1) * np.random.rand()

            elif sh == 0:
                x = a1

            else:
                raise ValueError(f"Unsupported prior shape {sh} at i={i}")

            if lo < x < hi:
                th[i] = x
                break

    return th

# -----------------------------
# 5) gensys
# -----------------------------


def _qzdiv(div, A, B, Q, Z):
    """
    MATLAB-like qzdiv(div, a, b, q, z):
    reorder QZ so that "stable" generalized eigenvalues |b_ii| <= div*|a_ii|
    appear first (upper-left block).

    We implement this by calling ordqz with a stable-first sorting rule.
    """
    div = float(div)
    realsmall = 1e-6

    def stable_first(alpha, beta):
        # alpha ~ diag(A), beta ~ diag(B) for generalized eigenvalues beta/alpha.
        # Stable if |beta| <= div*|alpha|, except alpha≈0 treated as unstable.
        alpha = np.asarray(alpha)
        beta = np.asarray(beta)
        out = np.zeros_like(alpha, dtype=bool)
        az = np.abs(alpha) < realsmall
        idx = ~az
        out[idx] = (np.abs(beta[idx]) <= div * np.abs(alpha[idx]))
        out[az] = False
        return out

    AA, BB, alpha, beta, QQ, ZZ = linalg.ordqz(A, B, sort=stable_first, output="complex")
    # ordqz returns AA = QQ^H A ZZ, BB = QQ^H B ZZ. We also need total Q,Z:
    Qnew = QQ.conj().T @ Q
    Znew = Z @ ZZ
    return AA, BB, Qnew, Znew


def gensys(g0, g1, c=None, psi=None, pi=None, div=None):
    """
    Python port of Chris Sims' gensys.m (as uploaded).

    Returns:
      G1, C, impact, fmat, fwt, ywt, gev, eu, loose

    Notes:
      - eu[0]=existence flag, eu[1]=uniqueness flag.
      - eu=[-2,-2] for coincident zeros.
      - If div is None, uses Sims' default behavior (div=1.01, with optional adjustment).
      - Uses realsmall=1e-6 like MATLAB code.
    """
    eu = np.array([0.0, 0.0])
    realsmall = 1e-6

    g0 = np.asarray(g0, float)
    g1 = np.asarray(g1, float)
    n = g0.shape[0]

    if c is None:
        c = np.zeros((n, 1))
    else:
        c = np.asarray(c, float).reshape(n, 1)

    if psi is None:
        psi = np.zeros((n, 0))
    else:
        psi = np.asarray(psi, float)
        if psi.ndim == 1:
            psi = psi.reshape(n, 1)

    if pi is None:
        pi = np.zeros((n, 0))
    else:
        pi = np.asarray(pi, float)
        if pi.ndim == 1:
            pi = pi.reshape(n, 1)

    fixdiv = (div is not None)
    if not fixdiv:
        div = 1.01
    div = float(div)

    # QZ decomposition: q*g0*z = a, q*g1*z = b
    A, B, Q, Z = linalg.qz(g0, g1, output="complex")

    nunstab = 0
    zxz = False

    # div adjustment loop (only if div not fixed)
    for i in range(n):
        if not fixdiv:
            if abs(A[i, i]) > 0:
                divhat = abs(B[i, i]) / abs(A[i, i])
                # Sims bugfix: use <= not < to avoid misclassification at boundary
                if (1 + realsmall) < divhat and divhat <= div:
                    div = 0.5 * (1 + divhat)

        nunstab += int(abs(B[i, i]) > div * abs(A[i, i]))
        if abs(A[i, i]) < realsmall and abs(B[i, i]) < realsmall:
            zxz = True

    # sort QZ (like qzdiv)
    if zxz:
        # coincident zeros
        eu = np.array([-2.0, -2.0])
        # MATLAB returns empty outputs on zxz
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]),
                np.array([]), eu, np.array([]))

    A, B, Q, Z = _qzdiv(div, A, B, Q, Z)

    gev = np.column_stack([np.diag(A), np.diag(B)])  # like MATLAB gev=[diag(a) diag(b)]

    nstab = n - nunstab
    q1 = Q[:nstab, :]
    q2 = Q[nstab:, :]
    # z1,z2 are not used directly below, but kept for parity with MATLAB
    z1 = Z[:, :nstab].conj().T
    z2 = Z[:, nstab:].conj().T

    a2 = A[nstab:, nstab:]
    b2 = B[nstab:, nstab:]

    neta = pi.shape[1]
    etawt = q2 @ pi

    # SVD of etawt (unstable block)
    if nunstab == 0:
        ueta = np.zeros((0, 0), dtype=complex)
        veta = np.zeros((neta, 0), dtype=complex)
        deta = np.zeros((0, 0), dtype=float)
        bigev0 = np.array([], dtype=int)
    else:
        U, s, Vh = np.linalg.svd(etawt, full_matrices=False)
        bigev0 = np.where(s > realsmall)[0]
        ueta = U[:, bigev0]
        veta = Vh.conj().T[:, bigev0]
        deta = np.diag(s[bigev0]) if bigev0.size > 0 else np.zeros((0, 0))

    # Existence (corrected code 3/10/04)
    eu[0] = 1.0 if (bigev0.size >= nunstab) else 0.0

    # SVD of etawt1 (stable block)
    if nunstab == n:
        etawt1 = np.zeros((0, neta), dtype=complex)
        ueta1 = np.zeros((0, 0), dtype=complex)
        veta1 = np.zeros((neta, 0), dtype=complex)
        deta1 = np.zeros((0, 0), dtype=float)
    else:
        etawt1 = q1 @ pi
        U1, s1, Vh1 = np.linalg.svd(etawt1, full_matrices=False)
        bigev1 = np.where(s1 > realsmall)[0]
        ueta1 = U1[:, bigev1]
        veta1 = Vh1.conj().T[:, bigev1]
        deta1 = np.diag(s1[bigev1]) if bigev1.size > 0 else np.zeros((0, 0))

    # Uniqueness check
    if veta1.size == 0:
        unique = True
        loose_v = np.zeros((neta, 0), dtype=complex)
        nloose = 0
    else:
        # loose = veta1 - veta*veta'*veta1
        proj = veta @ (veta.conj().T @ veta1) if veta.size > 0 else 0.0
        loose_v = veta1 - proj
        Ul, sl, Vhl = np.linalg.svd(loose_v, full_matrices=False)
        nloose = int(np.sum(np.abs(sl) > realsmall * n))
        unique = (nloose == 0)

    eu[1] = 1.0 if unique else 0.0

    # Build tmat
    # tmat = [I  -(ueta*(deta\veta')*veta1*deta1*ueta1')'];
    if nunstab == 0:
        tmat = np.eye(nstab, dtype=complex)  # nstab==n
    else:
        I_stab = np.eye(nstab, dtype=complex)

        if ueta.size == 0 or deta.size == 0 or veta.size == 0 or veta1.size == 0 or deta1.size == 0 or ueta1.size == 0:
            right = np.zeros((nstab, nunstab), dtype=complex)
        else:
            # deta \ veta'  -> solve(deta, veta^H)
            X = linalg.solve(deta, veta.conj().T)          # (k, neta)
            Utmp = ueta @ X                                 # (nunstab, neta)
            mid = Utmp @ veta1                              # (nunstab, k1)
            mid = mid @ deta1                               # (nunstab, k1)
            right_raw = mid @ ueta1.conj().T                # (nunstab, nstab)
            right = right_raw.conj().T                      # (nstab, nunstab)

        tmat = np.hstack([I_stab, -right])

    # Construct reduced system in z-space
    G0_top = tmat @ A
    G1_top = tmat @ B

    G0 = np.vstack([G0_top,
                    np.hstack([np.zeros((nunstab, nstab), dtype=complex),
                               np.eye(nunstab, dtype=complex)])])
    G1 = np.vstack([G1_top,
                    np.zeros((nunstab, n), dtype=complex)])

    G0I = np.linalg.inv(G0)
    G1z = G0I @ G1

    usix = slice(nstab, n)

    # Constants
    if nunstab > 0:
        C_low = linalg.solve((A[usix, usix] - B[usix, usix]), (q2 @ c))
        Cstack = np.vstack([tmat @ (Q @ c), C_low])
    else:
        Cstack = tmat @ (Q @ c)
    Cz = G0I @ Cstack

    # Impact
    impact_stack = np.vstack([tmat @ (Q @ psi),
                              np.zeros((nunstab, psi.shape[1]), dtype=complex)])
    impact_z = G0I @ impact_stack

    # Forward-looking term objects (mostly unused if shocks i.i.d.)
    if nunstab > 0:
        fmat = linalg.solve(B[usix, usix], A[usix, usix])          # b2\a2
        fwt  = -linalg.solve(B[usix, usix], (q2 @ psi))            # -b2\q2*psi
        ywt  = G0I[:, usix]
    else:
        fmat = np.zeros((0, 0))
        fwt  = np.zeros((0, psi.shape[1]))
        ywt  = np.zeros((n, 0))

    # loose output (5/07/2009 correction: premultiply by G0I)
    if nunstab == n:
        # nstab=0
        loose = G0I @ np.vstack([np.zeros((0, neta), dtype=complex),
                                 np.zeros((nunstab, neta), dtype=complex)])
    else:
        I_eta = np.eye(neta, dtype=complex)
        proj_eta = (veta @ veta.conj().T) if veta.size > 0 else 0.0
        loose_top = etawt1 @ (I_eta - proj_eta)
        loose = G0I @ np.vstack([loose_top,
                                 np.zeros((nunstab, neta), dtype=complex)])

    # Transform back to y-space: y = Z z
    G1 = np.real(Z @ G1z @ Z.conj().T)
    C  = np.real(Z @ Cz)
    impact = np.real(Z @ impact_z)
    loose = np.real(Z @ loose)

    ywt = Z @ ywt  # note: NOT real() per Sims comment

    return G1, C, impact, fmat, fwt, ywt, gev, eu, loose

def gensys_debug_qz(G0, G1, C=None, PSI=None, PI=None, div=1+1e-8, realsmall=None):
    """
    Debug-oriented QZ wrapper:
    - Computes generalized eigenvalues relevant for discrete-time stability
      (transition eigenvalues mu = beta/alpha from ordqz on (G0,G1)).
    - Returns eu flags based on a BK-style *count* diagnostic only.
    - DOES NOT fully implement Sims gensys expectational-error elimination.
      Use it to diagnose why you're rejecting everything.
    """
    G0 = np.asarray(G0, float)
    G1 = np.asarray(G1, float)
    n = G0.shape[0]

    if C is None:   C = np.zeros((n,1))
    if PSI is None: PSI = np.zeros((n,0))
    if PI is None:  PI = np.zeros((n,0))

    if realsmall is None:
        realsmall = np.sqrt(np.finfo(float).eps) * 10.0

    # QZ decomposition (unordered)
    AA, BB, Q, Z = linalg.qz(G0, G1, output='complex')

    alpha = np.diag(AA)
    beta  = np.diag(BB)

    # Discrete-time transition eigenvalues are mu = beta/alpha
    mu = np.full(n, np.inf, dtype=complex)
    ok = np.abs(alpha) > realsmall
    mu[ok] = beta[ok] / alpha[ok]

    # Count "unstable" transition roots
    nunstab = int(np.sum(np.abs(mu) > div))

    # Very rough BK-style uniqueness diagnostic:
    # number of forward-looking errors = number of eta shocks (columns of PI)
    neta = PI.shape[1]
    eu = np.array([1.0, 1.0 if nunstab == neta else 0.0])

    # report "gevmax" as max |mu|
    gevmax = float(np.max(np.abs(mu[np.isfinite(mu)]))) if np.any(np.isfinite(mu)) else float("inf")

    # For now, return None for solution matrices so you don't mistakenly rely on them.
    return None, None, None, None, None, None, np.array([gevmax, eu[1]])

# -----------------------------
# 6) Model solving
# -----------------------------
def model_solution(theta):
    tau,kappa,psi1,psi2,rA,piA,gammaQ,rho_R,rho_g,rho_z,sigma_R,sigma_g,sigma_z = theta
    bet = 1.0/(1.0 + rA/400.0)

    neq, neta, neps = 8, 2, 3
    GAM0 = np.zeros((neq, neq))
    GAM1 = np.zeros((neq, neq))
    C    = np.zeros((neq, 1))
    PSI  = np.zeros((neq, neps))
    PPI  = np.zeros((neq, neta))

    y_t, pi_t, R_t, y1_t, g_t, z_t, Ey_t1, Epi_t1 = range(8)
    z_sh, g_sh, R_sh = range(3)
    ey_sh, epi_sh = range(2)

    # Eq1: IS
    GAM0[0, y_t]    = 1
    GAM0[0, R_t]    = 1/tau
    GAM0[0, g_t]    = -(1-rho_g)
    GAM0[0, z_t]    = -rho_z/tau
    GAM0[0, Ey_t1]  = -1
    GAM0[0, Epi_t1] = -1/tau

    # Eq2: Phillips
    GAM0[1, y_t]    = -kappa
    GAM0[1, pi_t]   = 1
    GAM0[1, g_t]    = kappa
    GAM0[1, Epi_t1] = -bet

    # Eq3: Taylor
    GAM0[2, y_t]   = -(1-rho_R)*psi2
    GAM0[2, pi_t]  = -(1-rho_R)*psi1
    GAM0[2, R_t]   = 1
    GAM0[2, g_t]   = (1-rho_R)*psi2
    GAM1[2, R_t]   = rho_R
    PSI[2, R_sh]   = 1

    # Eq4: y1_t = y_{t-1}
    GAM0[3, y1_t] = 1
    GAM1[3, y_t]  = 1

    # Eq5: g AR(1)
    GAM0[4, g_t] = 1
    GAM1[4, g_t] = rho_g
    PSI[4, g_sh] = 1

    # Eq6: z AR(1)
    GAM0[5, z_t] = 1
    GAM1[5, z_t] = rho_z
    PSI[5, z_sh] = 1

    # Eq7: aux expectations
    GAM0[6, y_t]   = 1
    GAM1[6, Ey_t1] = 1
    PPI[6, ey_sh]  = 1

    # Eq8: aux expectations
    GAM0[7, pi_t]   = 1
    GAM1[7, Epi_t1] = 1
    PPI[7, epi_sh]  = 1

    G1_sol, C_sol, impact, fmat, fwt, ywt, gev, eu, loose = gensys(GAM0, GAM1, c=C, psi=PSI, pi=PPI, div=1.0 + 1e-8)

    if G1_sol.size == 0 or eu[1] < 0.5:
        return None, None, np.array([float(np.max(np.abs(gev[:,1] / gev[:,0]))) if gev.size else np.inf, 0.0])

    return G1_sol, impact, np.array([float(np.max(np.abs(gev[:,1] / gev[:,0]))), 1.0])




# -----------------------------
# 7) generating State space and Kalman filter
# -----------------------------
def sysmat(T1, T0, theta):
    tau,kappa,psi1,psi2,rA,piA,gammaQ,rho_R,rho_g,rho_z,sigma_R,sigma_g,sigma_z = theta

    ny = 3
    y_t, pi_t, R_t, y1_t, g_t, z_t, Ey_t1, Epi_t1 = range(8)

    Phi = T1
    R   = T0  # impact on eps

    Se = np.zeros((3,3))
    Se[0,0] = sigma_z**2
    Se[1,1] = sigma_g**2
    Se[2,2] = sigma_R**2

    A = np.zeros((ny,1))
    A[0,0] = gammaQ
    A[1,0] = piA
    A[2,0] = piA + rA + 4*gammaQ

    B = np.zeros((ny, Phi.shape[0]))
    B[0, y_t]  =  1
    B[0, y1_t] = -1
    B[0, z_t]  =  1
    B[1, pi_t] =  4
    B[2, R_t]  =  4

    H = np.zeros((ny,ny))
    H[0,0] = (0.20*0.579923)**2
    H[1,1] = (0.20*1.470832)**2
    H[2,2] = (0.20*2.237937)**2

    return A, B, H, R, Se, Phi

def kalman_loglik(A,B,H,R,Se,Phi,y, jitter=1e-10):
    y = np.asarray(y)
    T, l = y.shape
    n = Phi.shape[0]
    Q = R @ Se @ R.T

    try:
        P = linalg.solve_discrete_lyapunov(Phi, Q)
        P = nearest_spd(P)
    except Exception:
        P = nearest_spd(100.0*np.eye(n))

    s = np.zeros((n,1))
    ll = 0.0
    I_l = np.eye(l)

    for t in range(T):
        s_pred = Phi @ s
        P_pred = Phi @ P @ Phi.T + Q

        y_pred = A + B @ s_pred
        v = y[t,:].reshape(-1,1) - y_pred

        F = B @ P_pred @ B.T + H
        F = (F + F.T)/2 + jitter*I_l

        ld = safe_logdet(F)
        if not np.isfinite(ld):
            F = (F + F.T)/2 + 1e-6*I_l
            ld = safe_logdet(F)
            if not np.isfinite(ld):
                return -np.inf

        Finv = np.linalg.inv(F)
        ll += (-0.5*l*np.log(2*np.pi) - 0.5*ld - 0.5*(v.T @ Finv @ v).item())

        K = P_pred @ B.T @ Finv
        s = s_pred + K @ v
        P = P_pred - K @ B @ P_pred
        P = nearest_spd((P + P.T)/2)

    return ll

def kalman_loglik_kalman2(A,B,H,R,Se,Phi,y, jitter=0.0):
    y = np.asarray(y)
    T, l = y.shape
    n = Phi.shape[0]
    Q = R @ Se @ R.T

    # H&S kalman2: P0 = nearestSPD(dlyap(Phi, Q))
    try:
        P = linalg.solve_discrete_lyapunov(Phi, Q)
        P = nearest_spd(P)
    except Exception:
        return -np.inf

    s = np.zeros((n,1))
    ll = 0.0
    I_l = np.eye(l)

    for t in range(T):
        s_pred = Phi @ s
        P_pred = Phi @ P @ Phi.T + Q

        y_pred = A + B @ s_pred
        v = y[t,:].reshape(-1,1) - y_pred

        F = B @ P_pred @ B.T + H
        F = (F + F.T)/2 + jitter*I_l   # keep jitter 0 unless needed

        sign, ld = np.linalg.slogdet(F)
        if sign <= 0 or not np.isfinite(ld):
            return -np.inf

        Finv = np.linalg.inv(F)
        ll += (-0.5*l*np.log(2*np.pi) - 0.5*ld - 0.5*(v.T @ Finv @ v).item())

        K = P_pred @ B.T @ Finv
        s = s_pred + K @ v
        P = P_pred - K @ B @ P_pred
        # Only SPD-fix if needed:
        P = (P + P.T)/2

    return ll


def log_posterior_tempered(theta, phi):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, -np.inf,  -np.inf

    T1, T0, GEV = model_solution(theta)
    if T1 is None or GEV[1] < 0.5:
        return -np.inf, -np.inf, lp

    A, B, H, R, Se, Phi = sysmat(T1, T0, theta)
  # ll = kalman_loglik(A, B, H, R, Se, Phi, data)
    ll = kalman_loglik_kalman2(A, B, H, R, Se, Phi, data)

    if not np.isfinite(ll):
        return -np.inf, -np.inf, lp

    return phi*ll + lp, ll, lp


def log_posterior_tempered_debug(theta, phi, debug=False):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        if debug: print("[FAIL prior]", theta)
        return -np.inf, -np.inf, -np.inf

    out = model_solution(theta)
    if out is None:
        if debug: print("[FAIL model_solution returned None]", theta)
        return -np.inf, -np.inf, lp

    T1, T0, GEV = out
    if T1 is None:
        if debug: print("[FAIL no solution] GEV:", GEV, "theta:", theta)
        return -np.inf, -np.inf, lp

    if debug:
        print("[OK solution] GEV:", GEV)

    A, B, H, R, Se, Phi = sysmat(T1, T0, theta)
    ll = kalman_loglik(A, B, H, R, Se, Phi, data)
    if not np.isfinite(ll):
        if debug:
            print("[FAIL kalman] ll:", ll)
            print("A:", A.ravel(), "diag(H):", np.diag(H), "diag(Se):", np.diag(Se))
        return -np.inf, -np.inf, lp

    return phi*ll + lp, ll, lp

#version 2
def log_posterior_tempered_debug_v2(theta, phi, debug=False, div=1.0 + 1e-8):
    """
    Diagnosing version of log_posterior_tempered.

    - Uses your usual log_prior() and sysmat()/kalman_loglik().
    - Calls gensys_debug_qz(...) (the debug QZ/eigenvalue diagnostic) instead of full gensys.
      This lets you see whether failures are from:
        (a) prior support
        (b) DSGE solution construction (e.g., NaNs)
        (c) Kalman likelihood numerics
        (d) root stability diagnostics (gevmax / BK count)

    Returns:
      (logpost_tempered, ll, lp)

    If debug=True, prints the failure stage and key diagnostics for this theta.
    """

    # --- 1) prior ---
    lp = log_prior(theta)
    if not np.isfinite(lp):
        if debug:
            print("[FAIL prior] lp not finite. theta =", np.asarray(theta))
        return -np.inf, -np.inf, -np.inf

    # --- 2) build the model matrices via your model_solution, but swap gensys inside it ---
    # IMPORTANT: For this function to use gensys_debug_qz, your model_solution(theta)
    # must call gensys_debug_qz(...) internally instead of gensys(...).
    # (Easiest: temporarily rename/alias gensys = gensys_debug_qz before calling model_solution.)
    T1, T0, GEV = model_solution(theta)

    # If model_solution returns None, we can't proceed to likelihood
    if T1 is None or T0 is None:
        if debug:
            print("[FAIL model_solution] returned None. theta =", np.asarray(theta))
            print("  GEV =", GEV)
        return -np.inf, -np.inf, lp

    # --- 3) optional: inspect stability diagnostic but DO NOT reject on it during debugging ---
    if debug:
        print("[OK model_solution] GEV =", GEV, "  (NOTE: not rejecting on GEV in debug mode)")

    # --- 4) state-space + kalman likelihood ---
    try:
        A, B, H, R, Se, Phi = sysmat(T1, T0, theta)
    except Exception as e:
        if debug:
            print("[FAIL sysmat] exception:", repr(e))
            print("  theta =", np.asarray(theta))
        return -np.inf, -np.inf, lp

    ll = kalman_loglik(A, B, H, R, Se, Phi, data)
    if not np.isfinite(ll):
        if debug:
            print("[FAIL kalman] ll not finite. ll =", ll)
            print("  theta =", np.asarray(theta))
            # useful extra diagnostics
            try:
                print("  A.ravel() =", A.ravel())
                print("  diag(H)   =", np.diag(H))
                print("  diag(Se)  =", np.diag(Se))
            except Exception:
                pass
        return -np.inf, -np.inf, lp

    # --- 5) tempered posterior ---
    return phi * ll + lp, ll, lp



def smc_log_evidence_from_logliks(logliks_by_stage, phis):
    """
    logliks_by_stage: list of arrays, length Nstages
        logliks_by_stage[n] is shape (Np,) containing ll(theta_i) at stage n
        (these are full-data loglik values, not incremental)
    phis: array-like of length Nstages with phi[0]=0, phi[-1]=1

    Returns: logZ = log marginal data density estimate
    """
    phis = np.asarray(phis, float)
    assert len(logliks_by_stage) == len(phis)
    logZ = 0.0

    for n in range(1, len(phis)):
        dphi = phis[n] - phis[n-1]
        ll = np.asarray(logliks_by_stage[n-1], float)  # use ll at previous stage particles
        # incremental weights: w_i ∝ exp(dphi * ll_i)
        a = dphi * ll
        m = np.max(a)
        log_mean_w = m + np.log(np.mean(np.exp(a - m)))
        logZ += log_mean_w

    return logZ

# -----------------------------
# 8) SMC settings
# -----------------------------
tune = dict(
    npart=300,
    nphi=60,
    lam=2.0,
    c=0.5,
    acpt=0.25,
    trgt=0.25,
    ess_threshold=0.5,
)
phi_grid = np.linspace(0,1,tune["nphi"])**tune["lam"]
tune["phi"] = phi_grid
print(f"SMC settings: npart={tune['npart']}, nphi={tune['nphi']}, lam={tune['lam']}")

# storage
parasim = np.zeros((tune["nphi"], tune["npart"], npara))
wtsim   = np.zeros((tune["npart"], tune["nphi"]))
ESSsim  = np.zeros((tune["nphi"],))
acptsim = np.zeros((tune["nphi"],))



# -----------------------------
# 9) Random-Walk MH for mutation
# -----------------------------
def mutation_rwmh(theta0, ll0, post0, phi_prev, phi_now, c_scale, R_cov):
    try:
        L = np.linalg.cholesky(R_cov + 1e-12*np.eye(R_cov.shape[0]))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(nearest_spd(R_cov) + 1e-12*np.eye(R_cov.shape[0]))

    prop = theta0 + c_scale * (L @ np.random.randn(npara))
    postx, llx, _ = log_posterior_tempered(prop, phi_now)

    post0_updated = post0 + (phi_now - phi_prev) * ll0

    if not np.isfinite(postx):
        return theta0, ll0, post0_updated, 0

    loga = postx - post0_updated
    if np.log(np.random.rand()) < min(0.0, loga):
        return prop, llx, postx, 1
    else:
        return theta0, ll0, post0_updated, 0

# -----------------------------
# 10) Initialization
# -----------------------------

phi0 = tune["phi"][0]
N = tune["npart"]

theta0  = np.zeros((N, npara))
loglh   = np.full(N, -np.inf)
logpost = np.full(N, -np.inf)

print("Initializing particles")
t0 = time.time()
last_t = time.time()
bad = 0

for i in range(N):
    th = draw_one_prior()
    lpst, ll, _ = log_posterior_tempered(th, phi0)

    if not np.isfinite(ll):
        bad += 1
        tries = 0
        while (not np.isfinite(ll)) and tries < 5000:
            tries += 1
            th = draw_one_prior()
            lpst, ll, _ = log_posterior_tempered(th, phi0)
        if not np.isfinite(ll):
            raise RuntimeError(f"Init failed at particle {i}: could not find finite likelihood after retries.")

    theta0[i, :] = th
    loglh[i]     = ll
    logpost[i]   = lpst

    if (i + 1) % 100 == 0 or (i + 1) == N:
        now = time.time()
        dt_total = now - t0
        dt_last  = now - last_t
        last_rate = (100 / dt_last) if (i + 1) % 100 == 0 else float("nan")
        avg_rate  = (i + 1) / max(dt_total, 1e-9)
        eta_sec   = (N - (i + 1)) / max(avg_rate, 1e-9)
        print(f"  init {i+1:>4}/{N} | last100: {last_rate:>6.1f}/s | avg: {avg_rate:>6.1f}/s | bad={bad} | ETA {eta_sec/60:.2f} min")
        last_t = now

parasim[0, :, :] = theta0
wtsim[:, 0]      = 1.0 / N
print(f"Initialized {N} particles. bad(retried)={bad} | time={time.time()-t0:.2f}s")
print("Init complete. Entering SMC loop")

# -----------------------------
# 11) SMC loop
# ==============================
import numpy as np, time, os, sys, faulthandler
from joblib import Parallel, delayed, parallel_backend

faulthandler.dump_traceback_later(120, repeat=True)

N_JOBS  = 2
N_TASKS = 8

def _mutate_chunk(j0, j1, theta_mat, loglh, logpost, phi_prev, phi_now, c_scale, R_cov):
    t0 = time.time()
    pid = os.getpid()

    out_theta = np.empty((j1 - j0, theta_mat.shape[1]), dtype=float)
    out_ll    = np.empty((j1 - j0,), dtype=float)
    out_post  = np.empty((j1 - j0,), dtype=float)
    out_acc   = 0

    for k, j in enumerate(range(j0, j1)):
        thj, llj, pj, aj = mutation_rwmh(
            theta_mat[j, :], loglh[j], logpost[j],
            phi_prev, phi_now, c_scale, R_cov
        )
        out_theta[k, :] = thj
        out_ll[k]       = llj
        out_post[k]     = pj
        out_acc        += aj

    return j0, out_theta, out_ll, out_post, out_acc, pid, (time.time() - t0)

with parallel_backend("loky", inner_max_num_threads=1):
    pool = Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes", verbose=0)

    # Pre-warm
    _ = pool(delayed(lambda x: x)(k) for k in range(N_JOBS))
    print(f"Joblib pre-warm done with {N_JOBS} workers.")

    for t in range(1, tune["nphi"]):
        t_stage0 = time.time()

        phi_prev = tune["phi"][t-1]
        phi_now  = tune["phi"][t]
        dphi     = phi_now - phi_prev

        # (1) correction
        w_prev = wtsim[:, t-1]
        logw = np.log(w_prev + 1e-300) + dphi * loglh
        lz = logsumexp(logw)
        if not np.isfinite(lz):
            raise RuntimeError(f"Weight collapse at stage {t+1}/{tune['nphi']}.")

        w = np.exp(logw - lz)
        w = w / np.sum(w)
        wtsim[:, t] = w

        # (2) ESS
        ESS = 1.0 / np.sum(w**2)
        ESSsim[t] = ESS

        # (3) resample
        do_resample = ESS < (tune["ess_threshold"] * tune["npart"])
        if do_resample:
            idx = systematic_resampling(w)
            parasim[t-1, :, :] = parasim[t-1, idx, :]
            loglh   = loglh[idx]
            logpost = logpost[idx]
            w = np.ones(tune["npart"]) / tune["npart"]
            wtsim[:, t] = w

        # (4) proposal covariance
        theta_mat = parasim[t-1, :, :]
        mu = np.sum(theta_mat * w.reshape(-1, 1), axis=0)
        z  = theta_mat - mu.reshape(1, -1)
        R_cov = (z * w.reshape(-1, 1)).T @ z
        R_cov = nearest_spd(R_cov + 1e-12 * np.eye(npara))

        # (5) adapt c
        tune["c"] *= np.exp(0.5 * (tune["acpt"] - tune["trgt"]))

        # (6) chunk tasks
        Np = tune["npart"]
        cuts = np.linspace(0, Np, N_TASKS + 1, dtype=int)
        tasks = [(cuts[i], cuts[i+1]) for i in range(N_TASKS) if cuts[i] < cuts[i+1]]

        # (7) mutation
        t_mut0 = time.time()
        try:
            results = pool(
                delayed(_mutate_chunk)(j0, j1, theta_mat, loglh, logpost, phi_prev, phi_now, tune["c"], R_cov)
                for (j0, j1) in tasks
            )

            #   results = Parallel(n_jobs=N_JOBS, backend="threading", prefer="threads", batch_size=20)(
            #       delayed(_mutate_one_global)(j) for j in range(Np)
            #   )


        except Exception as e:
            print("\n[ERROR] Parallel mutation crashed:", repr(e))
            # print traceback immediately
            import traceback
            traceback.print_exc()
            raise

        mut_time = time.time() - t_mut0

        # (8) collect
        new_theta = np.zeros_like(theta_mat)
        new_ll    = np.zeros_like(loglh)
        new_post  = np.zeros_like(logpost)
        acc = 0
        pids = []
        secs = []

        for j0, th_blk, ll_blk, post_blk, acc_blk, pid, sec in results:
            j1 = j0 + th_blk.shape[0]
            new_theta[j0:j1, :] = th_blk
            new_ll[j0:j1]       = ll_blk
            new_post[j0:j1]     = post_blk
            acc += acc_blk
            pids.append(pid)
            secs.append(sec)

        parasim[t, :, :] = new_theta
        loglh   = new_ll
        logpost = new_post
        tune["acpt"] = acc / Np
        acptsim[t]   = tune["acpt"]

        stage_time = time.time() - t_stage0

        print(
            f"Stage {t+1:>3}/{tune['nphi']} | phi={phi_now: .4f} | dphi={dphi:.4g} | "
            f"ESS={ESS: .1f} | resample={do_resample} | c={tune['c']:.3f} | acc={tune['acpt']:.3f} | "
            f"mut={mut_time:.1f}s | stage={stage_time:.1f}s"
        )
        print(f"    [parallel diag] workers={len(set(pids))}/{N_JOBS} | slowest_chunk={max(secs):.2f}s")

print("SMC finished.")


#später löschen:
print("parasim last stage min/max:", np.min(parasim[-1,:,:]), np.max(parasim[-1,:,:]))
print("wtsim last stage sum:", np.sum(wtsim[:, -1]))


# -----------------------------
# 12) Posterior draws and SMC trace
# -----------------------------
theta_prior = parasim[0,:,:]
theta_post  = parasim[-1,:,:]
w_post      = wtsim[:, -1]
w_post      = w_post / np.sum(w_post)

trace_mean = np.zeros((tune["nphi"], npara))
for t in range(tune["nphi"]):
    w_t = wtsim[:, t]
    w_t = w_t / np.sum(w_t)
    trace_mean[t, :] = np.sum(parasim[t, :, :] * w_t.reshape(-1, 1), axis=0)

# -----------------------------
# 13) Plottings
# -----------------------------
def plot_prior_posterior():
    ncols = 3
    nrows = int(np.ceil(npara / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8*nrows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.hist(theta_prior[:,i], bins=40, density=True, alpha=0.5, label="Prior (valid)")
        ax.hist(theta_post[:,i],  bins=40, density=True, alpha=0.5, label="Posterior")
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend()

    for k in range(npara, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Prior vs Posterior (SMC output)", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_trace_means():
    ncols = 3
    nrows = int(np.ceil(npara / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2*nrows))
    axes = np.array(axes).reshape(-1)

    x = np.arange(tune["nphi"])
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.plot(x, trace_mean[:,i])
        ax.set_title(f"{name} (weighted mean across stages)")
        ax.set_xlabel("SMC stage")
        ax.grid(True, alpha=0.2)

    for k in range(npara, len(axes)):
        axes[k].axis("off")

    fig.suptitle("SMC Trace Plots (weighted means)", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_diagnostics():
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(ESSsim, label="ESS")
    ax.set_title("Effective Sample Size by stage")
    ax.set_xlabel("SMC stage")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(acptsim, label="MH acceptance rate")
    ax.set_title("Mutation acceptance rate by stage")
    ax.set_xlabel("SMC stage")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_prior_posterior()
plot_trace_means()
plot_diagnostics()

# -----------------------------
# 14) Posterior summary
# -----------------------------
def weighted_quantile(x, w, q):
    idx = np.argsort(x)
    x = x[idx]; w = w[idx]
    cw = np.cumsum(w)
    return np.interp(q, cw, x)

print("\nPosterior summaries (weighted):")
for i, name in enumerate(param_names):
    m = np.sum(theta_post[:,i] * w_post)
    q10 = weighted_quantile(theta_post[:,i], w_post, 0.10)
    q50 = weighted_quantile(theta_post[:,i], w_post, 0.50)
    q90 = weighted_quantile(theta_post[:,i], w_post, 0.90)
    print(f"{name:>7}: mean={m: .4f} | q10={q10: .4f} | q50={q50: .4f} | q90={q90: .4f}")

#%%
# ============================================================
# EXTRA SMC PLOTS: 2D likelihood surfaces + SMC path + invalid regions
# Paste AFTER the SMC run (needs parasim, wtsim, tune, param_names, log_posterior_tempered).
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------
# Helpers
# ---------------------------
def weighted_mean_stage(parasim, wtsim):
    nphi, N, npara = parasim.shape
    m = np.zeros((nphi, npara))
    for t in range(nphi):
        w = wtsim[:, t]
        w = w / np.sum(w)
        m[t, :] = np.sum(parasim[t, :, :] * w[:, None], axis=0)
    return m

def weighted_cov(xy, w):
    w = w / np.sum(w)
    mu = np.sum(xy * w[:, None], axis=0)
    z = xy - mu[None, :]
    C = (z * w[:, None]).T @ z
    return mu, C

def weighted_kde2d(x, y, w, gridsize=120, bw=None, xlim=None, ylim=None):
    """
    Light-weight weighted Gaussian KDE on a grid (no seaborn).
    bw: bandwidth scalar (if None: Silverman-like heuristic).
    """
    x = np.asarray(x); y = np.asarray(y); w = np.asarray(w)
    w = w / np.sum(w)

    if xlim is None: xlim = (np.min(x), np.max(x))
    if ylim is None: ylim = (np.min(y), np.max(y))

    gx = np.linspace(xlim[0], xlim[1], gridsize)
    gy = np.linspace(ylim[0], ylim[1], gridsize)
    Xg, Yg = np.meshgrid(gx, gy, indexing="xy")

    # heuristic bandwidth from weighted covariance
    mu, C = weighted_cov(np.column_stack([x, y]), w)
    s = np.sqrt(np.maximum(np.diag(C), 1e-12))
    if bw is None:
        n_eff = 1.0 / np.sum(w**2)
        bw = (n_eff ** (-1/6))  # 2D rule-of-thumb factor
    hx, hy = bw * s[0], bw * s[1]
    hx = max(hx, 1e-6); hy = max(hy, 1e-6)

    # compute KDE
    Z = np.zeros_like(Xg, dtype=float)
    for i in range(len(x)):
        dx = (Xg - x[i]) / hx
        dy = (Yg - y[i]) / hy
        Z += w[i] * np.exp(-0.5 * (dx*dx + dy*dy))
    Z /= (2*np.pi*hx*hy)
    return gx, gy, Z

def grid_loglik(theta_center, idx_x, idx_y, gx, gy, phi=1.0):
    """
    Evaluate log-likelihood (not posterior) on a grid for two params, holding others fixed at theta_center.
    Uses log_posterior_tempered(theta, phi) which returns (phi*ll+lp, ll, lp).
    We extract ll and also identify invalid regions.
    """
    Z_ll = np.full((len(gy), len(gx)), -np.inf)
    Z_ok = np.zeros((len(gy), len(gx)), dtype=bool)

    th = theta_center.copy()
    for iy, yv in enumerate(gy):
        for ix, xv in enumerate(gx):
            th[idx_x] = xv
            th[idx_y] = yv
            _, ll, _ = log_posterior_tempered(th, phi)
            if np.isfinite(ll):
                Z_ll[iy, ix] = ll
                Z_ok[iy, ix] = True
    return Z_ll, Z_ok

def grad_magnitude(Z):
    """
    simple finite-diff gradient magnitude proxy on a grid (handles -inf by nan).
    """
    Zn = Z.copy()
    Zn[~np.isfinite(Zn)] = np.nan
    gx = np.nan_to_num(np.gradient(Zn, axis=1))
    gy = np.nan_to_num(np.gradient(Zn, axis=0))
    G = np.sqrt(gx*gx + gy*gy)
    G[np.isnan(G)] = 0.0
    return G

# ---------------------------
# Choose parameter pairs to visualize
# ---------------------------
pairs = [
    ("rho_R", "psi1"),   # policy smoothing vs inflation response
    ("kappa", "psi1"),   # slope of Phillips curve vs policy
    ("sigma_g", "rho_g") # shock sd vs persistence
]

name_to_idx = {n:i for i,n in enumerate(param_names)}
stage_means = weighted_mean_stage(parasim, wtsim)

# pick representative stage indices for particle clouds
stage_ids = [0, max(1, tune["nphi"]//3), max(1, 2*tune["nphi"]//3), tune["nphi"]-1]

# ---------------------------
# 1) Likelihood surface + SMC path + particle clouds
# ---------------------------
for (nx, ny) in pairs:
    ix, iy = name_to_idx[nx], name_to_idx[ny]

    # center grid around posterior mean (last stage weighted mean)
    th_center = stage_means[-1].copy()
    x0, y0 = th_center[ix], th_center[iy]

    # choose grid spans from posterior spread (last stage particles)
    w_last = wtsim[:, -1]; w_last = w_last / np.sum(w_last)
    x_samp = parasim[-1, :, ix]
    y_samp = parasim[-1, :, iy]

    # robust range (weighted quantiles)
    def wquant(a, w, q):
        idx = np.argsort(a)
        a = a[idx]; w = w[idx]
        cw = np.cumsum(w)
        return np.interp(q, cw, a)

    xlo, xhi = wquant(x_samp, w_last, 0.02), wquant(x_samp, w_last, 0.98)
    ylo, yhi = wquant(y_samp, w_last, 0.02), wquant(y_samp, w_last, 0.98)

    # widen a bit to see curvature / invalid regions
    pad_x = 0.35 * (xhi - xlo + 1e-12)
    pad_y = 0.35 * (yhi - ylo + 1e-12)
    xlim = (xlo - pad_x, xhi + pad_x)
    ylim = (ylo - pad_y, yhi + pad_y)

    gx = np.linspace(xlim[0], xlim[1], 90)
    gy = np.linspace(ylim[0], ylim[1], 90)

    print(f"Computing likelihood grid for ({nx}, {ny}) ...")
    Z_ll, Z_ok = grid_loglik(th_center, ix, iy, gx, gy, phi=1.0)
    Z_grad = grad_magnitude(Z_ll)

    # rescale likelihood for plotting
    finite_ll = Z_ll[np.isfinite(Z_ll)]
    if finite_ll.size == 0:
        print("  -> no finite likelihood on this grid. Skipping.")
        continue
    vmax = np.max(finite_ll)
    Z_rel = Z_ll - vmax  # log-lik relative to max (<=0)

    # ---- Plot: likelihood surface with SMC path + clouds
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        Z_rel,
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="auto"
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("log-likelihood relative to grid max")

    # overlay invalid region contour
    ax.contour(
        gx, gy, (~Z_ok).astype(float),
        levels=[0.5], linewidths=1.0
    )

    # overlay SMC path (weighted means)
    path_x = stage_means[:, ix]
    path_y = stage_means[:, iy]
    ax.plot(path_x, path_y, marker="o", markersize=2.5, linewidth=1.0, label="SMC path (stage means)")

    # overlay particle clouds at selected stages
    for sid in stage_ids:
        xs = parasim[sid, :, ix]
        ys = parasim[sid, :, iy]
        ax.scatter(xs, ys, s=6, alpha=0.12)

    ax.set_title(f"Likelihood surface + SMC exploration: {nx} vs {ny}\n(black contour = invalid / -inf likelihood)")
    ax.set_xlabel(nx)
    ax.set_ylabel(ny)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.show()

    # ---- Plot: invalid mask + nonlinearity proxy (gradient magnitude)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    axs[0].imshow(
        (~Z_ok).astype(float),
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="auto"
    )
    axs[0].set_title(f"Invalid region mask: {nx} vs {ny}\n(1=invalid, 0=valid)")
    axs[0].set_xlabel(nx); axs[0].set_ylabel(ny)
    axs[0].grid(True, alpha=0.15)

    im2 = axs[1].imshow(
        Z_grad,
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="auto"
    )
    cb2 = plt.colorbar(im2, ax=axs[1])
    cb2.set_label("gradient magnitude proxy")
    axs[1].set_title(f"Nonlinearity proxy (|∇ loglik|): {nx} vs {ny}")
    axs[1].set_xlabel(nx); axs[1].set_ylabel(ny)
    axs[1].grid(True, alpha=0.15)

    plt.tight_layout()
    plt.show()

# ---------------------------
# 2) Posterior pair-density (weighted KDE) for same pairs
# ---------------------------
for (nx, ny) in pairs:
    ix, iy = name_to_idx[nx], name_to_idx[ny]
    w = wtsim[:, -1]; w = w / np.sum(w)
    x = parasim[-1, :, ix]
    y = parasim[-1, :, iy]

    # limits from weighted quantiles
    def wq(a, w, q):
        idx = np.argsort(a); a=a[idx]; w=w[idx]
        cw=np.cumsum(w)
        return np.interp(q, cw, a)

    xlim = (wq(x, w, 0.01), wq(x, w, 0.99))
    ylim = (wq(y, w, 0.01), wq(y, w, 0.99))

    gx, gy, Z = weighted_kde2d(x, y, w, gridsize=140, xlim=xlim, ylim=ylim)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="auto",
        norm=LogNorm(vmin=max(np.min(Z[Z>0]), 1e-12), vmax=np.max(Z))
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("posterior density (weighted KDE, log scale)")
    ax.scatter(x, y, s=6, alpha=0.10)

    # mark posterior mean point
    mx = np.sum(x * w); my = np.sum(y * w)
    ax.plot([mx], [my], marker="x", markersize=10, linewidth=0, label="posterior mean")

    ax.set_title(f"Posterior joint density: {nx} vs {ny}")
    ax.set_xlabel(nx); ax.set_ylabel(ny)
    ax.grid(True, alpha=0.15)
    ax.legend()
    plt.tight_layout()
    plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Optional smoothing
try:
    from scipy.ndimage import gaussian_filter
    HAVE_SMOOTH = True
except Exception:
    HAVE_SMOOTH = False

# --- Pull posterior particles + weights from final stage ---
theta = np.asarray(parasim[-1, :, :])
w = np.asarray(wtsim[:, -1]).copy()
w = np.clip(w, 0.0, np.inf)
w = w / (w.sum() + 1e-300)

param_names = [
    "tau","kappa","psi1","psi2","rA","piA","gammaQ",
    "rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"
]
npara = theta.shape[1]
assert npara == len(param_names)

# --- Helpers ---
def weighted_hist2d(x, y, w, bins=60, quant_clip=(0.002, 0.998), smooth_sigma=1.0):
    x = np.asarray(x); y = np.asarray(y); w = np.asarray(w)

    # clip tails for plot bounds
    if quant_clip is not None:
        qlo, qhi = quant_clip
        xlo, xhi = np.quantile(x, [qlo, qhi])
        ylo, yhi = np.quantile(y, [qlo, qhi])
        x = np.clip(x, xlo, xhi)
        y = np.clip(y, ylo, yhi)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=w, density=True)

    if HAVE_SMOOTH and smooth_sigma and smooth_sigma > 0:
        H = gaussian_filter(H, sigma=smooth_sigma)

    return H, xedges, yedges

def weighted_hist1d(x, w, bins=90, quant_clip=(0.002, 0.998), smooth_sigma=1.0):
    x = np.asarray(x); w = np.asarray(w)

    if quant_clip is not None:
        qlo, qhi = quant_clip
        xlo, xhi = np.quantile(x, [qlo, qhi])
        x = np.clip(x, xlo, xhi)

    H, edges = np.histogram(x, bins=bins, weights=w, density=True)

    if HAVE_SMOOTH and smooth_sigma and smooth_sigma > 0:
        H = gaussian_filter(H, sigma=smooth_sigma)

    centers = 0.5*(edges[:-1] + edges[1:])
    return H, centers, edges

# --- Refinement knobs ---
BINS_2D = 70          # more bins => more gradation (but needs enough particles)
BINS_1D = 120
SMOOTH_2D = 0.9       # gentle smoothing helps continuous-looking densities
SMOOTH_1D = 0.8
TAIL_CLIP = (0.002, 0.998)

# Choose ONE: "lognorm" (good) or "power" (also good)
NORM_MODE = "lognorm"   # "lognorm" or "power"
POWER_GAMMA = 0.35      # only used if NORM_MODE="power"

SHOW_UPPER_TRIANGLE = False

# --- Compute global vmin/vmax for consistent scale across all panels ---
vals = []
for i in range(npara):
    for j in range(npara):
        if i == j:
            continue
        if (j > i) and (not SHOW_UPPER_TRIANGLE):
            continue
        H, _, _ = weighted_hist2d(theta[:, j], theta[:, i], w,
                                  bins=BINS_2D, quant_clip=TAIL_CLIP, smooth_sigma=SMOOTH_2D)
        nz = H[H > 0]
        if nz.size:
            vals.append(nz)

vals = np.concatenate(vals) if len(vals) else np.array([1.0])

# Quantile clip for color scale: prevents 1-2 bins from saturating everything
vmin = np.quantile(vals, 0.02)
vmax = np.quantile(vals, 0.995)
vmin = max(vmin, 1e-12)

if NORM_MODE == "lognorm":
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cbar_label = "joint density (log scale)"
else:
    norm = mcolors.PowerNorm(gamma=POWER_GAMMA, vmin=vmin, vmax=vmax)
    cbar_label = f"joint density (PowerNorm gamma={POWER_GAMMA})"

# --- Plot matrix ---
figsize = (1.7*npara, 1.7*npara)
fig, axes = plt.subplots(npara, npara, figsize=figsize)

last_im = None
for i in range(npara):
    for j in range(npara):
        ax = axes[i, j]

        if i == j:
            dens, centers, _ = weighted_hist1d(theta[:, i], w, bins=BINS_1D,
                                               quant_clip=TAIL_CLIP, smooth_sigma=SMOOTH_1D)
            ax.plot(centers, dens)
            ax.set_yticks([])
            if i == npara - 1:
                ax.set_xlabel(param_names[i], fontsize=9)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=9)
            continue

        if (j > i) and (not SHOW_UPPER_TRIANGLE):
            ax.axis("off")
            continue

        H, xedges, yedges = weighted_hist2d(theta[:, j], theta[:, i], w,
                                            bins=BINS_2D, quant_clip=TAIL_CLIP, smooth_sigma=SMOOTH_2D)
        # Avoid all-zero panels crashing LogNorm
        H = np.maximum(H, 1e-300)

        last_im = ax.imshow(
            H.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            norm=norm
        )

        if i == npara - 1:
            ax.set_xlabel(param_names[j], fontsize=9)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.set_ylabel(param_names[i], fontsize=9)
        else:
            ax.set_yticks([])

# Shared colorbar
cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.65, pad=0.01)
cbar.set_label(cbar_label)

fig.suptitle("Posterior pairwise joint densities (refined)", y=0.995, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.985])
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

def _safe_get(name):
    return globals().get(name, None)

phi = _safe_get("tune").get("phi", None) if _safe_get("tune") is not None else _safe_get("phi")
ESSsim = _safe_get("ESSsim")
acptsim = _safe_get("acptsim")
wtsim = _safe_get("wtsim")
parasim = _safe_get("parasim")

# ---- 1) ESS vs phi ----
if phi is not None and ESSsim is not None and _safe_get("tune") is not None:
    N = int(_safe_get("tune")["npart"])
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(phi, ESSsim / N)
    ax.set_title("SMC diagnostic: ESS / N vs tempering level ϕ")
    ax.set_xlabel("ϕ")
    ax.set_ylabel("ESS / N")
    ax.grid(True, alpha=0.3)
    plt.show()
else:
    print("Skipping ESS plot: need tune['phi'] and ESSsim (and tune['npart']).")

# ---- 2) Acceptance rate vs phi ----
if phi is not None and acptsim is not None:
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(phi, acptsim)
    ax.set_title("SMC diagnostic: RW-MH acceptance rate vs tempering level ϕ")
    ax.set_xlabel("ϕ")
    ax.set_ylabel("acceptance rate")
    ax.grid(True, alpha=0.3)
    plt.show()
else:
    print("Skipping acceptance plot: need tune['phi'] and acptsim.")

# ---- 3) Weight degeneracy by stage: entropy / effective number of particles ----
# Uses weights at each phi stage: wtsim shape in your notebook is (npart, nphi)
if wtsim is not None and phi is not None:
    W = np.asarray(wtsim)
    # handle shape either (npart,nphi) or (nphi,npart)
    if W.shape[1] == len(phi):
        W_stage = W
    elif W.shape[0] == len(phi):
        W_stage = W.T
    else:
        W_stage = None

    if W_stage is not None:
        # normalize per stage
        Wn = W_stage / (W_stage.sum(axis=0, keepdims=True) + 1e-300)
        # entropy-based effective number: exp(H), scaled by N
        H = -np.sum(Wn * np.log(Wn + 1e-300), axis=0)
        Neff_entropy = np.exp(H)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(phi, Neff_entropy / Wn.shape[0])
        ax.set_title("SMC diagnostic: exp(entropy(weights)) / N vs ϕ")
        ax.set_xlabel("ϕ")
        ax.set_ylabel("entropy-effective N / N")
        ax.grid(True, alpha=0.3)
        plt.show()

        # also plot max weight per stage (simple degeneracy indicator)
        maxw = np.max(Wn, axis=0)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(phi, maxw)
        ax.set_title("SMC diagnostic: max normalized weight vs ϕ")
        ax.set_xlabel("ϕ")
        ax.set_ylabel("max weight")
        ax.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Skipping weight-degeneracy plots: wtsim shape not compatible with phi.")
else:
    print("Skipping weight-degeneracy plots: need wtsim and phi.")

# ---- 4) Optional: log-likelihood distribution by stage (if you stored it) ----
# Many SMC implementations store ll by stage; try common names:
for cand in ["llsim", "ll_stage", "loglh_sim", "loglh_store"]:
    arr = _safe_get(cand)
    if arr is not None and phi is not None:
        ll = np.asarray(arr)
        # Try to reshape to (nphi,npart)
        if ll.ndim == 2 and ll.shape[0] == len(phi):
            ll_stage = ll
        elif ll.ndim == 2 and ll.shape[1] == len(phi):
            ll_stage = ll.T
        else:
            ll_stage = None

        if ll_stage is not None:
            # show median and IQR by stage
            med = np.median(ll_stage, axis=1)
            q25 = np.quantile(ll_stage, 0.25, axis=1)
            q75 = np.quantile(ll_stage, 0.75, axis=1)

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(phi, med)
            ax.fill_between(phi, q25, q75, alpha=0.2)
            ax.set_title(f"SMC diagnostic: log-likelihood distribution vs ϕ (from '{cand}')")
            ax.set_xlabel("ϕ")
            ax.set_ylabel("log L")
            ax.grid(True, alpha=0.3)
            plt.show()
        else:
            print(f"Found '{cand}' but shape not understood; skipping.")
        break
else:
    print("No stored log-likelihood-by-stage array found (that's fine).")

#%%
import numpy as np
import matplotlib.pyplot as plt

# ---- Grab final-stage posterior particles + weights ----
theta = np.asarray(parasim[-1, :, :])          # (npart, npara)
w = np.asarray(wtsim[:, -1]).copy()           # (npart,)
w = np.clip(w, 0.0, np.inf)
w = w / (w.sum() + 1e-300)

param_names = [
    "tau","kappa","psi1","psi2","rA","piA","gammaQ",
    "rho_R","rho_g","rho_z","sigma_R","sigma_g","sigma_z"
]
npara = theta.shape[1]
assert npara == len(param_names), "param_names must match theta dimension"

def weighted_quantile(x, w, qs):
    x = np.asarray(x)
    w = np.asarray(w)
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws)
    cdf = cdf / (cdf[-1] + 1e-300)
    return np.interp(qs, cdf, xs)

# ---- 1) Marginal histograms grid ----
cols = 4
rows = int(np.ceil(npara / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 2.8*rows))
axes = np.atleast_1d(axes).reshape(rows, cols)

for k in range(npara):
    ax = axes[k // cols, k % cols]
    x = theta[:, k]

    # clip tails for readability (posterior only)
    lo, hi = np.quantile(x, [0.002, 0.998])
    xc = np.clip(x, lo, hi)

    ax.hist(xc, bins=45, weights=w, density=True, alpha=0.9)
    q16, q50, q84 = weighted_quantile(x, w, [0.16, 0.50, 0.84])
    ax.axvline(q50, linewidth=1.5)
    ax.axvline(q16, linewidth=1.0, linestyle="--")
    ax.axvline(q84, linewidth=1.0, linestyle="--")

    ax.set_title(f"{param_names[k]}\nmedian={q50:.4g}, 68%=[{q16:.4g},{q84:.4g}]", fontsize=10)

# turn off unused axes
for kk in range(npara, rows*cols):
    axes[kk // cols, kk % cols].axis("off")

fig.suptitle("Posterior marginals (final SMC stage), weighted hist + 16/50/84% quantiles", y=1.02)
fig.tight_layout()
plt.show()

# ---- 2) Weighted correlation matrix heatmap ----
# weighted covariance: Cov = sum_i w_i (x_i - mu)(x_i - mu)'
mu = np.sum(theta * w[:, None], axis=0)
Xc = theta - mu[None, :]
Cov = (Xc * w[:, None]).T @ Xc
std = np.sqrt(np.diag(Cov) + 1e-300)
Corr = Cov / (std[:, None] * std[None, :] + 1e-300)

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow(Corr, origin="lower", aspect="auto", vmin=-1, vmax=1)
ax.set_title("Posterior weighted correlation matrix (final stage)")
ax.set_xticks(range(npara))
ax.set_yticks(range(npara))
ax.set_xticklabels(param_names, rotation=90)
ax.set_yticklabels(param_names)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("correlation")
fig.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# ---- pick a representative posterior parameter vector (weighted mean) ----
theta_particles = np.asarray(parasim[-1, :, :])
w = np.asarray(wtsim[:, -1]).copy()
w = np.clip(w, 0.0, np.inf)
w = w / (w.sum() + 1e-300)
theta_hat = np.sum(theta_particles * w[:, None], axis=0)

# ---- build model matrices at theta_hat ----
T1, T0, GEV = model_solution(theta_hat)
if T1 is None or (np.asarray(GEV).ravel()[1] < 0.5):
    raise RuntimeError("model_solution(theta_hat) failed or diagnostics indicate no valid solution.")

A, B, H, R, Se, Phi = sysmat(T1, T0, theta_hat)
Y = np.asarray(data)  # expected shape (T, ny)
T = Y.shape[0]
ny = Y.shape[1]
nx = Phi.shape[0]
neps = Se.shape[0]

# ---- Kalman filter (returns filtered + predicted + innovations) ----
def kalman_filter(Y, A, B, H, Phi, R, Se,
                  x0=None, P0=None):
    T, ny = Y.shape
    nx = Phi.shape[0]

    if x0 is None:
        x0 = np.zeros(nx)
    if P0 is None:
        P0 = np.eye(nx) * 10.0  # diffuse-ish

    # storage
    x_pred = np.zeros((T, nx))
    P_pred = np.zeros((T, nx, nx))
    x_filt = np.zeros((T, nx))
    P_filt = np.zeros((T, nx, nx))
    y_pred = np.zeros((T, ny))
    innov  = np.zeros((T, ny))
    S_list = np.zeros((T, ny, ny))
    K_list = np.zeros((T, nx, ny))

    x_prev = x0
    P_prev = P0

    Q = R @ Se @ R.T

    I = np.eye(nx)

    for t in range(T):
        # predict
        xtt1 = Phi @ x_prev
        Ptt1 = Phi @ P_prev @ Phi.T + Q

        # predicted obs
        ytt1 = (A[:, 0] + B @ xtt1)
        vt = Y[t] - ytt1

        S = B @ Ptt1 @ B.T + H
        # stabilize inversion
        Sinv = np.linalg.pinv(S)
        K = Ptt1 @ B.T @ Sinv

        # update
        x_new = xtt1 + K @ vt
        P_new = (I - K @ B) @ Ptt1

        # store
        x_pred[t] = xtt1
        P_pred[t] = Ptt1
        x_filt[t] = x_new
        P_filt[t] = P_new
        y_pred[t] = ytt1
        innov[t]  = vt
        S_list[t] = S
        K_list[t] = K

        x_prev, P_prev = x_new, P_new

    return x_pred, P_pred, x_filt, P_filt, y_pred, innov, S_list, K_list

# ---- RTS smoother for states ----
def rts_smoother(Phi, x_pred, P_pred, x_filt, P_filt, Q):
    T, nx = x_filt.shape
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)

    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T-2, -1, -1):
        # smoother gain
        C = P_filt[t] @ Phi.T @ np.linalg.pinv(P_pred[t+1])
        x_smooth[t] = x_filt[t] + C @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t+1] - P_pred[t+1]) @ C.T

    return x_smooth, P_smooth

# run filter + smoother
x_pred, P_pred, x_filt, P_filt, y_pred, innov, S_list, K_list = kalman_filter(Y, A, B, H, Phi, R, Se)
Q = R @ Se @ R.T
x_smooth, P_smooth = rts_smoother(Phi, x_pred, P_pred, x_filt, P_filt, Q)

# filtered and smoothed observables (no measurement noise)
y_filt = (A[:, 0][None, :] + (x_filt @ B.T))
y_smooth = (A[:, 0][None, :] + (x_smooth @ B.T))

# ---- Plot 1: observed vs filtered & smoothed ----
fig, axes = plt.subplots(ny, 1, figsize=(10, 2.8*ny), sharex=True)
if ny == 1:
    axes = [axes]
for i in range(ny):
    ax = axes[i]
    ax.plot(Y[:, i], label="observed")
    ax.plot(y_filt[:, i], label="filtered mean")
    ax.plot(y_smooth[:, i], label="smoothed mean")
    ax.set_title(f"Observable {i+1}: observed vs filtered/smoothed")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("time")
axes[0].legend()
fig.tight_layout()
plt.show()

# ---- Plot 2: standardized innovations (one-step-ahead residuals) ----
# Standardize by sqrt(diag(S_t)) from filter
std_innov = np.zeros_like(innov)
for t in range(T):
    sdiag = np.sqrt(np.maximum(np.diag(S_list[t]), 1e-300))
    std_innov[t] = innov[t] / sdiag

fig, axes = plt.subplots(ny, 1, figsize=(10, 2.4*ny), sharex=True)
if ny == 1:
    axes = [axes]
for i in range(ny):
    ax = axes[i]
    ax.plot(std_innov[:, i])
    ax.axhline(0, linewidth=1)
    ax.axhline(2, linestyle="--", linewidth=1)
    ax.axhline(-2, linestyle="--", linewidth=1)
    ax.set_title(f"Standardized innovation (obs {i+1})")
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("time")
fig.tight_layout()
plt.show()

# ---- Historical shock decomposition ----
# 1) infer structural shocks from smoothed states:
#    x_t - Phi x_{t-1} = R eps_t  => eps_t = argmin ||R eps - dx|| (least squares)
#    (This is a pragmatic decomposition; exact E[eps|y] would require an augmented smoother.)
dx = np.zeros((T, nx))
dx[0] = x_smooth[0]  # assume x_{-1}=0
for t in range(1, T):
    dx[t] = x_smooth[t] - Phi @ x_smooth[t-1]

# least squares solve for eps_t each t
eps_hat = np.zeros((T, neps))
Rpinv = np.linalg.pinv(R)
for t in range(T):
    eps_hat[t] = Rpinv @ dx[t]

# 2) propagate each shock's contribution through the transition to get state contributions
x_contrib = np.zeros((neps, T, nx))
for k in range(neps):
    for t in range(T):
        if t == 0:
            x_contrib[k, t] = R[:, k] * eps_hat[t, k]
        else:
            x_contrib[k, t] = Phi @ x_contrib[k, t-1] + R[:, k] * eps_hat[t, k]

# 3) map to observables (excluding constant A): y = A + B x
y_contrib = np.zeros((neps, T, ny))
for k in range(neps):
    y_contrib[k] = x_contrib[k] @ B.T  # (T,nx)@(nx,ny) -> (T,ny)

# Compare sum of contributions to smoothed (minus constant)
y_smooth_centered = y_smooth - A[:, 0][None, :]
y_sum = np.sum(y_contrib, axis=0)

# Plot: each observable gets its own decomposition panel (stacked lines; no explicit colors)
shock_labels = [f"shock {k+1}" for k in range(neps)]
fig, axes = plt.subplots(ny, 1, figsize=(10, 3.0*ny), sharex=True)
if ny == 1:
    axes = [axes]

for i in range(ny):
    ax = axes[i]
    for k in range(neps):
        ax.plot(y_contrib[k, :, i], label=shock_labels[k])
    ax.plot(y_smooth_centered[:, i], linewidth=2, label="smoothed (centered)")
    ax.plot(y_sum[:, i], linestyle="--", linewidth=2, label="sum(contrib)")
    ax.set_title(f"Historical decomposition (observable {i+1})")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("time")
axes[0].legend(ncol=2)
fig.tight_layout()
plt.show()

print("Decomposition notes:")
print("- 'smoothed (centered)' is (A removed): y_smooth - A.")
print("- eps_hat uses least-squares inversion of R on smoothed state increments; it's a practical decomposition.")
print("- If you want the exact E[eps_t|y] from an augmented smoother, I can provide that too.")

#%%
import numpy as np
import matplotlib.pyplot as plt

# Load data (T x 3): [output gap, inflation, interest rate]
data = np.loadtxt("us.txt")
T = data.shape[0]
t = np.arange(1, T+1)

y_gap = data[:, 0]
infl  = data[:, 1]
rate  = data[:, 2]

# 1) Output gap
plt.figure()
plt.plot(t, y_gap)
plt.title("Output gap (data[:,0])")
plt.xlabel("Time index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.show()

# 2) Inflation
plt.figure()
plt.plot(t, infl)
plt.title("Inflation (data[:,1])")
plt.xlabel("Time index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.show()

# 3) Interest rate
plt.figure()
plt.plot(t, rate)
plt.title("Interest rate (data[:,2])")
plt.xlabel("Time index")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.show()
