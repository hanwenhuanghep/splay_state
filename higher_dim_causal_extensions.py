"""
higher_dim_causal_extensions.py

Two extensions of the circle-only Bessel/circulant analysis in
bifurcation_toy_model.py and splay_state_bessel_closed_form.py, implementing
Sec. 6 ("Two extensions beyond the circle") of paper_A_splay_state_bessel.tex.

Part 1 -- higher dimensions: the regular simplex
---------------------------------------------------
For n = d+1 particles on S^{d-1}, the regular simplex (all pairwise inner
products equal to -1/d) is the unique configuration invariant under the full
symmetric group S_n acting via its standard representation -- the natural
generalization of the circle's evenly-spaced splay state to d > 2, where no
generic closed-form "most dispersed" configuration exists otherwise.

We verify it is an equilibrium of the SA gradient flow (Proposition 6 of the
paper) and compute its Riemannian Hessian eigenvalues via an exact geodesic
(sphere exponential-map) perturbation and finite differences -- the same
finite-difference-Jacobian idea used throughout this project, generalized
from 1 angle/particle (the circle) to a (d-1)-dimensional tangent space per
particle. As a correctness check, the d=2 case (n=3 particles on the circle)
is required to reproduce the circle's own lambda_max(3, beta) exactly, since
3 points on S^1 forming a "regular 2-simplex" IS the circle's splay state.

Part 2 -- causal masking, exact for n=2
-------------------------------------------
With causal masking, particle i attends only to particles j <= i. For n=2
this is fully tractable by hand (particle 1's softmax is over a single key,
so its drift is identically 0) and gives an exact closed form, verified here
against a direct simulation.

Part 3 -- causal masking, numerical for n>2
-------------------------------------------
Particle 1 (index 0) attends only to itself for EVERY n, so theta_0 is
frozen for all time; this breaks the rotational symmetry the noncausal
technique relies on, and (per Sec. 2.4.2 of the paper) the splay state is
not an exact equilibrium of the causally-masked system for n>=3 -- indeed
numerical continuation from the splay state finds the true equilibrium
nearby is a degenerate, non-evenly-spaced configuration (later particles
collapsing exactly onto earlier ones). Rather than rely on that fragile,
non-generic branch, we report a diagnostic that is well-defined whether or
not splay is a fixed point: the residual drift magnitude and the spectral
abscissa (largest real eigenvalue of the finite-difference Jacobian) of the
causal drift evaluated AT the splay configuration itself, across a beta
sweep, checking whether both vanish as beta -> infinity as in the noncausal
case (Corollary 5 of the paper).

Part 4 -- the full Transformer with a feed-forward layer
-------------------------------------------
Eq. (2.8) of the Survey (Geshkovski, Letrouit, Polyanskiy & Rigollet 2025)
adds a token-wise feed-forward term w(t) sigma(a(t) x_i(t) + b(t)) to the
self-attention drift, tangentially projected together with the attention
term. We use a single head with Q=K=V=I (so the pure-attention part is
exactly the SA drift used throughout this paper) plus a small, fixed random
feed-forward network (tanh activation), scaled by a strength epsilon. A
generic feed-forward term of this form is not rotation-equivariant, so
(unlike pure attention) the splay state is no longer an exact equilibrium
once epsilon>0; we use the same at-splay diagnostic as Part 3 (residual
drift magnitude and Jacobian spectral abscissa, now for the combined
attention+feed-forward drift) across a beta sweep, at a fixed small
feed-forward strength, checking the same beta -> infinity vanishing
question.

Usage
-----
    python higher_dim_causal_extensions.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Part 1: regular simplex in d dimensions
# ---------------------------------------------------------------------------

def regular_simplex(n):
    """n points forming a regular (n-1)-simplex, represented in ambient R^n
    coordinates (each row lies in the (n-1)-dim sum-zero hyperplane, unit
    norm) -- a standard, simple construction avoiding an explicit R^{n-1}
    coordinate system."""
    X = np.eye(n) - np.ones((n, n)) / n
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def tangent_basis(x0, n, d_minus_1):
    """Orthonormal basis of the tangent space at x0 (unit vector in the
    sum-zero hyperplane of R^n), itself restricted to that hyperplane.
    Uses SVD (not plain QR) so the rank-deficient projection is handled
    correctly -- an earlier QR-based version of this function silently
    produced an incorrect basis direction and was caught by the d=2
    cross-check against the circle's own n=3 result."""
    H = np.eye(n) - np.ones((n, n)) / n
    U, S, Vt = np.linalg.svd(H)
    basis = U[:, : n - 1].T  # (n-1, n) orthonormal basis of the hyperplane
    coeffs = basis @ x0
    tangent = basis - np.outer(coeffs, x0)  # rank n-2
    U2, S2, Vt2 = np.linalg.svd(tangent, full_matrices=False)
    assert S2[d_minus_1] < 1e-8 * S2[0], f"unexpected rank: S2={S2}"
    return Vt2[:d_minus_1]  # (d_minus_1, n)


def sa_drift(X, beta):
    """SA drift (self-attention incl. self, tangentially projected), for
    particles given in ambient coordinates."""
    n = X.shape[0]
    logits = beta * (X @ X.T)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    V = P @ X
    dots = np.sum(V * X, axis=1, keepdims=True)
    return V - dots * X


def jacobian_at_simplex(n, beta, drift_fn=sa_drift, eps=1e-6):
    """Finite-difference Jacobian of the tangential drift at the regular
    simplex, in an orthonormal per-particle tangent coordinate system built
    from the exact sphere exponential map (so each finite-difference step
    stays exactly on every particle's sphere, no linearization error from
    the constraint itself -- only from the finite step size)."""
    X0 = regular_simplex(n)
    d_minus_1 = n - 2
    if d_minus_1 <= 0:
        return np.zeros((0, 0))
    bases = [tangent_basis(X0[k], n, d_minus_1) for k in range(n)]
    total_dof = n * d_minus_1

    def flat_to_X(theta_flat):
        theta = theta_flat.reshape(n, d_minus_1)
        Xnew = np.zeros_like(X0)
        for k in range(n):
            t = theta[k] @ bases[k]
            norm_t = np.linalg.norm(t)
            if norm_t < 1e-14:
                Xnew[k] = X0[k]
            else:
                Xnew[k] = np.cos(norm_t) * X0[k] + np.sin(norm_t) * (t / norm_t)
        return Xnew

    def velocity_flat(theta_flat):
        Xc = flat_to_X(theta_flat)
        drift = drift_fn(Xc, beta)
        out = np.zeros(n * d_minus_1)
        for k in range(n):
            out[k * d_minus_1 : (k + 1) * d_minus_1] = bases[k] @ drift[k]
        return out

    theta0 = np.zeros(total_dof)
    f0 = velocity_flat(theta0)
    assert np.max(np.abs(f0)) < 1e-8, f"regular simplex is not an equilibrium: max|f0|={np.max(np.abs(f0))}"
    J = np.zeros((total_dof, total_dof))
    for i in range(total_dof):
        tp = theta0.copy()
        tp[i] += eps
        J[:, i] = (velocity_flat(tp) - f0) / eps
    return J


def simplex_spectrum(n, beta):
    """Returns (n_zero_modes, sorted unique nonzero eigenvalues)."""
    J = jacobian_at_simplex(n, beta)
    eig = np.sort(np.linalg.eigvals(J).real)
    n_zero = int(np.sum(np.abs(eig) < 1e-4))
    nonzero = np.unique(np.round(eig[np.abs(eig) > 1e-4], 5))
    return n_zero, nonzero


# ---------------------------------------------------------------------------
# Part 2: causal masking, n=2 exact
# ---------------------------------------------------------------------------

def causal_two_particle_theta_dot(theta1, theta2, beta):
    """Ground-truth causal-masked n=2 ODE built directly from particle
    positions: particle 1 (earlier in sequence order) attends only to
    itself; particle 2 attends to both."""
    X1 = np.array([np.cos(theta1), np.sin(theta1)])
    X2 = np.array([np.cos(theta2), np.sin(theta2)])
    theta1_dot = 0.0  # softmax over a single key is always 1: no drift
    logits = np.array([beta * np.dot(X2, X1), beta * np.dot(X2, X2)])
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    v = p[0] * X1 + p[1] * X2
    v_tan = v - np.dot(v, X2) * X2
    T2 = np.array([-np.sin(theta2), np.cos(theta2)])
    theta2_dot = np.dot(v_tan, T2)
    return theta1_dot, theta2_dot


def g_beta_closed_form(theta, beta):
    """Eq. (causal-n2) of the paper: exactly f_beta(theta)/2."""
    return -np.sin(theta) / (1 + np.exp(beta * (1 - np.cos(theta))))


def f_beta_closed_form(theta, beta):
    """The noncausal n=2 closed form (bifurcation_toy_model.py)."""
    return -2 * np.sin(theta) / (1 + np.exp(beta * (1 - np.cos(theta))))


# ---------------------------------------------------------------------------
# Part 3: causal masking, numerical for n > 2
# ---------------------------------------------------------------------------

def splay_state(n):
    return 2.0 * np.pi * np.arange(n) / n


def causal_theta_dot_general(thetas, beta):
    """thetas: (n,) angles. Causal-masked SA drift: particle i attends only
    to particles j<=i (theta_dot[0] is identically 0)."""
    n = thetas.shape[0]
    diff = thetas[None, :] - thetas[:, None]  # diff[i,j] = theta_j - theta_i
    logits = beta * np.cos(diff)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)  # j>i: not visible
    logits = np.where(mask, -np.inf, logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(logits)
    p = np.where(mask, 0.0, p)
    p = p / p.sum(axis=1, keepdims=True)
    return (p * np.sin(diff)).sum(axis=1)


def causal_lambda_at_splay(n, beta, eps=1e-6):
    """Diagnostic for n>2, where the splay state is NOT an exact equilibrium
    of the causally-masked system (Sec. 2.4.2 of the paper): rather than
    solve for the true (generally non-evenly-spaced, and for n>=4 found by
    continuation to be a degenerate configuration with later particles
    collapsing exactly onto earlier ones) equilibrium, we evaluate the
    causal drift's own residual magnitude and the spectral abscissa (largest
    real eigenvalue of the finite-difference Jacobian) AT the splay
    configuration itself. This is an instantaneous separation-rate
    diagnostic (how fast nearby trajectories starting at the splay state
    pull apart right now), well-defined whether or not splay is a fixed
    point, and it reduces exactly to the usual lambda_max whenever the
    residual drift is itself negligible. Returns (lambda_max, max|drift|)."""
    theta0 = splay_state(n)
    f0 = causal_theta_dot_general(theta0, beta)
    J = np.zeros((n, n))
    for k in range(n):
        tp = theta0.copy()
        tp[k] += eps
        J[:, k] = (causal_theta_dot_general(tp, beta) - f0) / eps
    lam = np.linalg.eigvals(J).real.max()
    return lam, np.max(np.abs(f0))


# ---------------------------------------------------------------------------
# Part 4: full Transformer with a feed-forward layer (Eq. 2.8 of the Survey)
# ---------------------------------------------------------------------------

def make_random_ffn(ell=8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((ell, 2))
    b = rng.standard_normal(ell)
    w = rng.standard_normal((2, ell)) / np.sqrt(ell)
    return a, b, w


def full_theta_dot(thetas, beta, a, b, w, eps):
    """Single-head (Q=K=V=I) self-attention + a token-wise feed-forward
    term eps*w*sigma(a x_i + b), sigma=tanh, tangentially projected -- the
    circle-restricted analogue of Eq. (2.8) of the Survey with H=1."""
    n = thetas.shape[0]
    X = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (n,2)
    diff = thetas[None, :] - thetas[:, None]
    logits = beta * np.cos(diff)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    V_att = p @ X  # (n,2)
    ffn = eps * (np.tanh(X @ a.T + b[None, :]) @ w.T)  # (n,2)
    F = V_att + ffn
    T = np.stack([-np.sin(thetas), np.cos(thetas)], axis=1)  # (n,2), tangent
    return np.sum(F * T, axis=1)


def ffn_lambda_at_splay(n, beta, a, b, w, eps, jac_eps=1e-6):
    """Same diagnostic as causal_lambda_at_splay, for the full attention +
    feed-forward drift: a generic feed-forward term is not rotation
    equivariant, so the splay state is generally not an exact equilibrium
    once eps>0 (it is exact at eps=0). We evaluate the residual drift and
    the Jacobian's spectral abscissa AT the splay configuration for the
    fully turned-on feed-forward layer (eps), across a beta sweep. Returns
    (lambda_max, max|drift|)."""
    theta0 = splay_state(n)
    f0 = full_theta_dot(theta0, beta, a, b, w, eps)
    J = np.zeros((n, n))
    for k in range(n):
        tp = theta0.copy()
        tp[k] += jac_eps
        J[:, k] = (full_theta_dot(tp, beta, a, b, w, eps) - f0) / jac_eps
    lam = np.linalg.eigvals(J).real.max()
    return lam, np.max(np.abs(f0))


def main():
    print("=== Part 1: regular simplex ===")
    print("Cross-check: d=2 (n=3) must reproduce the circle's own splay-state result.")
    n_zero, eigs = simplex_spectrum(3, 0.5)
    print(f"  n=3, beta=0.5: n_zero={n_zero}, nonzero unique eigenvalues={eigs} "
          f"(expect a single value 0.6376, matching lambda_max(3,0.5) from bifurcation_toy_model.py)")

    print("\nTwo-eigenvalue structure across dimensions (Table in the paper):")
    for n in [4, 5, 7, 10]:
        for beta in [0.5, 1.5, 4.0]:
            n_zero, eigs = simplex_spectrum(n, beta)
            expected_zero = (n - 1) * (n - 2) // 2  # dim SO(d), d=n-1
            print(f"  n={n:2d} (d={n-1}) beta={beta:4.1f}: n_zero={n_zero:3d} "
                  f"(expect {expected_zero}), nonzero={eigs}")

    print("\nPositivity check across a wide beta range:")
    for n in [4, 5, 7, 10]:
        all_pos = True
        for beta in [0.05, 0.2, 1.0, 3.0, 8.0, 15.0, 25.0]:
            _, eigs = simplex_spectrum(n, beta)
            if len(eigs) and np.any(eigs < 0):
                all_pos = False
        print(f"  n={n}: all nonzero eigenvalues positive for beta in [0.05, 25]: {all_pos}")

    print("\n=== Part 2: causal masking, n=2 ===")
    max_err = 0.0
    for beta in np.linspace(0.1, 10, 8):
        for theta in np.linspace(0.1, 2 * np.pi - 0.1, 12):
            d1, d2 = causal_two_particle_theta_dot(theta, 0.0, beta)
            lhs = d1 - d2
            rhs = g_beta_closed_form(theta, beta)
            max_err = max(max_err, abs(lhs - rhs))
    print(f"  max abs error, causal simulation vs g_beta closed form: {max_err:.3e}")

    max_err2 = 0.0
    for beta in [0.5, 2.0, 5.0]:
        for theta in [0.7, 2.1, 4.5]:
            max_err2 = max(max_err2, abs(g_beta_closed_form(theta, beta) - f_beta_closed_form(theta, beta) / 2))
    print(f"  max abs error, g_beta vs f_beta/2 (exact halving): {max_err2:.3e}")

    print("\n=== Part 3: causal masking, n>2, numerical (diagnostic at the splay state) ===")
    betas_causal = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    for n in [3, 4, 5, 6, 8]:
        print(f"  n={n}:")
        for beta in betas_causal:
            lam, drift = causal_lambda_at_splay(n, beta)
            print(f"    beta={beta:7.3f}  lambda_max={lam:10.5f}  max|drift at splay|={drift:.5f}")

    print("\n=== Part 4: full Transformer with feed-forward layer (Eq. 2.8), diagnostic at the splay state ===")
    a, b, w = make_random_ffn(ell=8, seed=0)
    eps = 0.1
    betas_ffn = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    for n in [4, 6, 8]:
        print(f"  n={n} (epsilon={eps}):")
        for beta in betas_ffn:
            lam, drift = ffn_lambda_at_splay(n, beta, a, b, w, eps)
            print(f"    beta={beta:7.3f}  lambda_max={lam:10.5f}  max|drift at splay|={drift:.5f}")


if __name__ == "__main__":
    main()
