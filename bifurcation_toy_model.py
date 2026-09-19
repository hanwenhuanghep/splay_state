"""
bifurcation_toy_model.py

Implements companion-note direction 1, "Entropy collapse as a bifurcation
crossing" (interacting_langevin_summary.tex, Sec. "A promising direction:
training-aware attention temperature, and its RLHF analogue"), by first
fully solving its prerequisite, T3 ("A solvable two-particle toy model for
the critical temperature"), then extending it to n particles and to the
finite-depth / finite-noise regime a real Transformer actually lives in.

No trained model, no network access, no GPU: this is pure ODE/SDE analysis
and numerics (numpy/scipy/matplotlib only). The real-network comparison
(does OLMo-2's measured beta_hat_h sit near the resulting beta_c(n)?) is a
separate script, olmo2_bifurcation_comparison.py, since it needs the
checkpoints.

Part 1 -- n=2 particles on a circle, Q=K=V=I (T3 exactly as stated)
---------------------------------------------------------------------
T3 asks: with n=2 tokens on a circle at angle difference theta=theta_1-
theta_2 and Q=K=V=I, Eq. (attention-mf) collapses to a single autonomous
ODE theta_dot=f_beta(theta) with f_beta(0)=f_beta(pi)=0; solve for the
stability of theta=pi as a function of beta and identify the bifurcation
value beta_c at which it destabilizes.

Derivation. Self-attention with Q=K=V=I, self-attention included (a real
softmax row includes the query token itself), projected onto the circle's
tangent space at each point, gives for particle i

    X_dot^i = v^i - <v^i, X^i> X^i,  v^i = sum_j p_ij X^j,
    p_ij = softmax_j( beta <X^i, X^j> ) = exp(beta cos(theta_i-theta_j)) / Z^i.

Writing X^2 - cos(theta) X^1 = -sin(theta) T^1 (T^1 the unit tangent at
X^1, theta=theta_1-theta_2; elementary trigonometric identity, verified
numerically below) gives theta_dot_1 = -p_{1,2} sin(theta), and by the
i<->j symmetry theta_dot_2 = +p_{1,2} sin(theta) with the SAME p_{1,2}
(since <X^1,X^1>=<X^2,X^2>=1 and <X^1,X^2>=<X^2,X^1>=cos(theta)). Hence

    theta_dot = theta_dot_1 - theta_dot_2 = -2 p_{1,2} sin(theta)
              = -2 sin(theta) / (1 + exp(beta*(1-cos(theta))))
              =: f_beta(theta).

f_beta(0) = f_beta(pi) = 0 as required (both fixed points for every beta).
Linearizing at theta=pi: cos(pi+eps) = -1 + O(eps^2), sin(pi+eps) = -eps +
O(eps^3), so f_beta(pi+eps) = [2/(1+exp(2*beta))] * eps + O(eps^3). The
bracketed coefficient, lambda_2(beta) = 2/(1+exp(2*beta)), is POSITIVE for
every finite beta >= 0 -- theta=pi is unstable at every beta, not just
above some threshold. There is therefore no finite pitchfork bifurcation
for the noiseless, self-attention-included, Q=K=V=I system: this is the
n=2 special case of the rigorous global-collapse theorem (Geshkovski,
Letrouit, Polyanskiy & Rigollet, NeurIPS 2023), and lambda_2(beta) -> 0
exponentially as beta -> infinity is exactly the mechanism behind the
"exponentially long metastable plateau" result of the 2024 follow-up
(Geshkovski, Koubbi, Polyanskiy & Rigollet) cited in Sec. 9.3 of the
companion note: large beta does not stabilize separation, it just makes
escaping it exponentially slow.

Part 2 -- n > 2 particles: splay-state linearization
------------------------------------------------------
The natural n-particle generalization of theta=pi is the splay/incoherent
state theta_k = 2*pi*k/n (maximally spread out, zero mean resultant
vector). The full n-particle ODE is

    theta_dot_i = sum_j p_ij sin(theta_j - theta_i),
    p_ij = softmax_j( beta cos(theta_i - theta_j) ),

a beta-weighted (rather than uniformly-weighted) Kuramoto-type system: at
beta=0 this is exactly the classical identical-frequency Kuramoto model.
We linearize numerically (finite-difference Jacobian) at the splay state
and report lambda_max(n, beta), the largest eigenvalue orthogonal to the
model's built-in zero mode (uniform rotation theta_i -> theta_i + c, which
is an exact symmetry of the dynamics and always contributes one exact-zero
eigenvalue).

Part 3 -- the practically relevant, finite-quantity substitutes for beta_c
------------------------------------------------------------------------
A real Transformer runs a FINITE number of layers L, not t -> infinity, and
has real regularizing noise D (dropout / residual-stream noise, playing the
role of D in T2 of the companion note). Two well-defined, falsifiable
substitutes for "beta_c(n)" follow directly from Parts 1-2:

  (a) Finite-depth crossover beta_c(n, L): solve lambda_max(n, beta) = 1/L
      for beta. Below it, L layers of noiseless dynamics could not plausibly
      have escaped the dispersed/splay configuration within the network's
      own depth (e-folding time exceeds L); above it, they plausibly could.

  (b) Finite-noise order/disorder transition beta_c(n, D): reinstate the
      diffusion term of T2, dtheta_i = [drift] dt + sqrt(2D) dW_i, simulate
      by Euler-Maruyama for a grid of beta at fixed (n, D), and locate the
      beta at which the long-time participation ratio PR (the same T1
      diagnostic used throughout this pilot) rises sharply from its
      low-beta (dispersed) plateau to its high-beta (collapsed) plateau --
      a genuine, finite-beta phase transition, the discrete-n analogue of
      the sharp order/disorder transitions in beta now established for the
      mean-field noisy Transformer (arXiv:2606.05140, cited in the
      companion note's SOTA survey, Sec. 9.3).

Usage
-----
    python bifurcation_toy_model.py --out-dir bifurcation_results
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Part 1: n=2 closed form
# ---------------------------------------------------------------------------

def f_beta_closed_form(theta, beta):
    return -2.0 * np.sin(theta) / (1.0 + np.exp(beta * (1.0 - np.cos(theta))))


def lambda_2(beta):
    """Linearized instability rate of theta=pi, n=2 case."""
    return 2.0 / (1.0 + np.exp(2.0 * beta))


def two_particle_theta_dot_from_scratch(theta1, theta2, beta):
    """Ground-truth n=2 ODE built directly from the particle positions
    (softmax self-attention + tangential projection), with no use of the
    closed form above, to numerically verify f_beta_closed_form."""
    X1 = np.array([np.cos(theta1), np.sin(theta1)])
    X2 = np.array([np.cos(theta2), np.sin(theta2)])
    X = [X1, X2]
    thetas_dot = []
    for i in range(2):
        logits = np.array([beta * np.dot(X[i], X[j]) for j in range(2)])
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        v = p[0] * X[0] + p[1] * X[1]
        v_tangential = v - np.dot(v, X[i]) * X[i]
        T_i = np.array([-np.sin([theta1, theta2][i]), np.cos([theta1, theta2][i])])
        thetas_dot.append(np.dot(v_tangential, T_i))
    return thetas_dot[0] - thetas_dot[1]


def verify_n2_closed_form(betas, thetas):
    max_err = 0.0
    for beta in betas:
        for theta in thetas:
            theta1, theta2 = theta, 0.0
            lhs = two_particle_theta_dot_from_scratch(theta1, theta2, beta)
            rhs = f_beta_closed_form(theta, beta)
            max_err = max(max_err, abs(lhs - rhs))
    return max_err


# ---------------------------------------------------------------------------
# Part 2: general-n splay-state linearization
# ---------------------------------------------------------------------------

def n_particle_theta_dot(thetas, beta):
    """thetas: (n,) angles. Returns (n,) theta_dot_i = sum_j p_ij sin(theta_j-theta_i)."""
    n = thetas.shape[0]
    diff = thetas[None, :] - thetas[:, None]  # diff[i,j] = theta_j - theta_i
    logits = beta * np.cos(diff)  # cos(theta_j-theta_i)=cos(theta_i-theta_j)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)
    return (p * np.sin(diff)).sum(axis=1)


def splay_state(n):
    return 2.0 * np.pi * np.arange(n) / n


def jacobian_fd(func, x, eps=1e-6):
    n = x.shape[0]
    J = np.zeros((n, n))
    f0 = func(x)
    for k in range(n):
        xp = x.copy()
        xp[k] += eps
        J[:, k] = (func(xp) - f0) / eps
    return J


def lambda_max_splay(n, beta):
    """Largest eigenvalue of the Jacobian at the splay state, excluding the
    exact zero mode from the model's built-in rotational symmetry."""
    theta0 = splay_state(n)
    J = jacobian_fd(lambda t: n_particle_theta_dot(t, beta), theta0)
    eigvals = np.linalg.eigvals(J)
    eigvals = eigvals.real
    eigvals_sorted = np.sort(eigvals)[::-1]
    # drop exactly one zero eigenvalue (uniform-rotation symmetry mode)
    zero_idx = np.argmin(np.abs(eigvals_sorted))
    remaining = np.delete(eigvals_sorted, zero_idx)
    return remaining.max()


# ---------------------------------------------------------------------------
# Part 3a: finite-depth crossover beta_c(n, L)
# ---------------------------------------------------------------------------

def beta_c_finite_depth(n, L, lo=0.01, hi=50.0, hi_cap=1e5):
    """Solve lambda_max_splay(n, beta) = 1/L for beta via bisection, on the
    DECREASING (large-beta, self-attention-dominated freezing) branch of
    lambda_max(n, .). Since lambda_max(n, beta) is non-monotonic (it first
    rises as beta starts to resolve nearest neighbors on the splay ring,
    then falls once beta is large enough that each particle attends almost
    only to itself), we search from a large beta downward for the first
    sign change, rather than bisecting an arbitrary fixed bracket that may
    miss it for larger n."""
    target = 1.0 / L

    def g(beta):
        return lambda_max_splay(n, beta) - target

    b = hi
    # expand hi until g(hi) < 0 (sub-threshold / frozen regime reached)
    while g(b) > 0 and b < hi_cap:
        b *= 2
    hi = b
    if g(lo) * g(hi) > 0:
        return None
    return brentq(g, lo, hi, xtol=1e-4)


# ---------------------------------------------------------------------------
# Part 3b: finite-noise order/disorder transition via Euler-Maruyama + PR
# ---------------------------------------------------------------------------

def participation_ratio_2d(points):
    """points: (n, 2) unit vectors. Same T1 diagnostic used throughout the
    pilot, applied to the toy model's particle positions."""
    mean = points.mean(axis=0, keepdims=True)
    dev = np.linalg.norm(points - mean, axis=1)
    n = points.shape[0]
    num = dev.sum() ** 2
    den = n * (dev ** 2).sum()
    if den < 1e-12:
        return 1.0
    return float(num / den)


def kuramoto_r(theta):
    """Classical Kuramoto synchronization order parameter r=|mean_j e^{i
    theta_j}| in [0,1]: r near 1 means the particles sit in one tight
    angular cluster; r near 0 means they are spread around the whole
    circle. Used as a second, independent cross-check on PR below -- PR and
    r measure different things (PR: how EVENLY the n particles share the
    total deviation from their own centroid; r: how tightly they cluster in
    absolute angle) and a genuine transition should move both, coherently,
    in the physically expected directions."""
    return float(np.abs(np.mean(np.exp(1j * theta))))


def simulate_noisy(n, beta, D, T=40.0, dt=0.01, seed=0, n_chains=8):
    """Euler-Maruyama simulation of the noisy n-particle system starting
    from the splay state, plus a small random perturbation. Returns
    per-chain time-averages (over the last quarter of the run) of PR and of
    the Kuramoto order parameter r, as two (n_chains,) arrays."""
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    burn_in = int(0.75 * n_steps)
    pr_chain = np.zeros(n_chains)
    r_chain = np.zeros(n_chains)
    for c in range(n_chains):
        theta = splay_state(n) + 0.05 * rng.standard_normal(n)
        prs, rs = [], []
        for step in range(n_steps):
            drift = n_particle_theta_dot(theta, beta)
            noise = np.sqrt(2 * D * dt) * rng.standard_normal(n)
            theta = theta + drift * dt + noise
            if step >= burn_in:
                pts = np.stack([np.cos(theta), np.sin(theta)], axis=1)
                prs.append(participation_ratio_2d(pts))
                rs.append(kuramoto_r(theta))
        pr_chain[c] = np.mean(prs)
        r_chain[c] = np.mean(rs)
    return pr_chain, r_chain


def _mean_and_sem(chain_vals):
    m = float(np.mean(chain_vals))
    sem = float(np.std(chain_vals, ddof=1) / np.sqrt(len(chain_vals)))
    return m, sem


def find_empirical_beta_c(n, D, beta_grid, **sim_kwargs):
    """Sweep beta_grid, return per-beta (mean, SEM) of PR and of the
    Kuramoto r, plus a located PR crossing point (midpoint-crossing of the
    low/high PR plateaus). SEM (not the raw per-chain std) is reported,
    since the quantity of interest is the precision of the estimated mean
    curve, which shrinks with more chains -- unlike the per-chain std,
    which reflects chain-to-chain variability and need not shrink."""
    pr_means, pr_sems, r_means, r_sems = [], [], [], []
    for beta in beta_grid:
        pr_chain, r_chain = simulate_noisy(n, beta, D, **sim_kwargs)
        m, s = _mean_and_sem(pr_chain)
        pr_means.append(m)
        pr_sems.append(s)
        m, s = _mean_and_sem(r_chain)
        r_means.append(m)
        r_sems.append(s)
    pr_means = np.array(pr_means)
    low_plateau = pr_means[:2].mean()
    high_plateau = pr_means[-2:].mean()
    mid = 0.5 * (low_plateau + high_plateau)
    beta_c = None
    for i in range(len(beta_grid) - 1):
        if (pr_means[i] - mid) * (pr_means[i + 1] - mid) <= 0 and pr_means[i] != pr_means[i + 1]:
            b0, b1 = beta_grid[i], beta_grid[i + 1]
            p0, p1 = pr_means[i], pr_means[i + 1]
            beta_c = b0 + (mid - p0) * (b1 - b0) / (p1 - p0)
            break
    return (pr_means, np.array(pr_sems), np.array(r_means), np.array(r_sems),
            beta_c, low_plateau, high_plateau)


def convergence_study(n, D, beta_grid, configs):
    """Re-run find_empirical_beta_c at several (n_chains, T) settings on the
    SAME beta grid, to show how the noisy order/disorder transition (PR and
    Kuramoto r vs. beta) sharpens as sampling effort increases -- i.e. that
    the transition is a real, increasingly well-resolved feature and not an
    artifact of a particular low-statistics run. configs: list of
    (n_chains, T) tuples. Returns a dict keyed by (n_chains, T)."""
    out = {}
    for i, (n_chains, T) in enumerate(configs):
        (pr_means, pr_sems, r_means, r_sems, beta_c,
         low_plateau, high_plateau) = find_empirical_beta_c(
            n, D, beta_grid, T=T, dt=0.01, n_chains=n_chains, seed=1000 + i
        )
        out[(n_chains, T)] = dict(
            pr_means=pr_means, pr_sems=pr_sems, r_means=r_means, r_sems=r_sems,
            beta_c=beta_c, low_plateau=low_plateau, high_plateau=high_plateau,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="bifurcation_results")
    parser.add_argument("--skip-convergence", action="store_true",
                         help="Skip Part 3c (chains/T convergence study); saves several minutes.")
    parser.add_argument("--n-values", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16, 20])
    parser.add_argument("--depths", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--noise-D", type=float, default=0.15)
    parser.add_argument("--noise-n", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    # --- Part 1: verify n=2 closed form ---
    betas_check = np.linspace(0.1, 10, 8)
    thetas_check = np.linspace(0.1, 2 * np.pi - 0.1, 12)
    max_err = verify_n2_closed_form(betas_check, thetas_check)
    results["part1_n2_closed_form_max_abs_error"] = max_err
    results["part1_lambda_2_beta_0"] = lambda_2(0.0)
    results["part1_lambda_2_beta_5"] = lambda_2(5.0)
    results["part1_lambda_2_beta_20"] = lambda_2(20.0)
    print(f"[Part 1] n=2 closed-form vs. from-scratch ODE, max abs error: {max_err:.3e}")
    print(f"[Part 1] lambda_2(0)={lambda_2(0):.4f}  lambda_2(5)={lambda_2(5):.4e}  lambda_2(20)={lambda_2(20):.4e}")
    print("[Part 1] lambda_2(beta) > 0 for all beta tested: theta=pi is unstable at every finite beta"
          " (no noiseless finite-n=2 bifurcation) -- consistent with the rigorous global-collapse theorem.")

    # --- Part 2: lambda_max(n, beta) grid ---
    beta_grid_part2 = np.array([0.1, 0.5, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50])
    lambda_grid = {}
    for n in args.n_values:
        row = [lambda_max_splay(n, b) for b in beta_grid_part2]
        lambda_grid[n] = row
        print(f"[Part 2] n={n:2d}  lambda_max(beta): " + ", ".join(f"{v:.3e}" for v in row))
    results["part2_beta_grid"] = beta_grid_part2.tolist()
    results["part2_lambda_max_by_n"] = {str(n): v for n, v in lambda_grid.items()}

    # --- Part 3a: finite-depth crossover beta_c(n, L) ---
    beta_c_depth = {}
    for L in args.depths:
        beta_c_depth[L] = {}
        for n in args.n_values:
            bc = beta_c_finite_depth(n, L)
            beta_c_depth[L][n] = bc
            print(f"[Part 3a] L={L:2d} n={n:2d}  beta_c(n,L) = {bc}")
    results["part3a_beta_c_finite_depth"] = {
        str(L): {str(n): v for n, v in d.items()} for L, d in beta_c_depth.items()
    }

    # --- Part 3b: finite-noise order/disorder transition ---
    # Finer grid around the transition (beta~3-5) than elsewhere, 48 chains
    # of length T=100 (vs. the earlier 16 chains of T=60) so SEM error bars
    # are tight enough to show the crossing clearly, and both PR and the
    # Kuramoto order parameter r are tracked as an internal cross-check.
    beta_grid_part3b = np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                                  4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0])
    (pr_means, pr_sems, r_means, r_sems, beta_c_noise,
     low_plateau, high_plateau) = find_empirical_beta_c(
        args.noise_n, args.noise_D, beta_grid_part3b, T=80.0, dt=0.01, n_chains=48
    )
    print(f"[Part 3b] n={args.noise_n} D={args.noise_D}: PR(beta)=" + ", ".join(f"{v:.3f}" for v in pr_means))
    print(f"[Part 3b] PR SEM(beta)=" + ", ".join(f"{v:.4f}" for v in pr_sems))
    print(f"[Part 3b] Kuramoto r(beta)=" + ", ".join(f"{v:.3f}" for v in r_means))
    print(f"[Part 3b] Kuramoto r SEM(beta)=" + ", ".join(f"{v:.4f}" for v in r_sems))
    print(f"[Part 3b] low-beta plateau PR={low_plateau:.3f}, high-beta plateau PR={high_plateau:.3f}, "
          f"empirical beta_c={beta_c_noise}")
    results["part3b_noise_D"] = args.noise_D
    results["part3b_n"] = args.noise_n
    results["part3b_beta_grid"] = beta_grid_part3b.tolist()
    results["part3b_pr_means"] = pr_means.tolist()
    results["part3b_pr_sems"] = pr_sems.tolist()
    results["part3b_r_means"] = r_means.tolist()
    results["part3b_r_sems"] = r_sems.tolist()
    results["part3b_low_plateau"] = low_plateau
    results["part3b_high_plateau"] = high_plateau
    results["part3b_empirical_beta_c"] = beta_c_noise

    # --- Part 3c: does the transition sharpen with more chains / longer T? ---
    conv_results = None
    if not args.skip_convergence:
        beta_grid_conv = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0])
        configs = [(4, 20.0), (8, 40.0), (16, 60.0), (48, 80.0)]
        conv_results = convergence_study(args.noise_n, args.noise_D, beta_grid_conv, configs)
        for (n_chains, T), d in conv_results.items():
            print(f"[Part 3c] chains={n_chains:3d} T={T:5.1f}  PR(beta)=" +
                  ", ".join(f"{v:.3f}" for v in d["pr_means"]) +
                  f"  beta_c={d['beta_c']}")
        results["part3c_beta_grid"] = beta_grid_conv.tolist()
        results["part3c_configs"] = {
            f"chains{n_chains}_T{T}": {
                "pr_means": d["pr_means"].tolist(), "pr_sems": d["pr_sems"].tolist(),
                "beta_c": d["beta_c"],
            }
            for (n_chains, T), d in conv_results.items()
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- plots ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        thetas_plot = np.linspace(0, 2 * np.pi, 400)
        for beta in [0.0, 1.0, 3.0, 8.0]:
            axes[0].plot(thetas_plot, f_beta_closed_form(thetas_plot, beta), label=f"$\\beta$={beta}")
        axes[0].axhline(0, color="gray", lw=0.5)
        axes[0].set_xlabel(r"$\theta$")
        axes[0].set_ylabel(r"$f_\beta(\theta)$")
        axes[0].set_title("T3: n=2 drift (Part 1)")
        axes[0].legend(fontsize=8)

        for n in args.n_values:
            axes[1].plot(beta_grid_part2, lambda_grid[n], marker="o", label=f"n={n}", markersize=3)
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$\beta$")
        axes[1].set_ylabel(r"$\lambda_{\max}(n,\beta)$")
        axes[1].set_title("Part 2: splay-state instability rate")
        axes[1].legend(fontsize=7, ncol=2)

        l1 = axes[2].errorbar(beta_grid_part3b, pr_means, yerr=pr_sems, marker="o",
                               color="tab:blue", label="PR")
        if beta_c_noise is not None:
            axes[2].axvline(beta_c_noise, color="red", ls="--", label=fr"$\hat\beta_c$={beta_c_noise:.2f}")
        axes[2].set_xlabel(r"$\beta$")
        axes[2].set_ylabel("long-time PR", color="tab:blue")
        axes[2].tick_params(axis="y", labelcolor="tab:blue")
        axes[2].set_title(f"Part 3b: noisy transition (n={args.noise_n}, D={args.noise_D})")

        ax2b = axes[2].twinx()
        l2 = ax2b.errorbar(beta_grid_part3b, r_means, yerr=r_sems, marker="s",
                            color="tab:orange", label="Kuramoto $r$")
        ax2b.set_ylabel("Kuramoto $r$", color="tab:orange")
        ax2b.tick_params(axis="y", labelcolor="tab:orange")

        lines = [l1, l2]
        labels = ["PR", "Kuramoto $r$"]
        if beta_c_noise is not None:
            from matplotlib.lines import Line2D
            lines.append(Line2D([0], [0], color="red", ls="--"))
            labels.append(fr"$\hat\beta_c$={beta_c_noise:.2f}")
        axes[2].legend(lines, labels, fontsize=7, loc="center right")

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "bifurcation_toy_model.png"), dpi=150)
        plt.close(fig)

        # --- separate figure: does the transition sharpen with more sampling? ---
        if conv_results is not None:
            fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))

            colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(conv_results)))
            for (n_chains, T), color in zip(conv_results, colors):
                d = conv_results[(n_chains, T)]
                axes2[0].errorbar(beta_grid_conv, d["pr_means"], yerr=d["pr_sems"],
                                   marker="o", markersize=4, color=color,
                                   label=f"chains={n_chains}, T={T:.0f} "
                                         f"(N={n_chains*T:.0f})")
            axes2[0].axvline(beta_c_noise, color="red", ls=":", lw=1,
                              label=fr"final-run $\hat\beta_c$={beta_c_noise:.2f}")
            axes2[0].set_xlabel(r"$\beta$")
            axes2[0].set_ylabel("long-time PR")
            axes2[0].set_title(f"PR($\\beta$) at increasing sampling effort (n={args.noise_n})")
            axes2[0].legend(fontsize=7)

            # SEM at the grid point closest to the transition, vs. total
            # sampling effort chains*T, on a log-log scale, against a 1/sqrt(N)
            # reference line anchored at the smallest config.
            i_mid = int(np.argmin(np.abs(beta_grid_conv - beta_c_noise)))
            Ns = np.array([n_chains * T for (n_chains, T) in conv_results])
            sems_mid = np.array([conv_results[k]["pr_sems"][i_mid] for k in conv_results])
            order = np.argsort(Ns)
            Ns, sems_mid = Ns[order], sems_mid[order]
            axes2[1].loglog(Ns, sems_mid, "o-", color="tab:blue", label="observed SEM")
            ref = sems_mid[0] * np.sqrt(Ns[0] / Ns)
            axes2[1].loglog(Ns, ref, "k--", lw=1, label=r"$\propto N^{-1/2}$ reference")
            axes2[1].set_xlabel(r"sampling effort $N=$chains$\times T$")
            axes2[1].set_ylabel(fr"SEM of PR at $\beta={beta_grid_conv[i_mid]:.1f}$")
            axes2[1].set_title("Error shrinkage near the transition")
            axes2[1].legend(fontsize=8)

            fig2.tight_layout()
            fig2.savefig(os.path.join(args.out_dir, "bifurcation_convergence.png"), dpi=150)
            plt.close(fig2)
    except ImportError:
        pass

    print(f"\nWrote summary.json, bifurcation_toy_model.png"
          f"{', bifurcation_convergence.png' if conv_results is not None else ''} to {args.out_dir}")


if __name__ == "__main__":
    main()
