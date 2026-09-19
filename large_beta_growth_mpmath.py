"""
large_beta_growth_mpmath.py

Reproduces, with genuine arbitrary-precision arithmetic, the two large-beta
claims of Sec. 2.2 ("An exact closed form via the Bessel expansion") and
Sec. 3.3 ("Large-beta growth of the Bessel closed form") of
paper_A_splay_state_bessel.tex that plain double-precision floating point
(scipy.special.iv, as used in splay_state_bessel_closed_form.py) cannot
reach:

    lambda_p^USA(n, beta) = (1/beta) * [ sum_j (p+jn)^2 I_{p+jn}(beta)
                                        - sum_j (jn)^2   I_{jn}(beta) ].

At beta=800, I_k(beta) ~ e^800/sqrt(2*pi*800) ~ 10^345 for k well below
beta -- far beyond float64's ~1.8e308 ceiling -- and the formula itself is
a DIFFERENCE of two such huge, nearly-equal sums (they agree to ~340+
digits before the subtraction, since lambda itself is only O(beta) or a
sub-e^beta exponential), so naive floating point loses the entire answer to
cancellation even where it does not simply overflow to inf. This is exactly
the "catastrophic cancellation" remarked on in the paper.

Method
------
mpmath's mpf type has no fixed exponent range, so overflow is not an issue;
the real requirement is *decimal precision* (mp.workdps) large enough that
each I_k(beta) carries enough correct digits to survive the cancellation --
roughly beta/ln(10) + a safety buffer. Naively calling mpmath's own
mp.besseli(k, beta) independently for every needed k at that precision is
correct but very slow (each call redoes real work at high precision, and we
need O(beta) values of k). Instead we compute the WHOLE array
I_0(beta), ..., I_{nmax}(beta) in one pass via Miller's algorithm: seed an
arbitrary trial value far above nmax, recurse downward via the stable
three-term recurrence

    I_{k-1}(x) = I_{k+1}(x) + (2k/x) I_k(x),

(downward recurrence is the numerically stable direction for I_k, since I_k
is the *minimal* solution of this recurrence as k -> infinity -- forward
recurrence would blow up the arbitrary seed's error instead of damping it),
and fix the overall scale with a single accurate reference call to
mp.besseli(0, beta). This turns O(nmax) expensive independent Bessel
evaluations into one O(nmax) sweep of cheap arbitrary-precision
multiply-adds, several orders of magnitude faster, and was verified here to
reproduce mp.besseli(k, 800) to the full working precision (~450 digits)
at every k checked from 0 to nmax.

The remaining subtlety is where to TRUNCATE the (mathematically infinite)
sum over j. Because I_k(beta) decays only slowly while k is below beta and
does not become truly negligible on the *cancellation* scale until k is
comfortably above beta (empirically, nmax ~ 2*beta was not quite enough for
n=4, beta=800; nmax ~ 2.5*beta was), lambda_p_usa_mp below verifies
convergence explicitly by doubling nmax until the answer stops changing at
the working precision, rather than trusting a fixed rule of thumb.

Reproduces
----------
  1. n=4: lambda_2^USA(4, beta)/beta == 1 exactly (Corollary n2n4), checked
     at beta up to 800 (paper: "lambda_2^USA(4,beta)/beta=1.0000 at every
     beta tested").
  2. n=6, 8, 10, 12: the p=n/2 mode grows exponentially; fitting
     ln(lambda_{n/2}^USA(n,beta)) by least squares over beta in [400, 800]
     (paper's Sec. 3.3) gives empirical rates that should match the paper's
     reported 0.502, 0.709, 0.811, 0.868.

Usage
-----
    python large_beta_growth_mpmath.py --out-dir splay_bessel_results
"""

import argparse
import json
import os

import mpmath as mp


# ---------------------------------------------------------------------------
# Arbitrary-precision Bessel array via Miller's (backward-recurrence) algorithm
# ---------------------------------------------------------------------------

def bessel_i_array_mp(nmax, x, dps, margin=None):
    """Returns [I_0(x), I_1(x), ..., I_nmax(x)] at dps decimal digits of
    working precision, via downward recurrence from an arbitrary seed above
    nmax, normalized by a single accurate reference call to I_0(x). margin
    controls how far above nmax the recurrence starts (bigger = more
    converged but a little more work); the default is generous."""
    with mp.workdps(dps):
        x = mp.mpf(x)
        if margin is None:
            margin = max(200, nmax // 2)
        start = nmax + margin
        f = [mp.mpf(0)] * (start + 2)
        f[start] = mp.mpf(1)
        for k in range(start, 0, -1):
            f[k - 1] = f[k + 1] + (mp.mpf(2 * k) / x) * f[k]
        scale = mp.besseli(0, x) / f[0]
        return [f[k] * scale for k in range(nmax + 1)]


def lambda_p_usa_mp(n, p, beta, dps, nmax_start=None, rel_tol=None, max_doublings=8):
    """Arbitrary-precision evaluation of the exact closed form
    lambda_p^USA(n, beta), with the summation range doubled until the
    result stops changing at the working precision (rel_tol), rather than
    trusting a fixed truncation rule -- see module docstring for why a
    fixed nmax ~ 2*beta silently gave a wrong answer for n=4, beta=800."""
    with mp.workdps(dps):
        beta_mp = mp.mpf(beta)
        if nmax_start is None:
            nmax_start = int(2.5 * float(beta)) + 50
        if rel_tol is None:
            rel_tol = mp.mpf(10) ** (-(dps - 30))
        nmax = nmax_start
        prev = None
        for _ in range(max_doublings):
            jmax = max(1, (nmax - p) // n)
            arr = bessel_i_array_mp(p + jmax * n, beta, dps)
            term_p = mp.mpf(0)
            term_0 = mp.mpf(0)
            for j in range(-jmax, jmax + 1):
                kp = p + j * n
                term_p += mp.mpf(kp) ** 2 * arr[abs(kp)]
                k0 = j * n
                term_0 += mp.mpf(k0) ** 2 * arr[abs(k0)]
            val = (term_p - term_0) / beta_mp
            if prev is not None and abs(val - prev) < rel_tol * max(abs(val), 1):
                return val
            prev = val
            nmax *= 2
        return val  # best estimate if max_doublings was somehow exhausted


def working_dps(beta, buffer=60):
    """Decimal precision needed for I_k(beta)-scale cancellation: I_k(beta)
    ~ e^beta/sqrt(2*pi*beta) has about beta/ln(10) decimal digits before the
    decimal point, and we want `buffer` correct digits left over after the
    cancellation in lambda_p_usa_mp."""
    return int(beta / mp.log(10)) + buffer


# ---------------------------------------------------------------------------
# Claim 2: exponential growth-rate fit
# ---------------------------------------------------------------------------

def fit_growth_rate(n, p, beta_lo, beta_hi, n_points=8, buffer=60):
    """Fits ln(lambda_p^USA(n,beta)) ~ rate*beta + const by ordinary least
    squares over beta in [beta_lo, beta_hi]. Returns (rate, betas, logs) as
    plain floats/lists (fine at this stage -- the arbitrary-precision work
    is already done inside lambda_p_usa_mp)."""
    betas = [beta_lo + (beta_hi - beta_lo) * i / (n_points - 1) for i in range(n_points)]
    logs = []
    for b in betas:
        val = lambda_p_usa_mp(n, p, b, working_dps(b, buffer))
        logs.append(float(mp.log(val)))
    N = len(betas)
    mean_b = sum(betas) / N
    mean_l = sum(logs) / N
    num = sum((b - mean_b) * (l - mean_l) for b, l in zip(betas, logs))
    den = sum((b - mean_b) ** 2 for b in betas)
    rate = num / den
    return rate, betas, logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="splay_bessel_results")
    parser.add_argument("--n4-betas", type=float, nargs="+",
                         default=[50, 100, 200, 400, 600, 800],
                         help="beta values at which to check the exact n=4 "
                              "ratio lambda_2^USA(4,beta)/beta == 1.")
    parser.add_argument("--growth-n", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--growth-beta-range", type=float, nargs=2, default=[400.0, 800.0])
    parser.add_argument("--growth-n-points", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    print("=== Claim 1: n=4, lambda_2^USA(4,beta) = beta exactly (Corollary n2n4) ===")
    print(f"(reproducing the paper's 'lambda_2^USA(4,beta)/beta = 1.0000 at every "
          f"beta tested, out to beta=800', via arbitrary-precision arithmetic)")
    ratios = []
    for beta in args.n4_betas:
        dps = working_dps(beta)
        val = lambda_p_usa_mp(4, 2, beta, dps)
        ratio = float(val / beta)
        ratios.append(ratio)
        print(f"  beta={beta:6.1f}  dps={dps:4d}  lambda_2^USA(4,beta)={float(val):.6e}  "
              f"ratio/beta={ratio:.12f}")
    max_dev = max(abs(r - 1.0) for r in ratios)
    print(f"  max |ratio - 1| over beta in {args.n4_betas}: {max_dev:.3e}")
    results["n4_betas"] = args.n4_betas
    results["n4_ratio_to_beta"] = ratios
    results["n4_max_abs_deviation_from_1"] = max_dev

    print(f"\n=== Claim 2: n={args.growth_n}, exponential growth rate of the p=n/2 mode ===")
    beta_lo, beta_hi = args.growth_beta_range
    print(f"fitting ln(lambda_{{n/2}}^USA(n,beta)) over beta in [{beta_lo:.0f}, {beta_hi:.0f}] "
          f"({args.growth_n_points} points, arbitrary precision)")
    reported = {6: 0.502, 8: 0.709, 10: 0.811, 12: 0.868}
    rates = {}
    fit_data = {}
    for n in args.growth_n:
        p = n // 2
        rate, betas, logs = fit_growth_rate(n, p, beta_lo, beta_hi, args.growth_n_points)
        rates[n] = rate
        fit_data[n] = {"betas": betas, "log_lambda": logs}
        ref = reported.get(n)
        ref_str = f" (paper reports {ref})" if ref is not None else ""
        print(f"  n={n:2d} (p={p}): fitted rate = {rate:.4f}{ref_str}")
    results["large_beta_growth_rates"] = {str(n): r for n, r in rates.items()}
    results["large_beta_growth_fit_data"] = {
        str(n): d for n, d in fit_data.items()
    }
    results["paper_reported_rates"] = {str(n): r for n, r in reported.items() if n in rates}

    with open(os.path.join(args.out_dir, "large_beta_growth_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- plot ---
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        axes[0].axhline(1.0, color="gray", lw=0.8, ls=":")
        axes[0].plot(args.n4_betas, ratios, "o-", color="tab:blue")
        axes[0].set_xlabel(r"$\beta$")
        axes[0].set_ylabel(r"$\lambda_2^{\mathrm{USA}}(4,\beta)/\beta$")
        axes[0].set_title(f"n=4: exact linear growth (max dev {max_dev:.1e})")
        pad = max(1e-9, max_dev * 3)
        axes[0].set_ylim(1 - pad, 1 + pad)

        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(args.growth_n)))
        for n, color in zip(args.growth_n, colors):
            d = fit_data[n]
            axes[1].plot(d["betas"], d["log_lambda"], "o", color=color, markersize=4,
                         label=f"n={n} (rate={rates[n]:.3f})")
            rate = rates[n]
            b0, l0 = d["betas"][0], d["log_lambda"][0]
            fit_line = [rate * (b - b0) + l0 for b in d["betas"]]
            axes[1].plot(d["betas"], fit_line, "-", color=color, lw=1)
        axes[1].set_xlabel(r"$\beta$")
        axes[1].set_ylabel(r"$\ln \lambda_{n/2}^{\mathrm{USA}}(n,\beta)$")
        axes[1].set_title("n=6,8,10,12: exponential growth-rate fit")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "large_beta_growth_mpmath.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print(f"\nWrote large_beta_growth_summary.json, large_beta_growth_mpmath.png to {args.out_dir}")


if __name__ == "__main__":
    main()
