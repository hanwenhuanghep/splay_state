# The Splay State's Critical-Temperature Instability in Self-Attention Dynamics

Code and manuscript for **"Critical Temperature and Strict-Saddle Instability of the Splay State in Self-Attention Dynamics."**

Mean-field analyses of self-attention model token representations across depth as an interacting particle system on a sphere, with clustering driven by an effective inverse temperature $\beta$. On the circle, Geshkovski, Letrouit, Polyanskiy & Rigollet (*A Mathematical Perspective on Transformers*, Bull. AMS 2025) pose as an open problem whether every critical point of the associated energy, other than the global maxima, is a strict saddle. This repository contains the paper that resolves this exactly for the maximally dispersed ("splay") configuration — via an exact Bessel-function/circulant-matrix solution, an exact bridge to real softmax-normalized self-attention, and extensions to the regular simplex, causal masking, and a full Transformer block with a feed-forward layer — together with every script needed to reproduce its numerical results and figures.

## Repository contents

```
bifurcation_toy_model.py           # Part 1-3: n=2 exact drift, general-n numerics,
                                    #   finite-depth/finite-noise beta_c substitutes
splay_state_bessel_closed_form.py  # exact Bessel closed form, n=2,4 special cases,
                                    #   the SA/USA bridge, closed-form beta_c(n,L)
higher_dim_causal_extensions.py    # regular simplex, causal masking (n=2 exact and
                                    #   n>2 numerical), full Transformer + feed-forward
large_beta_growth_mpmath.py        # arbitrary-precision (mpmath) reproduction of the
                                    #   large-beta (out to beta=800) closed-form claims
bifurcation_results/                # figures and summary.json produced by
                                    #   bifurcation_toy_model.py
splay_bessel_results/               # figures and summary.json produced by
                                    #   splay_state_bessel_closed_form.py and
                                    #   large_beta_growth_mpmath.py
requirements.txt
```

## Reproducing the results

No trained model and no network access are required — every result in the paper is pure ODE/SDE analysis and numerics.

```bash
pip install -r requirements.txt

python bifurcation_toy_model.py --out-dir bifurcation_results
python splay_state_bessel_closed_form.py --out-dir splay_bessel_results
python higher_dim_causal_extensions.py
python large_beta_growth_mpmath.py --out-dir splay_bessel_results
```

`bifurcation_toy_model.py --out-dir bifurcation_results` additionally runs a chains/`T` convergence study for the finite-noise order/disorder transition (Figure showing the crossover sharpening with more sampling); pass `--skip-convergence` to skip it for a faster run (this saves several minutes).

### What each script implements

- **`bifurcation_toy_model.py`** — the exact $n=2$ drift and its linearization; the general-$n$ splay-state instability rate $\lambda_{\max}(n,\beta)$ via finite-difference Jacobians; the finite-depth crossover $\beta_c(n,L)$; and the finite-noise order/disorder transition $\beta_c(n,D)$, tracked with two independent order parameters (the participation ratio and the classical Kuramoto synchronization parameter) plus the sampling-effort convergence study.
- **`splay_state_bessel_closed_form.py`** — the general Bessel-series closed form for the splay-state Jacobian eigenvalues, the exact $n=2,4$ special cases, the exact bridge from the unnormalized surrogate (USA) to real softmax-normalized self-attention (SA), and the closed-form evaluation of $\beta_c(n,L)$.
- **`higher_dim_causal_extensions.py`** — the regular-simplex equilibrium check and Riemannian Hessian spectrum on $\mathbb{S}^{d-1}$ ($d=3,\dots,9$); the exact causal-masking result for $n=2$ and a numerical at-splay diagnostic for $n>2$; and a full Transformer block (self-attention plus a token-wise feed-forward layer) showing that the low-temperature freezing mechanism is specific to self-attention and breaks down once a feed-forward nonlinearity is added.
- **`large_beta_growth_mpmath.py`** — reproduces, with genuine arbitrary-precision arithmetic (an efficient Miller's-algorithm Bessel-array evaluation, not repeated independent high-precision Bessel calls), the two large-$\beta$ claims that double-precision floating point cannot reach: the $n=4$ exact ratio $\lambda_2^{\mathrm{USA}}(4,\beta)/\beta=1$ verified out to $\beta=800$, and the exponential growth rates fitted for $n=6,8,10,12$ over $\beta\in[400,800]$ (reproduces $0.502, 0.709, 0.811, 0.868$).

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `mpmath`
- `matplotlib` (optional; only needed to regenerate the figures — the numerical results and console output do not require it)

## License

No license has been specified yet; please contact the author before reuse.
