# The Splay State's Critical-Temperature Instability in Self-Attention Dynamics

Code and manuscript for **"Critical Temperature and Strict-Saddle Instability of the Splay State in Self-Attention Dynamics."**

Mean-field analyses of self-attention model token representations across depth as an interacting particle system on a sphere, with clustering driven by an effective inverse temperature $\beta$. On the circle, Geshkovski, Letrouit, Polyanskiy & Rigollet (*A Mathematical Perspective on Transformers*, Bull. AMS 2025) pose as an open problem whether every critical point of the associated energy, other than the global maxima, is a strict saddle. This repository contains the paper that resolves this exactly for the maximally dispersed ("splay") configuration — via an exact Bessel-function/circulant-matrix solution, an exact bridge to real softmax-normalized self-attention, and extensions to the regular simplex, causal masking, and a full Transformer block with a feed-forward layer — together with every script needed to reproduce its numerical results and figures.

See `paper_A_splay_state_bessel.pdf` for the full manuscript (abstract, theorems, proofs, and discussion).

## Repository contents

```
paper_A_splay_state_bessel.tex     # manuscript source
paper_A_splay_state_bessel.pdf     # compiled manuscript
bifurcation_toy_model.py           # Part 1-3: n=2 exact drift, general-n numerics,
                                    #   finite-depth/finite-noise beta_c substitutes
splay_state_bessel_closed_form.py  # exact Bessel closed form, n=2,4 special cases,
                                    #   the SA/USA bridge, closed-form beta_c(n,L)
higher_dim_causal_extensions.py    # regular simplex, causal masking (n=2 exact and
                                    #   n>2 numerical), full Transformer + feed-forward
bifurcation_results/                # figures and summary.json produced by
                                    #   bifurcation_toy_model.py
splay_bessel_results/               # figures and summary.json produced by
                                    #   splay_state_bessel_closed_form.py
requirements.txt
```

## Reproducing the results

No trained model and no network access are required — every result in the paper is pure ODE/SDE analysis and numerics.

```bash
pip install -r requirements.txt

python bifurcation_toy_model.py --out-dir bifurcation_results
python splay_state_bessel_closed_form.py --out-dir splay_bessel_results
python higher_dim_causal_extensions.py
```

`bifurcation_toy_model.py --out-dir bifurcation_results` additionally runs a chains/`T` convergence study for the finite-noise order/disorder transition (Figure showing the crossover sharpening with more sampling); pass `--skip-convergence` to skip it for a faster run (this saves several minutes).

### What each script implements

- **`bifurcation_toy_model.py`** — the exact $n=2$ drift and its linearization; the general-$n$ splay-state instability rate $\lambda_{\max}(n,\beta)$ via finite-difference Jacobians; the finite-depth crossover $\beta_c(n,L)$; and the finite-noise order/disorder transition $\beta_c(n,D)$, tracked with two independent order parameters (the participation ratio and the classical Kuramoto synchronization parameter) plus the sampling-effort convergence study.
- **`splay_state_bessel_closed_form.py`** — the general Bessel-series closed form for the splay-state Jacobian eigenvalues, the exact $n=2,4$ special cases, the exact bridge from the unnormalized surrogate (USA) to real softmax-normalized self-attention (SA), and the closed-form evaluation of $\beta_c(n,L)$.
- **`higher_dim_causal_extensions.py`** — the regular-simplex equilibrium check and Riemannian Hessian spectrum on $\mathbb{S}^{d-1}$ ($d=3,\dots,9$); the exact causal-masking result for $n=2$ and a numerical at-splay diagnostic for $n>2$; and a full Transformer block (self-attention plus a token-wise feed-forward layer) showing that the low-temperature freezing mechanism is specific to self-attention and breaks down once a feed-forward nonlinearity is added.

## Requirements

- Python 3.9+
- `numpy`, `scipy`
- `matplotlib` (optional; only needed to regenerate the figures — the numerical results and console output do not require it)

## License

No license has been specified yet; please contact the author before reuse.
