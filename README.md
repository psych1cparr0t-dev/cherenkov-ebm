# Cherenkov EBM

**A geometric reasoning layer for navigation between linear regimes.**

The Cherenkov EBM is an energy-based model that functions as a plug-in nonlinear parsing layer for machine learning pipelines. It synthesizes named geometric primitives from a thirteen-family library — inspired by the Chinese Zodiac — directly from the residual structure of a linear baseline. No neural backbone, no MCMC sampling, no fixed catalogue.

The system is positioned as a *navigation layer between linear contexts* rather than a competitor to linear regression. The hypothesis is that real-world geometry is mostly piecewise linear, with a finite alphabet of named transitions stitching the pieces together. The thirteen-family library is our proposed alphabet.

---

## Quick start

```bash
pip install -e .
```

```python
from cherenkov import CherenkovParsingLayer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, noise=0.05, factor=0.5)

# Build the parsing layer on the residual structure
layer = CherenkovParsingLayer()
layer.fit(X, y)
X_augmented = layer.transform(X)

# Now logistic regression can separate concentric circles
clf = LogisticRegression().fit(X_augmented, y)
print(f"Accuracy: {clf.score(X_augmented, y):.1%}")
# Expected: ~98% (vs ~50% on raw X)
```

For the full reinforcement-learning extension (V1):

```python
from cherenkov.v1_qlearning import V1Agent

agent = V1Agent(seed=42)
agent.train(X_train, y_train, X_test, y_test, n_episodes=60)
```

---

## What's in this repository

| Path                                | Purpose                                             |
|-------------------------------------|-----------------------------------------------------|
| `src/cherenkov/`                    | The sklearn-compatible parsing layer package        |
| `src/cherenkov/v1_qlearning.py`     | Verisimilitude V1 — Q-learning over primitives      |
| `examples/parsing_layer_demo.py`    | Minimum working example                             |
| `examples/cherenkov_v43_reference.py` | Single-file reference implementation (no install) |
| `tests/test_smoke.py`               | Smoke tests                                         |
| `docs/preprint_v3.pdf`              | Technical preprint (v3.0, April 2026)               |
| `docs/reference_lists.md`           | Position in ML / quantum / EBM literature           |
| `validation/synthetic_*`            | Eight-domain synthetic benchmark results            |
| `validation/realworld_*`            | Real-world validation results                       |
| `validation/v1_realworld_results.json` | V1 results on tabular real-world data            |

---

## The thirteen-family primitive library

Each primitive is a parameterised function `φ(x; θ) → ℝ`. Families are named after the Chinese Zodiac for memorability.

| Family              | Animal   | Geometric meaning                                |
|---------------------|----------|--------------------------------------------------|
| radial              | Horse    | Isotropic Gaussian bump                          |
| directional         | Rooster  | Sigmoid along a learned direction                |
| oscillatory         | Snake    | Sinusoidal along an axis                         |
| torsional           | Dragon   | Angular winding around a center                  |
| crossing            | Goat     | Pairwise feature interaction                     |
| trend               | Ox       | Directed flow with lateral decay                 |
| acute               | Tiger    | Sharp wedge / acute-angle boundary               |
| obtuse              | Rabbit   | Wide wedge / obtuse-angle boundary               |
| wormhole            | Rat      | Catenoid throat (topological gap)                |
| saddle              | Monkey   | Bifurcation saddle (cusp catastrophe)            |
| limit_cycle         | Dog      | Closed-orbit ring (Hopf bifurcation)             |
| cusp                | Pig      | Cusp boundary (cusp catastrophe)                 |
| linear_detachment   | Axolotl  | Directed crossing of a topological gap (13th)    |

The thirteenth family — **linear detachment** — models a directed signal passing through a wormhole throat with an approach field, a gap, and an exit field rotated by a learned angle θ.

---

## Validated results

### Synthetic geometric benchmarks (8 domains, 5 seeds, 70/30 split, 20-shot)

| Domain         | LR(X)         | LR(X+EBM)     | Δ        |
|----------------|---------------|---------------|----------|
| Circles        | 51.3% ± 3.3%  | 97.7% ± 3.0%  | +46.4pp  |
| Moons          | 85.2% ± 2.3%  | 97.7% ± 1.6%  | +12.5pp  |
| XOR            | 50.4% ± 3.9%  | 82.8% ± 8.3%  | +32.4pp  |
| Saddle         | 55.9% ± 2.5%  | 84.9% ± 1.4%  | +29.1pp  |
| Cusp           | 79.6% ± 4.0%  | 87.5% ± 3.6%  | +7.9pp   |
| Iris 3-class   | 89.3% ± 5.7%  | 92.0% ± 4.1%  | +2.7pp   |
| Wine 3-class   | 96.3% ± 1.2%  | 96.7% ± 0.7%  | +0.4pp   |
| Breast Cancer  | 95.1% ± 3.0%  | 95.1% ± 3.0%  | +0.0pp   |
| **Mean**       | **75.4%**     | **89.8%**     | **+16.4pp** |

Six of eight domains improve. The two that don't (Wine, Breast Cancer) are near-linearly separable, and the parsing layer correctly produces no gain — consistent with its role as a navigation layer rather than a universal nonlinear approximator.

### Real-world tabular data (5 domains)

| Domain              | LR(X)         | LR(X+EBM)     | Δ        |
|---------------------|---------------|---------------|----------|
| Protein fold SCOP   | 48.3% ± 12.5% | 48.3% ± 12.5% | +0.0pp   |
| EEG eye-state       | 64.8% ± 1.7%  | 64.4% ± 1.6%  | -0.4pp   |
| Anneal phases       | 96.2% ± 1.1%  | 96.2% ± 1.1%  | +0.0pp   |
| Ionosphere radar    | 87.6% ± 3.2%  | 88.0% ± 3.0%  | +0.4pp   |
| DNA splice sites    | 82.3% ± 1.0%  | 82.3% ± 1.0%  | +0.0pp   |

Near-zero gain across all five real-world tabular domains. This is **consistent with**, not contradictory to, the central claim: most real-world tabular data is already linearly separable after standard preprocessing, and the parsing layer is not designed to compete with linear regression on linear data.

The next validation step — **brain wave signal under multiple cognitive conditions** — is the domain where the navigation-layer framing is testable. See V1 pass criterion below.

---

## V1 pass criterion (Verisimilitude V1)

A working Verisimilitude V1 model is one that satisfies four criteria simultaneously on a real-world domain:

1. **Within-agreement** — fingerprint is stable across seeds on the same data
2. **Across-agreement** — fingerprint is stable across related datasets (e.g., across subjects on the same task)
3. **Distinguishability** — fingerprint differs meaningfully across different conditions
4. **Lift** — augmented features improve held-out accuracy over the linear baseline

This operationalizes a working definition of truth as *convergent agreement on togetherness in a linear context*.

**Status:** V1 implementation is complete (`src/cherenkov/v1_qlearning.py`) and runs end-to-end. The pass criterion has not yet been satisfied on real-world data and is the next concrete validation milestone.

---

## How it differs from existing work

**vs. neural EBMs** (Du & Mordatch 2019, Grathwohl et al. 2020, EB-JEPA): no neural backbone. The energy function is a sum of named primitives. No Langevin sampling, no MCMC.

**vs. JEPA** (LeCun 2022): predictions live in *named* primitive space rather than learned latent space. The fingerprint is auditable.

**vs. kernel methods** (RBF networks, kernel SVMs, Gaussian processes): the primitive library is heterogeneous (13 functional forms, not 13 instances of one form). Selection is autonomous — no kernel hyperparameter tuning, no support-vector selection. The system tells you *which* geometric structure it found, not just *that* it improved fit.

---

## Citation

```bibtex
@misc{bradford2026cherenkov,
  author       = {Bradford, Max},
  title        = {Cherenkov EBM: A Geometric Reasoning Layer for Navigation Between Linear Regimes},
  year         = {2026},
  publisher    = {Cherenkov Inc.},
  howpublished = {\url{https://github.com/psych1cparr0t-dev/cherenkov-ebm}},
  note         = {v3.0 preprint, April 2026}
}
```

---

## License

MIT. See `LICENSE`.

---

## Contact

Max Bradford · Cherenkov Inc. · Keene, New Hampshire, USA
[max@cherenkov.industries](mailto:max@cherenkov.industries) · [cherenkov.industries](https://cherenkov.industries)
