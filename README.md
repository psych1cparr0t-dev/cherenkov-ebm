# Cherenkov EBM — Development Graphics

Max Bradford · Cherenkov Inc. · March 2026 · cherenkov.industries

---

Visual output from the development of the Cherenkov EBM, a geometric reasoning system that fits basis functions to residual structure to approximate nonlinear decision boundaries. Each chart is from a real experimental run. Results include failures and underperformance where they occurred.

Technical preprint at cherenkov.industries.

---

## Architecture

The system operates in three separate layers. The EBM receives the residual from the linear layer — it does not replace linear regression.

```
raw signal
    ↓
linear regression        (noise, baseline signal)
    ↓  residual
Cherenkov EBM            (geometric primitive synthesis)
    ↓
action / execution
```

---

## Images

### v0.x — Early experiments

| File | Description |
|------|-------------|
| `v0.1_-_First_results` | Initial prototype. Label misalignment bug present. Domain B: 13.3%, Domain C: 52.5%. |
| `v0.2_-_Improved_results` | Balanced few-shot sampling, class-conditional seeding. Domain B: 96.0%, +10pp over full-data logistic baseline. |
| `v0.3_-_Greebling_ablation` | Added high-frequency perturbations to source domain. Checkerboard Δ: +7.8pp. Manifold Resolution Principle noted. |
| `v0.4_-_Multiscale_library` | K=44, K/N=2.2. Performance collapse. K/N capacity constraint identified. |
| `v0.5_-_Master_benchmark` | K=20, K/N=1.0. First configuration to exceed full-data baseline on both domains. |
| `v0.5_-_Results` | Per-domain breakdown for v0.5. |
| `v0.5_-_Full_chart` | Full v0.5 experimental suite. |
| `v0.6_-_Geometric_seeding` | Five seeding strategies: Random, Class-Conditional, Helix, Fibonacci, K-Means. K-Means best on Moons (92.5%), Helix best on Checkerboard. |
| `v0.7_-_Library` | Extended primitive library. |
| `v0.8_-_Chess_EBM` | Chess position evaluation. Piece movement as basis functions. |
| `v0.9_-_Finance` | First financial domain test on synthetic VIX-calibrated data. |
| `Tier_1_baselines` | Comparison against NCC, 1-NN, 5-NN, SVM-RBF, Poly-LR. |

### v1.x — Backbone removed, adaptive capacity

| File | Description |
|------|-------------|
| `v1.0_-_Library` | Backbone removed. Pure library model: E(x) = Σ wᵢ·φᵢ(x). |
| `v1.1_-_Navigation` | EBM as navigation layer between input and action. |
| `v1.2_-_Pure_library_(no_backbone)` | No backbone. Checker 68.0%, Moons 92.0%. |
| `v1.3_-_Autonomous` | First autonomous selection loop, sequential version. Bugs present. |
| `v1.4_-_Adaptive_K` | K = max(2, ⌊N_few / N_types⌋). Checker 80.4%, Moons 97.6%, Rings 97.4%. |
| `v1.5_-_Curved_trend` | Added curved_trend primitive. Finance 69.5% at 50-shot. |

### v2.x — Autonomous selection

| File | Description |
|------|-------------|
| `v2.0_-_Autonomous_loop` | Best-of-round selection loop. Multi-round accumulation bug fixed. Checkerboard: product_periodic selected, +20.3pp. |
| `v2.1_-_Real_context_benchmarks` | Tests on synthetic VIX, SPY trend, BTC drawdown data. EBM underperforms LR on financial domains at 20-shot. |

### v3.x — Primitive generator

| File | Description |
|------|-------------|
| `v3.0_-_Generator` | No fixed catalogue. Primitives synthesized from residual. Rings +44.5pp, Pinwheel +32.8pp, Checkerboard +30.5pp over LR at 20-shot. |
| `v3.1_-_Real-world_domains` | Competitive research (4-class) and storefront logistics (6-class). Per-class fingerprints. |
| `v3.2_-_Cross-domain_transfer` | Medical triage trained, applied to construction risk. Zero shared features. Zero-shot beats random baseline. |
| `v3.2_-_Diagnostic_funnel` | Primitive families mapped to reasoning steps. Oscillatory and trend steps appear in both domains independently. |

---

## Results (20-shot vs logistic regression)

| Domain | LR | EBM v3.0 | Δ |
|--------|----|----------|---|
| Rings | 50.7% | 95.2% | +44.5pp |
| Pinwheel | 65.2% | 98.0% | +32.8pp |
| Checkerboard | 51.5% | 82.0% | +30.5pp |
| Moons | 84.8% | 98.0% | +13.2pp |
| Spirals | 65.0% | 85.7% | +20.7pp |
| Finance (50-shot) | 66.8% | 73.2% | +6.4pp |

---

## Status

This is preliminary work. The conjecture — that decision boundaries are decomposable into universal geometric primitives — is consistent with the results but not proven. The experiments use synthetic data. Real-world validation and a broader primitive family are needed.

Preprint: cherenkov.industries
Contact: max@cherenkov.industries
