"""
Cherenkov EBM — Parsing Layer Example
=======================================
Run this after: pip install -e .

Tests the parsing layer across 6 domains where LR provably fails.
"""

import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from cherenkov import CherenkovParsingLayer, normalize

np.random.seed(42)


def run_domain(name, X, y, n_shot=20):
    X = normalize(X)

    # Raw LR
    rng = np.random.RandomState(7)
    classes = np.unique(y)
    idx = np.concatenate([
        rng.choice(np.where(y==c)[0], min(n_shot,(y==c).sum()), replace=False)
        for c in classes])
    mask = np.zeros(len(y), bool); mask[idx] = True

    lr = LogisticRegression(max_iter=1000).fit(X[mask], y[mask])
    acc_lr = (lr.predict(X) == y).mean()

    # Parsing layer + LR
    layer = CherenkovParsingLayer(n_shot=n_shot, max_rounds=10)
    X_aug = layer.fit_transform(X, y)
    lr_aug = LogisticRegression(max_iter=1000).fit(X_aug[mask], y[mask])
    acc_aug = (lr_aug.predict(X_aug) == y).mean()

    delta = acc_aug - acc_lr
    print(f"  {name:20s}  LR={acc_lr:.1%}  LR+EBM={acc_aug:.1%}  "
          f"Δ={delta:+.1%}  fp={layer.fingerprint_[:3]}")
    return acc_lr, acc_aug


print("Cherenkov EBM — Parsing Layer Demo")
print("="*60)

# Circles: radial boundary, LR=~50%
Xc, yc = make_circles(600, noise=0.05, factor=0.5, random_state=42)
run_domain("Circles", Xc, yc)

# XOR: interaction boundary, LR=~50%
X_xor = np.random.randn(600, 2)
y_xor = ((X_xor[:,0]>0) ^ (X_xor[:,1]>0)).astype(int)
run_domain("XOR", X_xor + np.random.randn(600,2)*0.2, y_xor)

# Moons: smooth curved boundary, LR=~85%
Xm, ym = make_moons(600, noise=0.15, random_state=42)
run_domain("Moons", Xm, ym)

# Saddle: sign(x1*x2), LR=~50%
X_sad = np.random.randn(600, 5)
y_sad = ((X_sad[:,0]*X_sad[:,1]) > 0).astype(int)
run_domain("Saddle boundary", X_sad + np.random.randn(600,5)*0.2, y_sad)

# Cusp: catastrophe fold
X_cusp = np.random.uniform(-2,2,(600,4))
y_cusp = (X_cusp[:,0]**2 > X_cusp[:,1]**3 + 0.5).astype(int)
run_domain("Cusp boundary", X_cusp + np.random.randn(600,4)*0.1, y_cusp)

# Real data: Breast Cancer
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
run_domain("Breast Cancer", bc.data, bc.target)

print("\nDone.")
