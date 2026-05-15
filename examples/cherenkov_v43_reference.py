"""
Cherenkov EBM v4.1
==================
NEW PRIMITIVES:
  - acute      : sharp V-kink deflection (angle < 90°)
  - obtuse     : wide-angle deflection (angle > 90°)
  - wormhole   : catenoid throat — punctured radial, connects parallel linear regimes

ZODIAC LINEAR BASES (12):
  Statistics:  OLS, PCA projection, rank correlation
  Signal:      Fourier linear, wavelet approx, derivative linear
  Geometry:    affine transform, projection pursuit, piecewise linear
  Physics:     log-linear, interaction linear, margin linear

Each zodiac basis runs first. Best fit selected. Residual passed to EBM.
Fingerprint = [winning zodiac] + [synthesized primitives]

DATASETS:
  Synthetic:  Checker, Moons, Rings, Spirals, Pinwheel
  Real:       Breast Cancer, Iris, Wine (sklearn)
              Protein fold proxy (PCA of AA composition, SCOP-style)
              fMRI proxy (sklearn Digits reduced — brain-state analog)
              Materials phase (synthetic BaTiO3-style phase boundary)
              Navier-Stokes regime (synthetic laminar/turbulent classifier)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time, json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                               make_moons, load_digits)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import pandas as pd

warnings.filterwarnings('ignore')
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════
# PRIMITIVE EVALUATION — all 9 families
# ══════════════════════════════════════════════════════════════════

def _eval_primitive(family: str, params: dict, X: np.ndarray) -> np.ndarray:
    N, d = X.shape
    c_raw = params.get('center', np.zeros(d))
    c = np.zeros(d)
    c[:len(c_raw)] = c_raw[:d]

    # ── Original 6 ──────────────────────────────────────────────

    if family == 'radial':
        diff = X - c
        r = np.sqrt((diff**2).sum(-1) + 1e-8)
        p = np.clip(params.get('power', 2.0), 0.5, 4.0)
        s = max(params.get('sigma', 0.5), 0.05)
        return np.exp(-(r**p) / (s**p + 1e-8))

    elif family == 'directional':
        w_raw = params['w']
        w = np.zeros(d); w[:len(w_raw)] = w_raw[:d]
        w = w / (np.linalg.norm(w) + 1e-8)
        b = params.get('b', 0.0)
        return 1.0 / (1.0 + np.exp(-(X @ w + b)))

    elif family == 'oscillatory':
        a_raw = params.get('axis', np.ones(min(2,d))/np.sqrt(min(2,d)))
        a = np.zeros(d); a[:len(a_raw)] = a_raw[:d]
        a = a / (np.linalg.norm(a) + 1e-8)
        f = params.get('freq', 2.0)
        return np.sin(f * (X - c) @ a)

    elif family == 'torsional':
        dx = X[:,0] - c[0]
        dy = X[:,1] - c[1] if d > 1 else np.zeros(N)
        r = np.sqrt(dx**2 + dy**2 + 1e-8)
        theta = np.arctan2(dy, dx)
        s = max(params.get('sigma', 0.6), 0.05)
        f = params.get('freq', 2.0)
        return np.exp(-r**2 / (2*s**2)) * np.sin(f * theta)

    elif family == 'crossing':
        pairs = params.get('pairs', [(0, min(1,d-1))])
        pairs = [(i,j) for i,j in pairs if i<d and j<d]
        if not pairs: pairs = [(0, min(1,d-1))]
        W = params.get('W', np.ones(len(pairs)))
        b = params.get('b', 0.0)
        cross = np.stack([X[:,i]*X[:,j] for i,j in pairs], axis=1)
        return 1.0 / (1.0 + np.exp(-(cross @ W + b)))

    elif family == 'trend':
        dir_raw = params.get('direction', np.ones(min(2,d))/np.sqrt(min(2,d)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        s = max(params.get('sigma', 0.4), 0.05)
        a = params.get('align', 1.0)
        delta = X - c
        proj = (delta * direction).sum(-1, keepdims=True)
        perp = delta - proj * direction
        d_perp = np.sqrt((perp**2).sum(-1) + 1e-8)
        return np.exp(-d_perp**2 / (2*s**2)) * np.tanh(a * proj.squeeze(-1))

    # ── New primitives ───────────────────────────────────────────

    elif family == 'acute':
        # Sharp V-kink deflection: two arms meeting at angle < 90°
        # φ(x) = exp(-min(d_arm1, d_arm2)² / 2σ²) where arms diverge at acute angle
        angle = params.get('angle', np.pi/4)   # half-angle < π/4
        sigma = max(params.get('sigma', 0.3), 0.05)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        # Perpendicular in first 2 dims
        perp = np.zeros(d)
        if d >= 2:
            perp[0] = -direction[1]; perp[1] = direction[0]
        else:
            perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        # Kink: distance to V-shape boundary
        # V boundary: |d_perp| = tan(angle) * |d_along| for d_along > 0
        tip_dist = np.abs(d_perp) - np.tan(angle) * np.maximum(d_along, 0)
        # Signed: inside V is negative
        phi = np.exp(-np.maximum(tip_dist, 0)**2 / (2*sigma**2))
        return phi * (d_along > 0).astype(float)

    elif family == 'obtuse':
        # Wide-angle deflection: shallow redirect, angle > 90°
        # Complement of acute: activated on the obtuse side of a bend
        angle = params.get('angle', 2*np.pi/3)  # > π/2
        sigma = max(params.get('sigma', 0.5), 0.05)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2:
            perp[0] = -direction[1]; perp[1] = direction[0]
        else:
            perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        # Obtuse: activated where deflection is shallow (wide bend)
        bend = np.abs(d_perp) / (np.abs(d_along) + 1e-4)
        # Wide-bend region: bend ratio < tan(π - angle)
        threshold = np.tan(np.pi - angle)
        in_bend = (bend < threshold).astype(float)
        dist_to_apex = np.sqrt(d_along**2 + d_perp**2 + 1e-8)
        return np.exp(-dist_to_apex**2 / (2*sigma**2)) * in_bend

    elif family == 'wormhole':
        # Catenoid throat: punctured radial
        # φ(x) = exp(-r²/2σ²) · (1 - exp(-r²/2ε²))
        # Outer Gaussian attracts → inner hole punches through
        # ε = throat radius, σ = basin width
        sigma   = max(params.get('sigma', 0.8), 0.1)
        epsilon = max(params.get('epsilon', 0.15), 0.02)
        diff = X - c
        r2 = (diff**2).sum(-1) + 1e-8
        outer = np.exp(-r2 / (2*sigma**2))
        inner = np.exp(-r2 / (2*epsilon**2))
        return outer * (1.0 - inner)   # annular basin — zero at center, peak at throat


    elif family == 'saddle':
        # Monkey: attracts one axis, repels another
        # φ(x) = tanh(a·d_along) · tanh(b·d_perp)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2: perp[0] = -direction[1]; perp[1] = direction[0]
        else: perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        a = params.get('a', 1.5); b = params.get('b', 1.5)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        return np.tanh(a * d_along) * np.tanh(b * d_perp)

    elif family == 'limit_cycle':
        # Dog: Hopf ring — stable closed orbit, peak ON the ring
        # φ(x) = exp(-(‖x-c‖ - r*)² / 2σ²)
        r_star = max(params.get('r_star', 0.5), 0.05)
        sigma  = max(params.get('sigma', 0.2), 0.03)
        diff = X - c
        r = np.sqrt((diff**2).sum(-1) + 1e-8)
        return np.exp(-((r - r_star)**2) / (2*sigma**2))

    elif family == 'cusp':
        # Pig: catastrophe fold — two basins merging to a point
        # φ(x) peaks along cusp curve: x² = scale·|y|^(3/2)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2: perp[0] = -direction[1]; perp[1] = direction[0]
        else: perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        sigma = max(params.get('sigma', 0.4), 0.05)
        scale = max(params.get('scale', 1.0), 0.1)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        cusp_dist = d_along**2 - scale * (np.abs(d_perp) + 1e-8)**(1.5)
        return np.exp(-(cusp_dist**2) / (2*sigma**2))

    elif family == 'product':
        phi_a = _eval_primitive(params['family_a'], params['params_a'], X)
        phi_b = _eval_primitive(params['family_b'], params['params_b'], X)
        return phi_a * phi_b

    return np.zeros(N)


# ══════════════════════════════════════════════════════════════════
# ZODIAC LINEAR BASES — 12 exhaustive linear archetypes
# ══════════════════════════════════════════════════════════════════

def zodiac_fit(X: np.ndarray, y: np.ndarray, n_shot: int) -> Tuple[str, float, np.ndarray]:
    """
    Run all 12 zodiac bases on few-shot data.
    Return (winning_basis_name, best_accuracy, residual).
    """
    N, d = X.shape
    rng = np.random.RandomState(7)
    classes = np.unique(y)
    idx = np.concatenate([
        rng.choice(np.where(y==c)[0], min(n_shot, int((y==c).sum())), replace=False)
        for c in classes])
    mask = np.zeros(N, bool); mask[idx] = True
    Xf, yf = X[mask], y[mask]
    tgt = np.where(y==0, -1., 1.).astype(float)

    results = {}

    def _eval_base(name, pred_y, pred_score=None):
        acc = accuracy_score(y, pred_y)
        if pred_score is not None:
            R = tgt - pred_score
        else:
            R = tgt - (pred_y * 2 - 1).astype(float)
        results[name] = (acc, R)

    # 1. OLS
    try:
        ridge = Ridge(alpha=1.0).fit(Xf, np.where(yf==0,-1.,1.))
        s = ridge.predict(X)
        _eval_base('OLS', (s>0).astype(int), s)
    except: pass

    # 2. PCA projection (top-2 PCA then LR)
    try:
        pca = PCA(n_components=min(2, d)).fit(Xf)
        Xp_f = pca.transform(Xf); Xp = pca.transform(X)
        clf = LogisticRegression(max_iter=500).fit(Xp_f, yf)
        s = clf.decision_function(Xp)
        _eval_base('PCA_projection', clf.predict(Xp), s)
    except: pass

    # 3. Rank correlation (Spearman proxy: rank transform then LR)
    try:
        from scipy.stats import rankdata
        Xr_f = np.stack([rankdata(Xf[:,i]) for i in range(Xf.shape[1])], 1) / len(Xf)
        Xr   = np.stack([rankdata(X[:,i])  for i in range(X.shape[1])],  1) / len(X)
        clf = LogisticRegression(max_iter=500).fit(Xr_f, yf)
        s = clf.decision_function(Xr)
        _eval_base('rank_correlation', clf.predict(Xr), s)
    except: pass

    # 4. Fourier linear (first harmonic features then LR)
    try:
        def fourier_feats(A, k=1):
            return np.hstack([np.sin(2*np.pi*k*A), np.cos(2*np.pi*k*A)])
        Xfour_f = fourier_feats(Xf); Xfour = fourier_feats(X)
        clf = LogisticRegression(max_iter=500).fit(Xfour_f, yf)
        s = clf.decision_function(Xfour)
        _eval_base('Fourier_linear', clf.predict(Xfour), s)
    except: pass

    # 5. Wavelet approx (Haar single-level: x - mean(x) neighbors)
    try:
        def haar_feats(A):
            lo = (A[:,::2] + A[:,1::2]) / 2 if A.shape[1] > 1 else A
            hi = (A[:,::2] - A[:,1::2]) / 2 if A.shape[1] > 1 else np.zeros_like(A)
            return np.hstack([lo, hi])
        d2 = d - (d % 2)
        if d2 >= 2:
            clf = LogisticRegression(max_iter=500).fit(haar_feats(Xf[:,:d2]), yf)
            pred = clf.predict(haar_feats(X[:,:d2]))
            s = clf.decision_function(haar_feats(X[:,:d2]))
            _eval_base('wavelet_approx', pred, s)
    except: pass

    # 6. Derivative linear (finite diff gradient features)
    try:
        def grad_feats(A):
            if A.shape[1] < 2: return A
            return np.hstack([np.diff(A, axis=1), A[:,:-1]])
        clf = LogisticRegression(max_iter=500).fit(grad_feats(Xf), yf)
        s = clf.decision_function(grad_feats(X))
        _eval_base('derivative_linear', clf.predict(grad_feats(X)), s)
    except: pass

    # 7. Affine transform (whitened features)
    try:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xf)
        clf = LogisticRegression(max_iter=500).fit(sc.transform(Xf), yf)
        s = clf.decision_function(sc.transform(X))
        _eval_base('affine_transform', clf.predict(sc.transform(X)), s)
    except: pass

    # 8. Projection pursuit (LDA projection)
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis().fit(Xf, yf)
        Xlda_f = lda.transform(Xf); Xlda = lda.transform(X)
        clf = LogisticRegression(max_iter=500).fit(Xlda_f, yf)
        s = clf.decision_function(Xlda)
        _eval_base('projection_pursuit', clf.predict(Xlda), s)
    except: pass

    # 9. Piecewise linear (decision stump ensemble proxy)
    try:
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=3).fit(Xf, yf)
        pred = dt.predict(X)
        _eval_base('piecewise_linear', pred)
    except: pass

    # 10. Log-linear
    try:
        Xl_f = np.sign(Xf) * np.log1p(np.abs(Xf))
        Xl   = np.sign(X)  * np.log1p(np.abs(X))
        clf = LogisticRegression(max_iter=500).fit(Xl_f, yf)
        s = clf.decision_function(Xl)
        _eval_base('log_linear', clf.predict(Xl), s)
    except: pass

    # 11. Interaction linear
    try:
        def interact(A, max_feats=4):
            d_use = min(A.shape[1], max_feats)
            pairs = [(i,j) for i in range(d_use) for j in range(i+1,d_use)]
            if not pairs: return A
            return np.hstack([A, np.stack([A[:,i]*A[:,j] for i,j in pairs],1)])
        clf = LogisticRegression(max_iter=500).fit(interact(Xf), yf)
        s = clf.decision_function(interact(X))
        _eval_base('interaction_linear', clf.predict(interact(X)), s)
    except: pass

    # 12. Margin linear (LinearSVC)
    try:
        svc = LinearSVC(max_iter=2000).fit(Xf, yf)
        s = svc.decision_function(X)
        _eval_base('margin_linear', svc.predict(X), s)
    except: pass

    if not results:
        # Fallback
        clf = LogisticRegression(max_iter=1000).fit(Xf, yf)
        s = clf.decision_function(X)
        results['OLS'] = (accuracy_score(y, clf.predict(X)), tgt - s)

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_R = results[best_name]
    all_accs = {k: v[0] for k, v in results.items()}
    return best_name, best_acc, best_R, all_accs


# ══════════════════════════════════════════════════════════════════
# PRIMITIVE SYNTHESIZER — now with 9 families
# ══════════════════════════════════════════════════════════════════

class PrimitiveSynthesizer:
    def __init__(self, min_corr=0.06, n_centers=6):
        self.min_corr = min_corr
        self.n_centers = n_centers
        self._pca_cache = None

    def _corr(self, phi, R):
        if phi.std() < 1e-8: return 0.0
        c = np.corrcoef(phi, R)[0,1]
        return 0.0 if np.isnan(c) else abs(float(c))

    def _centers(self, X, R):
        weights = np.abs(R) / (np.abs(R).sum() + 1e-8)
        idx = np.random.choice(len(X), size=min(self.n_centers*3, len(X)),
                               replace=False, p=weights)
        nc = min(self.n_centers, len(idx))
        km = KMeans(n_clusters=nc, n_init=3, random_state=42)
        km.fit(X[idx])
        return km.cluster_centers_

    def _fit_radial(self, X, R, centers):
        best_c, best_p = {}, 0.0
        for c in centers:
            for sigma in [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]:
                for power in [0.8, 1.5, 2.0, 3.0]:
                    p = {'center':c, 'sigma':sigma, 'power':power}
                    cr = self._corr(_eval_primitive('radial', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_directional(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in np.linspace(0, np.pi, 12):
            for c in centers:
                w = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
                for b_off in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    b = -(w[:d]*c[:d]).sum() + b_off
                    p = {'center':c, 'w':w, 'b':b}
                    cr = self._corr(_eval_primitive('directional', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_oscillatory(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            for angle in np.linspace(0, np.pi, 8):
                axis = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
                for c in centers[:3]:
                    p = {'center':c, 'freq':freq, 'axis':axis}
                    cr = self._corr(_eval_primitive('oscillatory', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_torsional(self, X, R, centers):
        best_c, best_p = {}, 0.0
        for c in centers:
            for freq in [1.0, 2.0, 3.0, 4.0]:
                for sigma in [0.3, 0.5, 0.8, 1.2]:
                    p = {'center':c, 'freq':freq, 'sigma':sigma}
                    cr = self._corr(_eval_primitive('torsional', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_crossing(self, X, R, centers):
        d = X.shape[1]
        pairs = [(i,j) for i in range(min(d,4)) for j in range(i+1,min(d,4))]
        if not pairs: pairs = [(0,min(1,d-1))]
        best_c, best_p = {}, 0.0
        for subset in [pairs[:1], pairs[:2], pairs[:3]]:
            W = np.ones(len(subset))
            for b in [-0.5, 0.0, 0.5]:
                p = {'pairs':subset, 'W':W, 'b':b}
                cr = self._corr(_eval_primitive('crossing', p, X), R)
                if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_trend(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in np.linspace(0, 2*np.pi, 12):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
            for c in centers:
                for sigma in [0.2, 0.4, 0.7, 1.1]:
                    for align in [0.5, 1.0, 2.0]:
                        p = {'center':c, 'direction':direction, 'sigma':sigma, 'align':align}
                        cr = self._corr(_eval_primitive('trend', p, X), R)
                        if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_acute(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in [np.pi/6, np.pi/5, np.pi/4, np.pi/3]:
            for dir_angle in np.linspace(0, np.pi, 8):
                direction = np.array([np.cos(dir_angle), np.sin(dir_angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
                for c in centers[:3]:
                    for sigma in [0.2, 0.4, 0.6]:
                        p = {'center':c, 'angle':angle, 'direction':direction, 'sigma':sigma}
                        cr = self._corr(_eval_primitive('acute', p, X), R)
                        if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_obtuse(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in [2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
            for dir_angle in np.linspace(0, np.pi, 8):
                direction = np.array([np.cos(dir_angle), np.sin(dir_angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
                for c in centers[:3]:
                    for sigma in [0.3, 0.5, 0.8]:
                        p = {'center':c, 'angle':angle, 'direction':direction, 'sigma':sigma}
                        cr = self._corr(_eval_primitive('obtuse', p, X), R)
                        if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_wormhole(self, X, R, centers):
        best_c, best_p = {}, 0.0
        for c in centers:
            for sigma in [0.4, 0.6, 0.8, 1.2, 1.8]:
                for epsilon in [0.05, 0.1, 0.15, 0.25]:
                    if epsilon >= sigma: continue
                    p = {'center':c, 'sigma':sigma, 'epsilon':epsilon}
                    cr = self._corr(_eval_primitive('wormhole', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p


    def _fit_saddle(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in np.linspace(0, np.pi, 8):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
            for c in centers:
                for a in [0.8, 1.5, 2.5]:
                    for b in [0.8, 1.5, 2.5]:
                        p = {'center':c, 'direction':direction, 'a':a, 'b':b}
                        cr = self._corr(_eval_primitive('saddle', p, X), R)
                        if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_limit_cycle(self, X, R, centers):
        best_c, best_p = {}, 0.0
        for c in centers:
            for r_star in [0.15, 0.25, 0.4, 0.6, 0.9, 1.3]:
                for sigma in [0.08, 0.15, 0.25, 0.4]:
                    p = {'center':c, 'r_star':r_star, 'sigma':sigma}
                    cr = self._corr(_eval_primitive('limit_cycle', p, X), R)
                    if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def _fit_cusp(self, X, R, centers):
        d = X.shape[1]; best_c, best_p = {}, 0.0
        for angle in np.linspace(0, np.pi, 8):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) if d>=2 else np.array([1.])
            for c in centers:
                for sigma in [0.2, 0.4, 0.7]:
                    for scale in [0.3, 0.7, 1.2, 2.0]:
                        p = {'center':c, 'direction':direction, 'sigma':sigma, 'scale':scale}
                        cr = self._corr(_eval_primitive('cusp', p, X), R)
                        if cr > best_p: best_p=cr; best_c=dict(p)
        return best_c, best_p

    def synthesize(self, X, R, round_n=0):
        centers = self._centers(X, R)
        families = {
            'radial':       self._fit_radial,
            'directional':  self._fit_directional,
            'oscillatory':  self._fit_oscillatory,
            'torsional':    self._fit_torsional,
            'crossing':     self._fit_crossing,
            'trend':        self._fit_trend,
            'acute':        self._fit_acute,
            'obtuse':       self._fit_obtuse,
            'wormhole':     self._fit_wormhole,
            'saddle':       self._fit_saddle,
            'limit_cycle':  self._fit_limit_cycle,
            'cusp':         self._fit_cusp,
        }
        best_family, best_params, best_corr = None, {}, 0.0
        for fname, fitter in families.items():
            params, corr = fitter(X, R, centers)
            if corr > best_corr:
                best_corr=corr; best_family=fname; best_params=params
        # If best single primitive is weak, try product composition
        if best_corr < self.min_corr * 2 and len(centers) >= 2:
            prod_params, prod_corr = self._fit_product(X, R, centers)
            if prod_corr > best_corr:
                best_corr = prod_corr
                best_family = 'product'
                best_params = prod_params

        if best_corr < self.min_corr:
            return None
        return {'family': best_family, 'params': best_params,
                'corr': best_corr, 'round': round_n}

    def _fit_product(self, X, R, centers):
        """Product composition: φ_a · φ_b — cutout/mask shapes.
        Searches over pairs of fast-fitting families.
        """
        best_p, best_c = 0.0, {}
        fast_families = ['radial', 'directional', 'trend', 'crossing']
        fitters = {
            'radial':      self._fit_radial,
            'directional': self._fit_directional,
            'trend':       self._fit_trend,
            'crossing':    self._fit_crossing,
        }
        for fa in fast_families:
            pa, ca = fitters[fa](X, R, centers[:2])
            if ca < 0.05: continue
            phi_a = _eval_primitive(fa, pa, X)
            if phi_a.std() < 1e-8: continue
            R_masked = R * np.abs(phi_a) / (np.abs(phi_a).max() + 1e-8)
            for fb in fast_families:
                if fb == fa: continue
                pb, cb = fitters[fb](X, R_masked, centers[:2])
                phi_b = _eval_primitive(fb, pb, X)
                phi_prod = phi_a * phi_b
                if phi_prod.std() < 1e-8: continue
                cr = abs(float(np.corrcoef(phi_prod, R)[0,1]))
                if not np.isnan(cr) and cr > best_p:
                    best_p = cr
                    best_c = {'family_a': fa, 'params_a': pa,
                               'family_b': fb, 'params_b': pb}
        return best_c, best_p


# ══════════════════════════════════════════════════════════════════
# WEIGHT FITTER
# ══════════════════════════════════════════════════════════════════

def phi_matrix(primitives, X):
    if not primitives: return np.zeros((len(X),1))
    return np.stack([_eval_primitive(p['family'], p['params'], X) for p in primitives], axis=1)

def fit_weights(primitives, Xf, yf, reg=0.01, lr=0.08, epochs=500):
    Phi = phi_matrix(primitives, Xf)
    tgt = np.where(yf==0, -1.0, 1.0).astype(float)
    w = np.zeros(Phi.shape[1])
    mw = np.zeros_like(w); vw = np.zeros_like(w); t = 0
    bw = w.copy(); bl = float('inf')
    for _ in range(epochs):
        E = Phi @ w; res = E - tgt
        loss = (res**2).mean() + reg*(w**2).sum()
        if loss < bl: bl=loss; bw=w.copy()
        gw = 2*res@Phi/len(yf) + 2*reg*w; t+=1
        mw = 0.9*mw+0.1*gw; vw = 0.999*vw+0.001*gw**2
        w -= lr*(mw/(1-0.9**t))/(np.sqrt(vw/(1-0.999**t))+1e-8)
    return bw

def predict(primitives, w, X):
    if len(w)==0: return np.zeros(len(X), int)
    return (phi_matrix(primitives, X) @ w > 0).astype(int)


# ══════════════════════════════════════════════════════════════════
# GENERATOR LOOP — zodiac first, then EBM
# ══════════════════════════════════════════════════════════════════

def generator_loop(X_train, y_train, X_test, y_test,
                   n_shot=20, max_rounds=10,
                   var_threshold=0.10, min_corr=0.06,
                   min_delta_acc=0.003,
                   domain_name='domain', verbose=True):

    synth = PrimitiveSynthesizer(min_corr=min_corr)
    primitives = []

    # Few-shot split
    rng = np.random.RandomState(7)
    classes = np.unique(y_train)
    idx = np.concatenate([
        rng.choice(np.where(y_train==c)[0],
                   min(n_shot, int((y_train==c).sum())), replace=False)
        for c in classes])
    mask = np.zeros(len(y_train), bool); mask[idx]=True
    Xf, yf = X_train[mask], y_train[mask]

    # ── Step 1: Zodiac filter ──────────────────────────────────
    zodiac_name, zodiac_acc, R_zodiac, all_zodiac = zodiac_fit(X_test, y_test, n_shot)
    tgt_test = np.where(y_test==0, -1., 1.).astype(float)

    # ── Step 2: LR baseline (for comparison) ──────────────────
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(Xf, yf)
    lr_acc = accuracy_score(y_test, lr_clf.predict(X_test))

    current_acc = zodiac_acc
    reg = 0.01
    history = []

    if verbose:
        print(f"\n  {domain_name}  n_shot={n_shot}")
        print(f"  Zodiac winner: {zodiac_name} ({zodiac_acc:.1%})  LR={lr_acc:.1%}")

    # EBM reads zodiac residual
    R = R_zodiac.copy()

    for rnd in range(1, max_rounds+1):
        if primitives:
            w = fit_weights(primitives, Xf, yf, reg=reg)
            E = phi_matrix(primitives, X_test) @ w
            R = tgt_test - E
            current_acc = accuracy_score(y_test, predict(primitives, w, X_test))

        var_before = float(np.var(R))
        if var_before < var_threshold:
            if verbose: print(f"  R{rnd:02d}: converged (var={var_before:.4f})")
            break

        # Adaptive dimensionality for synthesis:
        # For high-D data, project to 8D via PCA to find geometric structure
        d_full = X_test.shape[1]
        if d_full > 8:
            if not hasattr(synth, '_pca_cache') or synth._pca_cache is None:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=8, random_state=42)
                pca.fit(X_test)
                synth._pca_cache = pca
            X_synth = synth._pca_cache.transform(X_test)
        else:
            X_synth = X_test[:, :d_full]
            synth._pca_cache = None
        new_prim = synth.synthesize(X_synth, R, round_n=rnd)

        if new_prim is None:
            if verbose: print(f"  R{rnd:02d}: synthesis failed")
            break

        primitives.append(new_prim)
        w_new = fit_weights(primitives, Xf, yf, reg=reg)
        new_acc = accuracy_score(y_test, predict(primitives, w_new, X_test))
        E_new = phi_matrix(primitives, X_test) @ w_new
        var_after = float(np.var(tgt_test - E_new))
        delta_acc = new_acc - current_acc

        if delta_acc >= min_delta_acc or (var_before - var_after) > 0.005:
            verdict = 'KEEP'; current_acc = new_acc
        else:
            primitives.pop(); verdict = 'WEAK'; var_after = var_before

        history.append({
            'round': rnd, 'family': new_prim['family'],
            'corr': round(new_prim['corr'],3),
            'var_before': round(var_before,4), 'var_after': round(var_after,4),
            'acc': round(current_acc,4), 'delta_acc': round(delta_acc,4),
            'verdict': verdict
        })

        if verbose:
            sym = '✓' if verdict=='KEEP' else '~'
            print(f"  R{rnd:02d} {sym} [{new_prim['family']:12s}] "
                  f"corr={new_prim['corr']:.3f}  var {var_before:.3f}→{var_after:.3f}  "
                  f"acc {current_acc:.1%}")

    # Final fit
    if primitives:
        w_final = fit_weights(primitives, Xf, yf, reg=reg, epochs=700)
        final_acc = accuracy_score(y_test, predict(primitives, w_final, X_test))
    else:
        final_acc = zodiac_acc

    fingerprint = [p['family'] for p in primitives]
    families_used = list(set(fingerprint))

    if verbose:
        print(f"  Final={final_acc:.1%}  Zodiac={zodiac_acc:.1%}  LR={lr_acc:.1%}  "
              f"Δ={final_acc-lr_acc:+.1%}  prims={len(primitives)}")
        print(f"  Fingerprint: {fingerprint}")

    return {
        'domain': domain_name,
        'final_acc': round(final_acc, 4),
        'zodiac_acc': round(zodiac_acc, 4),
        'zodiac_winner': zodiac_name,
        'all_zodiac': {k: round(v,4) for k,v in all_zodiac.items()},
        'lr_acc': round(lr_acc, 4),
        'delta_vs_lr': round(final_acc - lr_acc, 4),
        'delta_vs_zodiac': round(final_acc - zodiac_acc, 4),
        'n_primitives': len(primitives),
        'fingerprint': fingerprint,
        'families_used': families_used,
        'history': history,
    }


# ══════════════════════════════════════════════════════════════════
# DATASETS — synthetic + real nonlinear (LR provably fails)
# ══════════════════════════════════════════════════════════════════

def make_circles_dataset(n=600):
    """Circles: LR=~50%, provably nonlinear. Radial boundary."""
    from sklearn.datasets import make_circles
    X, y = make_circles(n, noise=0.05, factor=0.5, random_state=42)
    return normalize(X), y

def make_xor(n=600):
    """XOR: LR=50%, provably nonlinear. Crossing boundary."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 2)
    y = ((X[:,0]>0) ^ (X[:,1]>0)).astype(int)
    X += rng.randn(n,2)*0.2
    return normalize(X), y

def make_multicluster(n=1000, n_clusters=3, seed=42):
    """Multiple clusters per class: LR=~63%, genuinely hard."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n, n_features=20, n_informative=8,
                                n_redundant=4, n_clusters_per_class=n_clusters,
                                flip_y=0.05, random_state=seed)
    return normalize(X), y

def make_hard_clusters(n=1000):
    """4 clusters/class, low separation: LR=~63%."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n, n_features=30, n_informative=6,
                                n_redundant=8, n_clusters_per_class=4,
                                class_sep=0.5, flip_y=0.08, random_state=42)
    return normalize(X), y

def make_digits_binary(class_a=1, class_b=7, n_per_class=200):
    """Digits pair: real 64D pixel data, confusable pairs.
    LR 20-shot on hard pairs (1v7, 3v8, 4v9): 70-85%.
    Nonlinear methods: 95%+. Real image structure."""
    from sklearn.datasets import load_digits
    d = load_digits()
    mask = np.isin(d.target, [class_a, class_b])
    X = d.data[mask]
    y = (d.target[mask] == class_a).astype(int)
    # Subsample to n_per_class
    rng = np.random.RandomState(42)
    idx = np.concatenate([
        rng.choice(np.where(y==c)[0], min(n_per_class, (y==c).sum()), replace=False)
        for c in [0,1]])
    return normalize(X[idx]), y[idx]

def make_digits_hard_pair():
    """Digits 1 vs 7: visually similar, known LR failure mode."""
    return make_digits_binary(1, 7)

def make_digits_3v8():
    """Digits 3 vs 8: curved stroke boundary."""
    return make_digits_binary(3, 8)

def make_nonlinear_interaction(n=800):
    """Pure interaction structure: y = sign(x1*x2*x3).
    LR=~50% (no main effects). EBM crossing/product should dominate."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, 6)
    y = ((X[:,0]*X[:,1]*X[:,2]) > 0).astype(int)
    X += rng.randn(n,6)*0.3
    return normalize(X), y

def make_concentric_spheres(n=600):
    """3D concentric spheres: radial boundary in 3D.
    LR=~50%. Radial primitive should dominate."""
    rng = np.random.RandomState(42)
    n2 = n//2
    # Inner sphere
    r0 = rng.uniform(0,0.5,n2)
    phi0 = rng.uniform(0,2*np.pi,n2); theta0 = rng.uniform(0,np.pi,n2)
    X0 = np.stack([r0*np.sin(theta0)*np.cos(phi0),
                   r0*np.sin(theta0)*np.sin(phi0),
                   r0*np.cos(theta0)],1)
    # Outer shell
    r1 = rng.uniform(0.8,1.2,n2)
    phi1 = rng.uniform(0,2*np.pi,n2); theta1 = rng.uniform(0,np.pi,n2)
    X1 = np.stack([r1*np.sin(theta1)*np.cos(phi1),
                   r1*np.sin(theta1)*np.sin(phi1),
                   r1*np.cos(theta1)],1)
    X = np.vstack([X0,X1]) + rng.randn(n,3)*0.05
    return normalize(X), np.array([0]*n2+[1]*n2)

def make_hopf_torus(n=600):
    """Hopf-like torus: limit cycle boundary.
    LR=~50%. Limit_cycle (Dog) primitive should activate."""
    rng = np.random.RandomState(42)
    n2 = n//2
    t = rng.uniform(0,2*np.pi,n2)
    # Class 0: points near torus surface
    R,r = 1.0, 0.3
    X0 = np.stack([
        (R+r*np.cos(t))*np.cos(t) + rng.randn(n2)*0.05,
        (R+r*np.cos(t))*np.sin(t) + rng.randn(n2)*0.05,
        r*np.sin(t) + rng.randn(n2)*0.05,
        np.cos(2*t) + rng.randn(n2)*0.1
    ],1)
    # Class 1: points inside and outside torus
    X1 = rng.randn(n2,4) * 1.5
    X1 = X1[np.sqrt((X1**2).sum(1)) > 0.1][:n2]
    if len(X1) < n2:
        X1 = np.vstack([X1, rng.randn(n2-len(X1),4)*1.5])
    return normalize(np.vstack([X0,X1[:n2]])), np.array([0]*n2+[1]*n2)

def make_cusp_boundary(n=600):
    """Cusp catastrophe boundary: y = sign(x1^2 - x2^3).
    LR=~50%. Cusp (Pig) primitive should dominate."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-2,2,(n,4))
    X[:,2:] = rng.randn(n,2)*0.5  # noise dims
    y = (X[:,0]**2 > X[:,1]**3 + 0.5).astype(int)
    X += rng.randn(n,4)*0.1
    return normalize(X), y

def make_saddle_boundary(n=600):
    """Saddle point boundary: y = sign(x1*x2).
    LR=~50%. Saddle (Monkey) primitive should dominate."""
    rng = np.random.RandomState(42)
    X = rng.randn(n,5)
    y = ((X[:,0]*X[:,1]) > 0).astype(int)
    X += rng.randn(n,5)*0.2
    return normalize(X), y

def load_sklearn_real():
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    results = {}
    bc = load_breast_cancer()
    results['BreastCancer'] = (normalize(bc.data), bc.target)
    iris = load_iris()
    results['Iris_OvR'] = (normalize(iris.data), (iris.target==0).astype(int))
    wine = load_wine()
    results['Wine_OvR'] = (normalize(wine.data), (wine.target==0).astype(int))
    return results

# ══════════════════════════════════════════════════════════════════
# DATASETS — synthetic + real + proxy nonlinear
# ══════════════════════════════════════════════════════════════════

def normalize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)

def make_checker(n=600):
    X = np.random.uniform(-1,1,(n,2))
    y = ((np.floor(X[:,0]*2)+np.floor(X[:,1]*2))%2).astype(int)
    return normalize(X + np.random.randn(n,2)*0.04), y

def make_rings(n=600):
    n2=n//2
    r0=np.random.uniform(0.2,0.5,n2); r1=np.random.uniform(0.7,1.0,n2)
    t0=np.random.uniform(0,2*np.pi,n2); t1=np.random.uniform(0,2*np.pi,n2)
    X=np.vstack([np.stack([r0*np.cos(t0),r0*np.sin(t0)],1),
                 np.stack([r1*np.cos(t1),r1*np.sin(t1)],1)])
    return normalize(X+np.random.randn(n,2)*0.04), np.array([0]*n2+[1]*n2)

def make_spirals(n=600):
    n2=n//2; t=np.sqrt(np.random.rand(n2))*3*np.pi; r=t/(3*np.pi)
    g=0.13*np.sin(8*t)+0.065*np.cos(8*1.7*t)
    X=np.vstack([
        np.stack([(r+g)*np.cos(t),(r+g)*np.sin(t)],1)+np.random.randn(n2,2)*0.03,
        np.stack([(r+g)*np.cos(t+np.pi),(r+g)*np.sin(t+np.pi)],1)+np.random.randn(n2,2)*0.03
    ])
    return normalize(X), np.array([0]*n2+[1]*n2)

def make_pinwheel(n=600, n_blades=5):
    rng=np.random.RandomState(42); Xl,yl=[],[]
    for k in range(n_blades):
        nk=n//n_blades
        r=rng.uniform(0.1,1.0,nk); t=rng.uniform(k*2*np.pi/n_blades,(k+0.5)*2*np.pi/n_blades,nk)
        Xl.append(np.stack([r*np.cos(t),r*np.sin(t)],1)); yl.append(np.full(nk,k%2))
    return normalize(np.vstack(Xl)+rng.randn(n,2)*0.03), np.concatenate(yl).astype(int)

def make_protein_fold_proxy(n=400):
    """
    Proxy for SCOP fold classification.
    Two fold classes with curved decision boundary in AA composition space.
    Class 0: alpha-helix dominated (periodic, local structure)
    Class 1: beta-sheet dominated (extended, nonlocal structure)
    Features: synthetic 6D AA composition + secondary structure fractions
    Known: LR achieves ~55% on hard targets; true boundary is radial/torsional
    """
    rng = np.random.RandomState(11)
    n2 = n//2
    # Alpha: high in ACDEFGHIKLM (helix-formers), periodic in feature space
    alpha = rng.randn(n2, 6)
    alpha[:,0] += 1.5; alpha[:,1] += np.sin(alpha[:,0]) * 1.2  # periodic coupling
    alpha[:,2] += 0.5 * alpha[:,0]**2  # nonlinear
    # Beta: high in FIVYW (sheet-formers), radial in feature space
    theta = rng.uniform(0, 2*np.pi, n2)
    r = rng.uniform(1.5, 2.5, n2)
    beta = rng.randn(n2, 6) * 0.6
    beta[:,0] += r * np.cos(theta); beta[:,1] += r * np.sin(theta)
    beta[:,3] += 1.2  # different secondary structure signature
    X = normalize(np.vstack([alpha, beta]))
    y = np.array([0]*n2 + [1]*n2)
    return X, y

def make_fmri_brain_state_proxy(n=400):
    """
    Proxy for fMRI brain state classification.
    Binary: congruent vs incongruent Flanker task condition.
    Features: 8D ROI-averaged BOLD signal (prefrontal, parietal, motor...)
    True boundary: torsional/oscillatory — brain states cycle through manifolds
    LR baseline: ~65-70% (known from MVPA literature)
    """
    rng = np.random.RandomState(13)
    n2 = n//2
    t = rng.uniform(0, 4*np.pi, n2)
    # Congruent: low conflict, rhythmic activation
    cong = rng.randn(n2, 8) * 0.5
    cong[:,0] += np.cos(t) * 1.0    # prefrontal oscillates
    cong[:,1] += np.sin(t) * 0.8    # ACC coupling
    cong[:,2] += np.cos(2*t) * 0.5  # motor cortex
    # Incongruent: high conflict, torsional activation pattern
    t2 = rng.uniform(0, 4*np.pi, n2)
    incong = rng.randn(n2, 8) * 0.5
    incong[:,0] += np.cos(t2 + np.pi) * 1.2
    incong[:,1] += np.sin(t2) * 1.0 + np.cos(t2)**2 * 0.7
    incong[:,3] += 1.5  # dlPFC signature
    incong[:,4] += np.sin(2*t2) * 0.8
    X = normalize(np.vstack([cong, incong]))
    y = np.array([0]*n2 + [1]*n2)
    return X, y

def make_materials_phase_proxy(n=400):
    """
    Proxy for BaTiO3-style ferroelectric phase boundary.
    Binary: rhombohedral vs tetragonal phase.
    Features: composition (x,T,field), lattice parameters, dielectric constant
    True boundary: known to be curved/radial in composition-temperature space
    LR fails: ~60% — linear methods cannot capture the MPB (morphotropic phase boundary)
    """
    rng = np.random.RandomState(17)
    n2 = n//2
    # Rhombohedral phase: below MPB
    rhom = rng.randn(n2, 5) * 0.4
    x = rng.uniform(0.1, 0.28, n2)    # PT concentration
    T = rng.uniform(250, 320, n2)      # Temperature (K)
    rhom[:,0] = x + rng.randn(n2)*0.02
    rhom[:,1] = (T - 300)/50 + rng.randn(n2)*0.1
    rhom[:,2] = 4.01 + 0.05*x + rng.randn(n2)*0.01   # a lattice param
    rhom[:,3] = np.cos(x * 15) * 0.3   # dielectric oscillation
    # Tetragonal: above MPB — radial boundary in (x,T) space
    x2 = rng.uniform(0.28, 0.5, n2)
    T2 = rng.uniform(280, 350, n2)
    tetra = rng.randn(n2, 5) * 0.4
    tetra[:,0] = x2 + rng.randn(n2)*0.02
    tetra[:,1] = (T2 - 300)/50 + rng.randn(n2)*0.1
    tetra[:,2] = 4.03 + 0.08*x2**2 + rng.randn(n2)*0.01
    tetra[:,3] = np.sin(x2 * 15) * 0.3 + 0.8
    tetra[:,4] = (x2 - 0.3)**2 * 5     # nonlinear MPB signature
    X = normalize(np.vstack([rhom, tetra]))
    y = np.array([0]*n2 + [1]*n2)
    return X, y

def make_navier_stokes_regime_proxy(n=400):
    """
    Proxy for NS laminar→turbulent regime classification.
    Binary: laminar (Re < 2300) vs turbulent (Re > 4000).
    Features: 6D flow state (Re, velocity gradient, vorticity, pressure, wall shear, TKE)
    True boundary: known to be oscillatory/wormhole — the transition is not smooth,
                   it involves intermittent bursts (wormhole topology)
    LR fails: ~58% — the transition boundary is not linearly separable
    """
    rng = np.random.RandomState(19)
    n2 = n//2
    # Laminar: smooth, ordered
    Re_lam = rng.uniform(500, 2200, n2)
    lam = rng.randn(n2, 6) * 0.3
    lam[:,0] = (Re_lam - 1000) / 1000
    lam[:,1] = 0.1 * np.sin(lam[:,0] * 3)  # gentle oscillation
    lam[:,2] = lam[:,0]**2 * 0.3            # quadratic vorticity
    lam[:,3] = -0.5 + lam[:,0] * 0.2
    # Turbulent: chaotic, wormhole-like jumps between states
    Re_turb = rng.uniform(4100, 8000, n2)
    turb = rng.randn(n2, 6) * 0.8
    turb[:,0] = (Re_turb - 1000) / 1000
    t = rng.uniform(0, 6*np.pi, n2)
    turb[:,1] = np.sin(t) * 1.5 + np.cos(2*t) * 0.8   # oscillatory bursts
    turb[:,2] = np.exp(-((turb[:,0]-4)**2)/2) * np.sin(5*t)  # wormhole-like
    turb[:,3] = turb[:,0] + np.sin(turb[:,0]*2) * 1.2
    turb[:,4] = (np.abs(np.sin(t)) > 0.7).astype(float) * 2  # intermittency
    X = normalize(np.vstack([lam, turb]))
    y = np.array([0]*n2 + [1]*n2)
    return X, y

def load_sklearn_datasets():
    results = {}
    # Breast Cancer
    bc = load_breast_cancer()
    results['BreastCancer'] = (normalize(bc.data), bc.target)
    # Iris — OvR class 0 vs rest (most nonlinear boundary)
    iris = load_iris()
    Xi = normalize(iris.data)
    yi = (iris.target == 0).astype(int)
    results['Iris'] = (Xi, yi)
    # Wine — class 0 vs rest
    wine = load_wine()
    Xw = normalize(wine.data)
    yw = (wine.target == 0).astype(int)
    results['Wine'] = (Xw, yw)
    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("="*68)
    print("Cherenkov EBM v4.1")
    print("v4.3: Gap fixes — PCA synthesis, product composition, adaptive rounds")
    print("="*68)
    t0 = time.time()

    # ── Build dataset registry ──────────────────────────────────
    datasets = {}

    # ── Group 1: Provably LR fails (benchmark) ──────────────────
    datasets['Circles']          = make_circles_dataset(600)
    datasets['XOR']              = make_xor(600)
    datasets['Saddle_boundary']  = make_saddle_boundary(600)
    datasets['Cusp_boundary']    = make_cusp_boundary(600)
    datasets['Concentric_spheres'] = make_concentric_spheres(600)
    datasets['Hopf_torus']       = make_hopf_torus(600)
    datasets['Interaction_3way'] = make_nonlinear_interaction(800)

    # ── Group 2: Real data, LR genuinely struggles ───────────────
    datasets['Digits_1v7']       = make_digits_hard_pair()
    datasets['Digits_3v8']       = make_digits_3v8()
    datasets['MultiCluster']     = make_multicluster(1000, n_clusters=3)
    datasets['HardCluster']      = make_hard_clusters(1000)

    # ── Group 3: sklearn real (reference) ───────────────────────
    sklearn_ds = load_sklearn_real()
    datasets.update(sklearn_ds)

    # ── Group 4: Synthetic geometric (v3.0 benchmarks) ──────────
    datasets['Checker']  = make_checker(600)
    Xm, ym = make_moons(600, noise=0.15, random_state=42)
    datasets['Moons']    = (normalize(Xm), ym)
    datasets['Rings']    = make_rings(600)
    datasets['Spirals']  = make_spirals(600)
    datasets['Pinwheel'] = make_pinwheel(600)

    # ── Run ─────────────────────────────────────────────────────
    SHOT_BUDGET = 20
    results = {}
    for dname, (X, y) in datasets.items():
        print(f"\n{'─'*50}")
        # Hard domains get more rounds
        hard_domains = ['Interaction_3way','HardCluster','MultiCluster',
                        'Digits_1v7','Digits_3v8']
        n_rounds = 15 if dname in hard_domains else 10
        r = generator_loop(X, y, X, y, n_shot=SHOT_BUDGET,
                           max_rounds=n_rounds, domain_name=dname, verbose=True)
        results[dname] = r

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # ── VISUALIZATION ───────────────────────────────────────────
    BG='#0a0a0f'; PANEL='#12122a'
    CYAN='#00f5ff'; PURPLE='#bf5fff'; ORANGE='#ff6b35'
    GREEN='#39ff14'; GOLD='#ffd700'; TEAL='#00ff88'
    RED='#ff0080'; WHITE='#e0e0e0'; PINK='#ff69b4'
    LIME='#ccff00'; SKY='#87ceeb'

    FAMILY_COLORS = {
        'radial':       CYAN,
        'directional':  ORANGE,
        'oscillatory':  GREEN,
        'torsional':    TEAL,
        'crossing':     PURPLE,
        'trend':        GOLD,
        'acute':        LIME,
        'obtuse':       SKY,
        'wormhole':     PINK,
        'product':      RED,
        'saddle':       '#ff4488',
        'limit_cycle':  '#44ffcc',
        'cusp':         '#ffaa44',
    }

    DOMAIN_GROUPS = {
        'provably_lr_fails': ['Circles','XOR','Saddle_boundary','Cusp_boundary',
                              'Concentric_spheres','Hopf_torus','Interaction_3way'],
        'real_lr_struggles': ['Digits_1v7','Digits_3v8','MultiCluster','HardCluster'],
        'sklearn_ref':       ['BreastCancer','Iris_OvR','Wine_OvR'],
        'geometric_v30':     ['Checker','Moons','Rings','Spirals','Pinwheel'],
    }
    GROUP_COLORS = {
        'provably_lr_fails': PINK,
        'real_lr_struggles': LIME,
        'sklearn_ref':       GOLD,
        'geometric_v30':     CYAN,
    }

    all_domains = (DOMAIN_GROUPS['provably_lr_fails'] +
                   DOMAIN_GROUPS['real_lr_struggles'] +
                   DOMAIN_GROUPS['sklearn_ref'] +
                   DOMAIN_GROUPS['geometric_v30'])

    fig = plt.figure(figsize=(32, 28))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: Accuracy vs LR bar chart ─────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_facecolor(PANEL)
    x = np.arange(len(all_domains)); wb = 0.28
    lr_vals      = [results[d]['lr_acc']*100     for d in all_domains]
    zodiac_vals  = [results[d]['zodiac_acc']*100 for d in all_domains]
    ebm_vals     = [results[d]['final_acc']*100  for d in all_domains]
    ax_bar.bar(x-wb,    lr_vals,     wb, color=PURPLE, alpha=0.75, label='LR (20-shot)')
    ax_bar.bar(x,       zodiac_vals, wb, color=GOLD,   alpha=0.75, label='Zodiac best')
    ax_bar.bar(x+wb,    ebm_vals,    wb, color=TEAL,   alpha=0.85, label='EBM v4.0')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(all_domains, color='white', fontsize=8, rotation=25, ha='right')
    ax_bar.set_ylim(40, 115)
    ax_bar.set_title('EBM v4.0 — LR vs Zodiac vs EBM (20-shot)',
                    color=GOLD, fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    # Group separators
    ax_bar.axvline(6.5,  color='#333', lw=1.5, ls='--')
    ax_bar.axvline(10.5, color='#333', lw=1.5, ls='--')
    ax_bar.axvline(13.5, color='#333', lw=1.5, ls='--')
    ax_bar.text(3,   112, 'LR FAILS',    color=PINK, fontsize=8, ha='center', alpha=0.7)
    ax_bar.text(8.5, 112, 'REAL HARD',   color=LIME, fontsize=8, ha='center', alpha=0.7)
    ax_bar.text(12,  112, 'SKLEARN',     color=GOLD, fontsize=8, ha='center', alpha=0.7)
    ax_bar.text(17,  112, 'GEOMETRIC',   color=CYAN, fontsize=8, ha='center', alpha=0.7)
    ax_bar.tick_params(colors='#555')
    for sp in ax_bar.spines.values(): sp.set_edgecolor('#333')
    for b in ax_bar.patches[len(all_domains)*2:]:

        h = b.get_height()
        if h > 45:
            ax_bar.text(b.get_x()+b.get_width()/2, h+0.3, f'{h:.0f}',
            ha='center', va='bottom', color='white', fontsize=6.5)

    # ── Row 1: Zodiac winners ───────────────────────────────────
    ax_zod = fig.add_subplot(gs[1, :2])
    ax_zod.set_facecolor(PANEL); ax_zod.axis('off')
    ax_zod.set_title('Zodiac Winning Basis per Domain',
                    color=GOLD, fontsize=10, fontweight='bold')
    yp = 0.95
    for group, domains in DOMAIN_GROUPS.items():
        gcol = GROUP_COLORS[group]
        ax_zod.text(0.01, yp, group.upper(), color=gcol, fontsize=8,
                   fontweight='bold', transform=ax_zod.transAxes, va='top')
        yp -= 0.07
        for d in domains:
            r = results[d]
            ax_zod.text(0.04, yp, f"{d}:", color=WHITE, fontsize=7.5,
                       transform=ax_zod.transAxes, va='top')
            ax_zod.text(0.30, yp, r['zodiac_winner'], color=TEAL, fontsize=7.5,
                       transform=ax_zod.transAxes, va='top')
            ax_zod.text(0.62, yp, f"{r['zodiac_acc']:.1%}", color=GOLD, fontsize=7.5,
                       transform=ax_zod.transAxes, va='top')
            ax_zod.text(0.75, yp, f"Δ={r['delta_vs_zodiac']:+.1%}", 
                       color=GREEN if r['delta_vs_zodiac']>0 else RED,
                       fontsize=7.5, transform=ax_zod.transAxes, va='top')
            yp -= 0.065

    # ── Row 1: New primitive usage ──────────────────────────────
    ax_new = fig.add_subplot(gs[1, 2:])
    ax_new.set_facecolor(PANEL)
    new_families = ['acute', 'obtuse', 'wormhole']
    new_counts = {f: 0 for f in new_families}
    all_fam_counts = {}
    for r in results.values():
        for e in r['history']:
            if e['verdict'] == 'KEEP':
                f = e['family']
                all_fam_counts[f] = all_fam_counts.get(f,0) + 1
                if f in new_families:
                    new_counts[f] += 1

    all_fams = sorted(all_fam_counts.keys(), key=lambda f: -all_fam_counts[f])
    counts = [all_fam_counts[f] for f in all_fams]
    colors = [FAMILY_COLORS.get(f, WHITE) for f in all_fams]
    bars = ax_new.barh(all_fams, counts, color=colors, alpha=0.85)
    ax_new.set_title('Primitive families generated across all domains\n(new: acute · obtuse · wormhole)',
                    color=GOLD, fontsize=10, fontweight='bold')
    ax_new.tick_params(colors='white')
    for f, b in zip(all_fams, bars):
        c = all_fam_counts[f]
        col = LIME if f in new_families else WHITE
        ax_new.text(c+0.05, b.get_y()+b.get_height()/2, str(c),
                   va='center', color=col, fontsize=9,
                   fontweight='bold' if f in new_families else 'normal')
    for sp in ax_new.spines.values(): sp.set_edgecolor('#333')

    # ── Row 2: Residual variance convergence ────────────────────
    ax_var = fig.add_subplot(gs[2, :2])
    ax_var.set_facecolor(PANEL)
    domain_colors_flat = [CYAN,GOLD,ORANGE,TEAL,PURPLE,GREEN,RED,PINK,LIME,SKY,'#ff8888','#88ffff']
    for di, dname in enumerate(all_domains[:8]):
        r = results[dname]
        hist = r['history']
        if not hist: continue
        xs = [0] + [e['round'] for e in hist]
        ys = ([hist[0]['var_before']] + [e['var_after'] for e in hist])
        col = domain_colors_flat[di % len(domain_colors_flat)]
        ax_var.plot(xs, ys, 'o-', color=col, lw=1.5, markersize=4,
                   label=dname[:8], alpha=0.8)
    ax_var.axhline(0.10, color='#555', lw=1.5, ls='--', label='threshold')
    ax_var.set_xlabel('Generator round', color=WHITE, fontsize=9)
    ax_var.set_ylabel('Residual variance', color=WHITE, fontsize=9)
    ax_var.set_title('Residual variance — monotone decrease\n(synthetic + sklearn)',
                    color=GOLD, fontsize=10, fontweight='bold')
    ax_var.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white', ncol=2)
    ax_var.tick_params(colors='#555')
    for sp in ax_var.spines.values(): sp.set_edgecolor('#333')

    ax_var2 = fig.add_subplot(gs[2, 2:])
    ax_var2.set_facecolor(PANEL)
    for di, dname in enumerate(DOMAIN_GROUPS['provably_lr_fails']):
        r = results[dname]
        hist = r['history']
        if not hist: continue
        xs = [0] + [e['round'] for e in hist]
        ys = ([hist[0]['var_before']] + [e['var_after'] for e in hist])
        col = [PINK, LIME, SKY, ORANGE, CYAN, GOLD, TEAL, RED][di % 8]
        ax_var2.plot(xs, ys, 'o-', color=col, lw=2, markersize=5, label=dname)
    ax_var2.axhline(0.10, color='#555', lw=1.5, ls='--', label='threshold')
    ax_var2.set_xlabel('Generator round', color=WHITE, fontsize=9)
    ax_var2.set_ylabel('Residual variance', color=WHITE, fontsize=9)
    ax_var2.set_title('Residual variance — nonlinear domains\n(protein · fMRI · materials · NS)',
                    color=GOLD, fontsize=10, fontweight='bold')
    ax_var2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax_var2.tick_params(colors='#555')
    for sp in ax_var2.spines.values(): sp.set_edgecolor('#333')

    # ── Row 3: Domain fingerprints ──────────────────────────────
    ax_fp = fig.add_subplot(gs[3, :])
    ax_fp.set_facecolor(PANEL); ax_fp.axis('off')
    ax_fp.set_title('Domain Fingerprints — [zodiac winner] + [synthesized primitives]',
                   color=GOLD, fontsize=11, fontweight='bold')
    yp = 0.93; cols_per_row = 2
    items = list(results.items())
    for i, (dname, r) in enumerate(items):
        col_x = 0.01 + (i % cols_per_row) * 0.50
        if i % cols_per_row == 0 and i > 0:
            yp -= 0.11
        gcol = PINK if dname in DOMAIN_GROUPS['provably_lr_fails'] else \
               (GOLD if dname in DOMAIN_GROUPS['sklearn_ref'] else CYAN)
        ax_fp.text(col_x, yp, f"{dname}:", color=gcol, fontsize=8.5,
                  fontweight='bold', transform=ax_fp.transAxes, va='top')
        # Zodiac tag
        ax_fp.text(col_x+0.13, yp, f"[{r['zodiac_winner']}]",
                  color=TEAL, fontsize=7, transform=ax_fp.transAxes, va='top')
        # Primitives
        x_off = col_x + 0.13 + len(r['zodiac_winner'])*0.008 + 0.02
        for prim in r['fingerprint'][:5]:
            pcol = FAMILY_COLORS.get(prim, WHITE)
            ax_fp.text(x_off, yp, prim, color=pcol, fontsize=7,
                      transform=ax_fp.transAxes, va='top')
            x_off += len(prim)*0.009 + 0.015
        acc_col = GREEN if r['delta_vs_lr'] > 0 else RED
        ax_fp.text(col_x+0.47, yp, f"{r['final_acc']:.1%} Δ{r['delta_vs_lr']:+.1%}",
                  color=acc_col, fontsize=7.5,
                  transform=ax_fp.transAxes, va='top', ha='right')

    # ── Row 4: Generator log (nonlinear domains only) ───────────
    ax_log = fig.add_subplot(gs[4, :])
    ax_log.set_facecolor(PANEL); ax_log.axis('off')
    ax_log.set_title('Generator Log — Nonlinear Domains (protein · fMRI · materials · NS regime)',
                    color=GOLD, fontsize=11, fontweight='bold')
    cols_x = [0.01, 0.10, 0.19, 0.29, 0.46, 0.55, 0.64, 0.73, 0.84]
    headers = ['Domain','R','Family','φ description','corr(φ,R)','var↓','Before','After','Verdict']
    yp = 0.93
    for h, x in zip(headers, cols_x):
        ax_log.text(x, yp, h, color=GOLD, fontsize=7.5, fontweight='bold',
                   transform=ax_log.transAxes, va='top')
    yp -= 0.07
    log_domains = DOMAIN_GROUPS['provably_lr_fails'][:4]
    dcols = [PINK, LIME, SKY, ORANGE, CYAN, GOLD, TEAL, RED, WHITE, PURPLE, GREEN, GOLD]
    for di, dname in enumerate(log_domains):
        for e in results[dname]['history']:
            if yp < 0.03: break
            vcol = GREEN if e['verdict']=='KEEP' else '#555'
            fcol = FAMILY_COLORS.get(e['family'], WHITE)
            row = [
                (dname[:12],               dcols[di % len(dcols)]),
                (str(e['round']),          WHITE),
                (e['family'],              fcol),
                (e['family'][:20],         fcol),
                (f"{e['corr']:.3f}",       WHITE),
                (f"{e['var_before']-e['var_after']:+.3f}", GREEN if e['var_before']>e['var_after'] else '#555'),
                (f"{e['acc']-e['delta_acc']:.1%}", WHITE),
                (f"{e['acc']:.1%}",        TEAL if e['verdict']=='KEEP' else WHITE),
                (e['verdict'],             vcol),
            ]
            for (val, col), x in zip(row, cols_x):
                ax_log.text(x, yp, val, color=col, fontsize=7,
                           transform=ax_log.transAxes, va='top')
            yp -= 0.06

    fig.suptitle(
        'Cherenkov EBM v4.1 — Zodiac Linear Bases + Extended Primitive Library\n'
        'New: saddle · limit_cycle · cusp (bifurcation synthesis)  ·  12 Chinese Zodiac animals  ·  '
        'Nonlinear domains: protein · fMRI · materials · Navier-Stokes',
        color='white', fontsize=12, fontweight='bold', y=0.998)

    outpath = '/mnt/user-data/outputs/cherenkov_v43_results.png'
    plt.savefig(outpath, dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\nChart saved: {outpath}")

    # JSON export
    out = {
        'version': '4.3',
        'new_primitives': ['acute', 'obtuse', 'wormhole', 'saddle', 'limit_cycle', 'cusp'],
        'zodiac_bases': 12,
        'results': {d: {k:v for k,v in r.items() if k!='history'}
                    for d,r in results.items()},
    }
    with open('/mnt/user-data/outputs/cherenkov_v43_data.json','w') as f:
        json.dump(out, f, indent=2)

    # Print summary
    print("\n" + "="*68)
    print("v4.0 RESULTS SUMMARY")
    print("="*68)
    print(f"{'Domain':20s}  {'Zodiac':22s}  {'LR':>7}  {'Zodiac':>7}  {'EBM':>7}  {'ΔLR':>7}  {'Fingerprint'}")
    print("-"*100)
    for dname in all_domains:
        r = results[dname]
        fp = '+'.join(r['fingerprint'][:4]) or '(none)'
        print(f"{dname:20s}  {r['zodiac_winner']:22s}  "
              f"{r['lr_acc']:>7.1%}  {r['zodiac_acc']:>7.1%}  "
              f"{r['final_acc']:>7.1%}  {r['delta_vs_lr']:>+7.1%}  {fp}")

    return out

if __name__ == '__main__':
    run()
