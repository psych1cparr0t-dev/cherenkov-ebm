"""
Cherenkov EBM — Zodiac Linear Bases
=====================================
12 linear archetypes that partition linear structure.
Each one is a different answer to: where does linearity live?

The zodiac filter runs first. The winning basis labels the domain's
linear coordinate. The EBM reads the residual.

Groups
------
Statistics:  OLS, PCA_projection, rank_correlation
Signal:      Fourier_linear, wavelet_approx, derivative_linear
Geometry:    affine_transform, projection_pursuit, piecewise_linear
Physics:     log_linear, interaction_linear, margin_linear
"""

import numpy as np
from typing import Tuple, Dict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def zodiac_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_shot: int = 20,
) -> Tuple[str, float, np.ndarray, Dict[str, float]]:
    """
    Run all 12 zodiac bases on few-shot data.

    Returns
    -------
    best_name : str
        Name of the winning basis.
    best_acc : float
        Accuracy of the winning basis.
    best_R : np.ndarray, shape (N,)
        Residual from the winning basis (target - score).
    all_accs : dict
        Accuracy of every basis tried.
    """
    N, d = X.shape
    rng = np.random.RandomState(7)
    classes = np.unique(y)
    idx = np.concatenate([
        rng.choice(np.where(y == c)[0],
                   min(n_shot, int((y == c).sum())),
                   replace=False)
        for c in classes
    ])
    mask = np.zeros(N, bool); mask[idx] = True
    Xf, yf = X[mask], y[mask]
    tgt = np.where(y == 0, -1., 1.).astype(float)

    results = {}

    def _add(name, pred_y, score=None):
        acc = float(np.mean(pred_y == y))
        R = tgt - score if score is not None \
            else tgt - (pred_y * 2 - 1).astype(float)
        results[name] = (acc, R)

    # 1. OLS
    try:
        ridge = Ridge(alpha=1.0).fit(Xf, np.where(yf==0,-1.,1.))
        s = ridge.predict(X)
        _add('OLS', (s>0).astype(int), s)
    except Exception: pass

    # 2. PCA projection
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(2, d)).fit(Xf)
        clf = LogisticRegression(max_iter=500).fit(pca.transform(Xf), yf)
        s = clf.decision_function(pca.transform(X))
        _add('PCA_projection', clf.predict(pca.transform(X)), s)
    except Exception: pass

    # 3. Rank correlation
    try:
        from scipy.stats import rankdata
        Xr_f = np.stack([rankdata(Xf[:,i]) for i in range(d)], 1) / len(Xf)
        Xr   = np.stack([rankdata(X[:,i])  for i in range(d)], 1) / len(X)
        clf = LogisticRegression(max_iter=500).fit(Xr_f, yf)
        s = clf.decision_function(Xr)
        _add('rank_correlation', clf.predict(Xr), s)
    except Exception: pass

    # 4. Fourier linear
    try:
        def _fourier(A):
            return np.hstack([np.sin(2*np.pi*A), np.cos(2*np.pi*A)])
        clf = LogisticRegression(max_iter=500).fit(_fourier(Xf), yf)
        s = clf.decision_function(_fourier(X))
        _add('Fourier_linear', clf.predict(_fourier(X)), s)
    except Exception: pass

    # 5. Wavelet approx
    try:
        d2 = d - (d % 2)
        if d2 >= 2:
            def _haar(A):
                lo = (A[:,::2] + A[:,1::2]) / 2
                hi = (A[:,::2] - A[:,1::2]) / 2
                return np.hstack([lo, hi])
            clf = LogisticRegression(max_iter=500).fit(_haar(Xf[:,:d2]), yf)
            s = clf.decision_function(_haar(X[:,:d2]))
            _add('wavelet_approx', clf.predict(_haar(X[:,:d2])), s)
    except Exception: pass

    # 6. Derivative linear
    try:
        def _grad(A):
            return np.hstack([np.diff(A, axis=1), A[:,:-1]]) if A.shape[1]>=2 else A
        clf = LogisticRegression(max_iter=500).fit(_grad(Xf), yf)
        s = clf.decision_function(_grad(X))
        _add('derivative_linear', clf.predict(_grad(X)), s)
    except Exception: pass

    # 7. Affine transform
    try:
        sc = StandardScaler().fit(Xf)
        clf = LogisticRegression(max_iter=500).fit(sc.transform(Xf), yf)
        s = clf.decision_function(sc.transform(X))
        _add('affine_transform', clf.predict(sc.transform(X)), s)
    except Exception: pass

    # 8. Projection pursuit (LDA)
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis().fit(Xf, yf)
        Xl = lda.transform(X); Xl_f = lda.transform(Xf)
        clf = LogisticRegression(max_iter=500).fit(Xl_f, yf)
        s = clf.decision_function(Xl)
        _add('projection_pursuit', clf.predict(Xl), s)
    except Exception: pass

    # 9. Piecewise linear
    try:
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=3).fit(Xf, yf)
        _add('piecewise_linear', dt.predict(X))
    except Exception: pass

    # 10. Log-linear
    try:
        Xl_f = np.sign(Xf) * np.log1p(np.abs(Xf))
        Xl   = np.sign(X)  * np.log1p(np.abs(X))
        clf = LogisticRegression(max_iter=500).fit(Xl_f, yf)
        s = clf.decision_function(Xl)
        _add('log_linear', clf.predict(Xl), s)
    except Exception: pass

    # 11. Interaction linear
    try:
        def _interact(A, max_f=4):
            du = min(A.shape[1], max_f)
            pairs = [(i,j) for i in range(du) for j in range(i+1,du)]
            if not pairs: return A
            return np.hstack([A, np.stack([A[:,i]*A[:,j] for i,j in pairs],1)])
        clf = LogisticRegression(max_iter=500).fit(_interact(Xf), yf)
        s = clf.decision_function(_interact(X))
        _add('interaction_linear', clf.predict(_interact(X)), s)
    except Exception: pass

    # 12. Margin linear (SVM)
    try:
        from sklearn.svm import LinearSVC
        svc = LinearSVC(max_iter=2000).fit(Xf, yf)
        s = svc.decision_function(X)
        _add('margin_linear', svc.predict(X), s)
    except Exception: pass

    if not results:
        clf = LogisticRegression(max_iter=1000).fit(Xf, yf)
        s = clf.decision_function(X)
        _add('OLS', clf.predict(X), s)

    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_R = results[best_name]
    all_accs = {k: round(v[0], 4) for k, v in results.items()}
    return best_name, best_acc, best_R, all_accs
