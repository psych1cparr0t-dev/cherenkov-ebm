"""
Cherenkov EBM — Primitive Synthesizer
======================================
Fits all 12 zodiac families to the residual field.
Returns whichever correlates best with R.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .primitives import eval_primitive


class PrimitiveSynthesizer:
    """
    Given a residual field R over points X, synthesize the primitive
    that maximally correlates with R.

    The synthesizer tries all 12 zodiac families via grid search and
    returns the best-fitting one. If nothing reaches min_corr, returns None.

    For high-D data (d > 8), automatically projects to 8D via PCA
    before searching for centers and fitting directional primitives.
    """

    def __init__(self, min_corr: float = 0.06, n_centers: int = 6):
        self.min_corr = min_corr
        self.n_centers = n_centers
        self._pca_cache = None

    # ── Utilities ─────────────────────────────────────────────────

    def _corr(self, phi: np.ndarray, R: np.ndarray) -> float:
        if phi.std() < 1e-8:
            return 0.0
        c = np.corrcoef(phi, R)[0, 1]
        return 0.0 if np.isnan(c) else abs(float(c))

    def _centers(self, X: np.ndarray, R: np.ndarray) -> np.ndarray:
        weights = np.abs(R) / (np.abs(R).sum() + 1e-8)
        idx = np.random.choice(
            len(X), size=min(self.n_centers * 3, len(X)),
            replace=False, p=weights)
        nc = min(self.n_centers, len(idx))
        km = KMeans(n_clusters=nc, n_init=3, random_state=42)
        km.fit(X[idx])
        return km.cluster_centers_

    def _best(self, family, params_list, X, R):
        best_p, best_c = 0.0, {}
        for p in params_list:
            phi = eval_primitive(family, p, X)
            cr = self._corr(phi, R)
            if cr > best_p:
                best_p = cr; best_c = p
        return best_c, best_p

    # ── Family fitters ─────────────────────────────────────────────

    def _fit_radial(self, X, R, centers):
        candidates = [
            {'center': c, 'sigma': s, 'power': p}
            for c in centers
            for s in [0.15, 0.3, 0.5, 0.8, 1.2, 2.0]
            for p in [0.8, 1.5, 2.0, 3.0]
        ]
        return self._best('radial', candidates, X, R)

    def _fit_directional(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in np.linspace(0, np.pi, 12):
            w = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) \
                if d >= 2 else np.array([1.0])
            for c in centers:
                for b_off in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                    b = -(w[:d]*c[:d]).sum() + b_off
                    candidates.append({'center': c, 'w': w, 'b': b})
        return self._best('directional', candidates, X, R)

    def _fit_oscillatory(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            for angle in np.linspace(0, np.pi, 8):
                axis = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) \
                       if d >= 2 else np.array([1.0])
                for c in centers[:3]:
                    candidates.append({'center': c, 'freq': freq, 'axis': axis})
        return self._best('oscillatory', candidates, X, R)

    def _fit_torsional(self, X, R, centers):
        candidates = [
            {'center': c, 'freq': f, 'sigma': s}
            for c in centers
            for f in [1.0, 2.0, 3.0, 4.0]
            for s in [0.3, 0.5, 0.8, 1.2]
        ]
        return self._best('torsional', candidates, X, R)

    def _fit_crossing(self, X, R, centers):
        d = X.shape[1]
        pairs = [(i,j) for i in range(min(d,4)) for j in range(i+1,min(d,4))]
        if not pairs:
            pairs = [(0, min(1,d-1))]
        candidates = []
        for subset in [pairs[:1], pairs[:2], pairs[:3]]:
            W = np.ones(len(subset))
            for b in [-0.5, 0.0, 0.5]:
                candidates.append({'pairs': subset, 'W': W, 'b': b})
        return self._best('crossing', candidates, X, R)

    def _fit_trend(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in np.linspace(0, 2*np.pi, 12):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) \
                        if d >= 2 else np.array([1.0])
            for c in centers:
                for sigma in [0.2, 0.4, 0.7, 1.1]:
                    for align in [0.5, 1.0, 2.0]:
                        candidates.append({'center': c, 'direction': direction,
                                           'sigma': sigma, 'align': align})
        return self._best('trend', candidates, X, R)

    def _fit_acute(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in [np.pi/6, np.pi/5, np.pi/4, np.pi/3]:
            for da in np.linspace(0, np.pi, 8):
                direction = np.array([np.cos(da), np.sin(da)] + [0.]*(d-2)) \
                            if d >= 2 else np.array([1.0])
                for c in centers[:3]:
                    for sigma in [0.2, 0.4, 0.6]:
                        candidates.append({'center': c, 'angle': angle,
                                           'direction': direction, 'sigma': sigma})
        return self._best('acute', candidates, X, R)

    def _fit_obtuse(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in [2*np.pi/3, 3*np.pi/4, 5*np.pi/6]:
            for da in np.linspace(0, np.pi, 8):
                direction = np.array([np.cos(da), np.sin(da)] + [0.]*(d-2)) \
                            if d >= 2 else np.array([1.0])
                for c in centers[:3]:
                    for sigma in [0.3, 0.5, 0.8]:
                        candidates.append({'center': c, 'angle': angle,
                                           'direction': direction, 'sigma': sigma})
        return self._best('obtuse', candidates, X, R)

    def _fit_wormhole(self, X, R, centers):
        candidates = [
            {'center': c, 'sigma': s, 'epsilon': e}
            for c in centers
            for s in [0.4, 0.6, 0.8, 1.2, 1.8]
            for e in [0.05, 0.1, 0.15, 0.25]
            if e < s
        ]
        return self._best('wormhole', candidates, X, R)

    def _fit_saddle(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in np.linspace(0, np.pi, 8):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) \
                        if d >= 2 else np.array([1.0])
            for c in centers:
                for a in [0.8, 1.5, 2.5]:
                    for b in [0.8, 1.5, 2.5]:
                        candidates.append({'center': c, 'direction': direction,
                                           'a': a, 'b': b})
        return self._best('saddle', candidates, X, R)

    def _fit_limit_cycle(self, X, R, centers):
        candidates = [
            {'center': c, 'r_star': r, 'sigma': s}
            for c in centers
            for r in [0.15, 0.25, 0.4, 0.6, 0.9, 1.3]
            for s in [0.08, 0.15, 0.25, 0.4]
        ]
        return self._best('limit_cycle', candidates, X, R)

    def _fit_cusp(self, X, R, centers):
        d = X.shape[1]
        candidates = []
        for angle in np.linspace(0, np.pi, 8):
            direction = np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2)) \
                        if d >= 2 else np.array([1.0])
            for c in centers:
                for sigma in [0.2, 0.4, 0.7]:
                    for scale in [0.3, 0.7, 1.2, 2.0]:
                        candidates.append({'center': c, 'direction': direction,
                                           'sigma': sigma, 'scale': scale})
        return self._best('cusp', candidates, X, R)

    def _fit_product(self, X, R, centers):
        """Product composition: φ_a · φ_b."""
        best_p, best_c = 0.0, {}
        fast = {
            'radial':      self._fit_radial,
            'directional': self._fit_directional,
            'trend':       self._fit_trend,
            'crossing':    self._fit_crossing,
        }
        for fa, fitter_a in fast.items():
            pa, ca = fitter_a(X, R, centers[:2])
            if ca < 0.05:
                continue
            phi_a = eval_primitive(fa, pa, X)
            if phi_a.std() < 1e-8:
                continue
            R_masked = R * np.abs(phi_a) / (np.abs(phi_a).max() + 1e-8)
            for fb, fitter_b in fast.items():
                if fb == fa:
                    continue
                pb, _ = fitter_b(X, R_masked, centers[:2])
                phi_b = eval_primitive(fb, pb, X)
                phi_prod = phi_a * phi_b
                if phi_prod.std() < 1e-8:
                    continue
                cr = abs(float(np.corrcoef(phi_prod, R)[0, 1]))
                if not np.isnan(cr) and cr > best_p:
                    best_p = cr
                    best_c = {'family_a': fa, 'params_a': pa,
                              'family_b': fb, 'params_b': pb}
        return best_c, best_p

    # ── Main synthesis call ────────────────────────────────────────

    def synthesize(self, X: np.ndarray, R: np.ndarray,
                   round_n: int = 0) -> dict | None:
        """
        Synthesize best primitive for residual R over points X.

        Returns dict with keys: family, params, corr, round.
        Returns None if no family reaches min_corr.
        """
        # Adaptive PCA for high-D
        d = X.shape[1]
        if d > 8:
            if self._pca_cache is None:
                self._pca_cache = PCA(n_components=8, random_state=42).fit(X)
            X_fit = self._pca_cache.transform(X)
        else:
            X_fit = X

        centers = self._centers(X_fit, R)

        fitters = {
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
        for fname, fitter in fitters.items():
            params, corr = fitter(X_fit, R, centers)
            if corr > best_corr:
                best_corr = corr
                best_family = fname
                best_params = params

        # Try product composition if single families are weak
        if best_corr < self.min_corr * 2:
            prod_params, prod_corr = self._fit_product(X_fit, R, centers)
            if prod_corr > best_corr:
                best_corr = prod_corr
                best_family = 'product'
                best_params = prod_params

        if best_corr < self.min_corr:
            return None

        return {
            'family': best_family,
            'params': best_params,
            'corr':   best_corr,
            'round':  round_n,
        }
