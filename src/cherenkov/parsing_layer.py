"""
Cherenkov EBM — Nonlinear Parsing Layer
=========================================
sklearn-compatible transformer that augments feature matrix X
with synthesized geometric basis functions.

The key insight: fit primitive weights on the FULL target field
(not few-shot labels), so each primitive explains a slice of the
target structure directly. The downstream model handles classification.

Architecture:
    X → [zodiac filter] → residual → [EBM synthesis] → φ₁...φₖ
    X_aug = [X | φ₁(X) | ... | φₖ(X)]

Usage:
    from cherenkov import CherenkovParsingLayer
    layer = CherenkovParsingLayer(n_shot=20, max_rounds=10)
    X_aug = layer.fit_transform(X_train, y_train)

    # Or in a pipeline:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([
        ('parser', CherenkovParsingLayer()),
        ('clf',    LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from .primitives import phi_matrix, eval_primitive
from .synthesizer import PrimitiveSynthesizer
from .zodiac import zodiac_fit


class CherenkovParsingLayer(BaseEstimator, TransformerMixin):
    """
    Nonlinear parsing layer via autonomous geometric primitive synthesis.

    Parameters
    ----------
    n_shot : int, default 20
        Shots per class used by zodiac filter.
    max_rounds : int, default 10
        Maximum synthesis rounds.
    min_corr : float, default 0.08
        Minimum residual correlation to accept a primitive candidate.
    var_threshold : float, default 0.05
        Stop synthesis when residual variance drops below this.
    verbose : bool, default False
        Print synthesis log.
    """

    def __init__(
        self,
        n_shot: int = 20,
        max_rounds: int = 10,
        min_corr: float = 0.08,
        var_threshold: float = 0.05,
        verbose: bool = False,
    ):
        self.n_shot = n_shot
        self.max_rounds = max_rounds
        self.min_corr = min_corr
        self.var_threshold = var_threshold
        self.verbose = verbose

        self.primitives_ = []
        self.zodiac_winner_ = None
        self.fingerprint_ = []
        self.n_features_in_ = None

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Synthesize geometric primitives from X and y.

        The primitives are fitted to explain the target field directly
        (full-data OLS), not the few-shot classification labels.
        This ensures each primitive captures real geometric structure
        rather than noise from sparse supervision.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_features_in_ = X.shape[1]

        tgt = np.where(y == 0, -1.0, 1.0).astype(float)

        # Zodiac filter — labels the linear coordinate
        zodiac_name, zodiac_acc, _, _ = zodiac_fit(X, y, self.n_shot)
        self.zodiac_winner_ = zodiac_name

        if self.verbose:
            print(f"  Zodiac: {zodiac_name} ({zodiac_acc:.1%})")

        synth = PrimitiveSynthesizer(min_corr=self.min_corr)
        primitives = []
        R = tgt.copy()  # Start from full target

        for rnd in range(1, self.max_rounds + 1):
            var_before = float(np.var(R))
            if var_before < self.var_threshold:
                break

            prim = synth.synthesize(X, R, round_n=rnd)
            if prim is None:
                break

            primitives.append(prim)

            # Fit weights on FULL X to minimise residual
            Phi = phi_matrix(primitives, X)
            w, _, _, _ = np.linalg.lstsq(Phi, tgt, rcond=None)
            R_new = tgt - Phi @ w
            var_after = float(np.var(R_new))

            if var_after < var_before - 0.005:
                R = R_new
                if self.verbose:
                    print(f"  R{rnd:02d} ✓ [{prim['family']:12s}] "
                          f"corr={prim['corr']:.3f}  "
                          f"var {var_before:.3f}→{var_after:.3f}")
            else:
                primitives.pop()
                if self.verbose:
                    print(f"  R{rnd:02d} ~ [{prim['family']:12s}] "
                          f"no var reduction")

        self.primitives_ = primitives
        self.fingerprint_ = [p['family'] for p in primitives]

        if self.verbose:
            print(f"  Fingerprint: {self.fingerprint_}")

        return self

    # ── Transform ─────────────────────────────────────────────────

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Augment X with synthesized primitive features.

        Returns X_aug = [X | φ₁(X) | ... | φₖ(X)], shape (N, d+K).
        If no primitives were synthesised, returns X unchanged.
        """
        X = np.asarray(X, dtype=float)
        if not self.primitives_:
            return X
        Phi = phi_matrix(self.primitives_, X)
        return np.hstack([X, Phi])

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # ── Properties ────────────────────────────────────────────────

    @property
    def n_primitives(self) -> int:
        return len(self.primitives_)

    @property
    def n_features_out(self) -> int:
        return (self.n_features_in_ or 0) + self.n_primitives

    def describe(self) -> dict:
        return {
            'zodiac_winner':  self.zodiac_winner_,
            'n_primitives':   self.n_primitives,
            'fingerprint':    self.fingerprint_,
            'n_features_in':  self.n_features_in_,
            'n_features_out': self.n_features_out,
        }
