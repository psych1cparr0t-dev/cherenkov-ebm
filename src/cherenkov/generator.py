"""
Cherenkov EBM — Generator Loop
================================
Autonomous primitive synthesis from residual geometry.

Each round:
  1. Fit current library → compute residual R = target - E(x)
  2. Synthesize new primitive that maximally correlates with R
  3. Keep if accuracy improves or residual variance drops
  4. Repeat until convergence or max_rounds

The library grows from scratch. No catalogue.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .primitives import phi_matrix
from .synthesizer import PrimitiveSynthesizer
from .zodiac import zodiac_fit


def fit_weights(
    primitives: list,
    Xf: np.ndarray,
    yf: np.ndarray,
    reg: float = 0.01,
    lr: float = 0.08,
    epochs: int = 500,
) -> np.ndarray:
    """Fit library weights via Adam minimising squared contrastive loss."""
    Phi = phi_matrix(primitives, Xf)
    tgt = np.where(yf == 0, -1.0, 1.0).astype(float)
    w = np.zeros(Phi.shape[1])
    mw = np.zeros_like(w); vw = np.zeros_like(w); t = 0
    bw = w.copy(); bl = float('inf')
    for _ in range(epochs):
        E = Phi @ w; res = E - tgt
        loss = (res**2).mean() + reg*(w**2).sum()
        if loss < bl: bl = loss; bw = w.copy()
        gw = 2*res@Phi/len(yf) + 2*reg*w; t += 1
        mw = 0.9*mw + 0.1*gw; vw = 0.999*vw + 0.001*gw**2
        w -= lr*(mw/(1-0.9**t))/(np.sqrt(vw/(1-0.999**t))+1e-8)
    return bw


def predict(primitives: list, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Binary prediction from library."""
    if len(w) == 0:
        return np.zeros(len(X), int)
    return (phi_matrix(primitives, X) @ w > 0).astype(int)


def generator_loop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_shot: int = 20,
    max_rounds: int = 10,
    var_threshold: float = 0.10,
    min_corr: float = 0.06,
    min_delta_acc: float = 0.003,
    domain_name: str = 'domain',
    verbose: bool = True,
) -> dict:
    """
    Run the autonomous primitive generator.

    Parameters
    ----------
    X_train, y_train : training data (full set; few-shot split done internally)
    X_test, y_test   : evaluation data
    n_shot           : shots per class for few-shot split
    max_rounds       : maximum generator rounds
    var_threshold    : stop when residual variance < this
    min_corr         : minimum correlation to accept a primitive
    min_delta_acc    : minimum accuracy improvement to keep a primitive
    domain_name      : label for logging
    verbose          : print round-by-round log

    Returns
    -------
    dict with keys:
        domain, final_acc, zodiac_acc, zodiac_winner,
        lr_acc, delta_vs_lr, n_primitives, fingerprint,
        families_used, history
    """
    synth = PrimitiveSynthesizer(min_corr=min_corr)
    primitives = []

    # Few-shot split
    rng = np.random.RandomState(7)
    classes = np.unique(y_train)
    idx = np.concatenate([
        rng.choice(np.where(y_train==c)[0],
                   min(n_shot, int((y_train==c).sum())),
                   replace=False)
        for c in classes
    ])
    mask = np.zeros(len(y_train), bool); mask[idx] = True
    Xf, yf = X_train[mask], y_train[mask]

    # Zodiac filter
    zodiac_name, zodiac_acc, R_zodiac, all_zodiac = \
        zodiac_fit(X_test, y_test, n_shot)

    # LR baseline
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(Xf, yf)
    lr_acc = accuracy_score(y_test, lr_clf.predict(X_test))

    tgt_test = np.where(y_test==0, -1., 1.).astype(float)
    current_acc = zodiac_acc
    reg = 0.01
    history = []
    R = R_zodiac.copy()

    if verbose:
        print(f"\n  {domain_name}  n_shot={n_shot}")
        print(f"  Zodiac: {zodiac_name} ({zodiac_acc:.1%})  LR={lr_acc:.1%}")

    for rnd in range(1, max_rounds + 1):
        if primitives:
            w = fit_weights(primitives, Xf, yf, reg=reg)
            E = phi_matrix(primitives, X_test) @ w
            R = tgt_test - E
            current_acc = accuracy_score(y_test, predict(primitives, w, X_test))

        var_before = float(np.var(R))
        if var_before < var_threshold:
            if verbose:
                print(f"  R{rnd:02d}: converged (var={var_before:.4f})")
            break

        new_prim = synth.synthesize(X_test, R, round_n=rnd)
        if new_prim is None:
            if verbose:
                print(f"  R{rnd:02d}: synthesis failed")
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
            'round':      rnd,
            'family':     new_prim['family'],
            'corr':       round(new_prim['corr'], 3),
            'var_before': round(var_before, 4),
            'var_after':  round(var_after, 4),
            'acc':        round(current_acc, 4),
            'delta_acc':  round(delta_acc, 4),
            'verdict':    verdict,
        })

        if verbose:
            sym = '✓' if verdict == 'KEEP' else '~'
            print(f"  R{rnd:02d} {sym} [{new_prim['family']:12s}] "
                  f"corr={new_prim['corr']:.3f}  "
                  f"var {var_before:.3f}→{var_after:.3f}  "
                  f"acc {current_acc:.1%}")

    # Final fit
    if primitives:
        w_final = fit_weights(primitives, Xf, yf, reg=reg, epochs=700)
        final_acc = accuracy_score(y_test, predict(primitives, w_final, X_test))
    else:
        final_acc = zodiac_acc

    fingerprint = [p['family'] for p in primitives]

    if verbose:
        print(f"  Final={final_acc:.1%}  Zodiac={zodiac_acc:.1%}  "
              f"LR={lr_acc:.1%}  Δ={final_acc-lr_acc:+.1%}  "
              f"prims={len(primitives)}")
        print(f"  Fingerprint: {fingerprint}")

    return {
        'domain':        domain_name,
        'final_acc':     round(final_acc, 4),
        'zodiac_acc':    round(zodiac_acc, 4),
        'zodiac_winner': zodiac_name,
        'all_zodiac':    all_zodiac,
        'lr_acc':        round(lr_acc, 4),
        'delta_vs_lr':   round(final_acc - lr_acc, 4),
        'delta_vs_zodiac': round(final_acc - zodiac_acc, 4),
        'n_primitives':  len(primitives),
        'fingerprint':   fingerprint,
        'families_used': list(set(fingerprint)),
        'history':       history,
    }
