"""
Cherenkov EBM — Primitive Evaluation
=====================================
12 Chinese Zodiac primitive families.

Each primitive is a scalar basis function φ: R^d → R.
The EBM energy is E(x) = Σᵢ wᵢ · φᵢ(x).

Families
--------
Original 6 (v3.0):
  radial       Horse  — isotropic blob attractor
  directional  Rooster — half-space sigmoid separator
  oscillatory  Snake  — repeating wave along axis
  torsional    Dragon — rotational spiral decay
  crossing     Goat   — quadratic cross-boundary
  trend        Ox     — directed momentum, lateral decay

Deflection family (v4.0):
  acute        Tiger  — sharp V-kink, angle < 90°
  obtuse       Rabbit — shallow bend, angle > 90°

Topological (v4.0):
  wormhole     Rat    — catenoid throat, regime jump

Bifurcation family (v4.1):
  saddle       Monkey — attracts one axis, repels another
  limit_cycle  Dog    — Hopf ring, closed orbit
  cusp         Pig    — catastrophe fold, two basins merge

Composition:
  product      —      — φ_a · φ_b cutout shape
"""

import numpy as np


ZODIAC_NAMES = {
    'radial':      'Horse',
    'directional': 'Rooster',
    'oscillatory': 'Snake',
    'torsional':   'Dragon',
    'crossing':    'Goat',
    'trend':       'Ox',
    'acute':       'Tiger',
    'obtuse':      'Rabbit',
    'wormhole':    'Rat',
    'saddle':      'Monkey',
    'limit_cycle': 'Dog',
    'cusp':        'Pig',
    'product':     'Composite',
}

ALL_FAMILIES = list(ZODIAC_NAMES.keys())


def eval_primitive(family: str, params: dict, X: np.ndarray) -> np.ndarray:
    """
    Evaluate primitive family at points X.

    Parameters
    ----------
    family : str
        One of the 12 zodiac family names (or 'product').
    params : dict
        Family-specific parameters (center, sigma, etc.)
    X : np.ndarray, shape (N, d)
        Input points.

    Returns
    -------
    phi : np.ndarray, shape (N,)
        Primitive values at each point.
    """
    N, d = X.shape
    c_raw = params.get('center', np.zeros(d))
    c = np.zeros(d)
    c[:len(c_raw)] = c_raw[:d]

    # ── Horse: radial blob ───────────────────────────────────────
    if family == 'radial':
        diff = X - c
        r = np.sqrt((diff**2).sum(-1) + 1e-8)
        p = np.clip(params.get('power', 2.0), 0.5, 4.0)
        s = max(params.get('sigma', 0.5), 0.05)
        return np.exp(-(r**p) / (s**p + 1e-8))

    # ── Rooster: directional half-space ─────────────────────────
    elif family == 'directional':
        w_raw = params['w']
        w = np.zeros(d); w[:len(w_raw)] = w_raw[:d]
        w = w / (np.linalg.norm(w) + 1e-8)
        b = params.get('b', 0.0)
        return 1.0 / (1.0 + np.exp(-(X @ w + b)))

    # ── Snake: oscillatory wave ──────────────────────────────────
    elif family == 'oscillatory':
        a_raw = params.get('axis', np.ones(min(2,d))/np.sqrt(min(2,d)))
        a = np.zeros(d); a[:len(a_raw)] = a_raw[:d]
        a = a / (np.linalg.norm(a) + 1e-8)
        f = params.get('freq', 2.0)
        return np.sin(f * (X - c) @ a)

    # ── Dragon: torsional spiral ─────────────────────────────────
    elif family == 'torsional':
        dx = X[:,0] - c[0]
        dy = X[:,1] - c[1] if d > 1 else np.zeros(N)
        r = np.sqrt(dx**2 + dy**2 + 1e-8)
        theta = np.arctan2(dy, dx)
        s = max(params.get('sigma', 0.6), 0.05)
        f = params.get('freq', 2.0)
        return np.exp(-r**2 / (2*s**2)) * np.sin(f * theta)

    # ── Goat: crossing / interaction ────────────────────────────
    elif family == 'crossing':
        pairs = params.get('pairs', [(0, min(1,d-1))])
        pairs = [(i,j) for i,j in pairs if i<d and j<d]
        if not pairs: pairs = [(0, min(1,d-1))]
        W = params.get('W', np.ones(len(pairs)))
        b = params.get('b', 0.0)
        cross = np.stack([X[:,i]*X[:,j] for i,j in pairs], axis=1)
        return 1.0 / (1.0 + np.exp(-(cross @ W + b)))

    # ── Ox: trend / directed momentum ───────────────────────────
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

    # ── Tiger: acute deflection (V-kink < 90°) ──────────────────
    elif family == 'acute':
        angle = params.get('angle', np.pi/4)
        sigma = max(params.get('sigma', 0.3), 0.05)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2: perp[0] = -direction[1]; perp[1] = direction[0]
        else: perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        tip_dist = np.abs(d_perp) - np.tan(angle) * np.maximum(d_along, 0)
        phi = np.exp(-np.maximum(tip_dist, 0)**2 / (2*sigma**2))
        return phi * (d_along > 0).astype(float)

    # ── Rabbit: obtuse deflection (wide bend > 90°) ──────────────
    elif family == 'obtuse':
        angle = params.get('angle', 2*np.pi/3)
        sigma = max(params.get('sigma', 0.5), 0.05)
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2: perp[0] = -direction[1]; perp[1] = direction[0]
        else: perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        bend = np.abs(d_perp) / (np.abs(d_along) + 1e-4)
        threshold = np.tan(np.pi - angle)
        in_bend = (bend < threshold).astype(float)
        dist_to_apex = np.sqrt(d_along**2 + d_perp**2 + 1e-8)
        return np.exp(-dist_to_apex**2 / (2*sigma**2)) * in_bend

    # ── Rat: wormhole (catenoid throat) ──────────────────────────
    elif family == 'wormhole':
        sigma   = max(params.get('sigma', 0.8), 0.1)
        epsilon = max(params.get('epsilon', 0.15), 0.02)
        diff = X - c
        r2 = (diff**2).sum(-1) + 1e-8
        outer = np.exp(-r2 / (2*sigma**2))
        inner = np.exp(-r2 / (2*epsilon**2))
        return outer * (1.0 - inner)

    # ── Monkey: saddle point ─────────────────────────────────────
    elif family == 'saddle':
        dir_raw = params.get('direction', np.array([1.,0.]+[0.]*(d-2)))
        direction = np.zeros(d); direction[:len(dir_raw)] = dir_raw[:d]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perp = np.zeros(d)
        if d >= 2: perp[0] = -direction[1]; perp[1] = direction[0]
        else: perp[0] = 1.0
        perp = perp / (np.linalg.norm(perp) + 1e-8)
        a = params.get('a', 1.5)
        b = params.get('b', 1.5)
        delta = X - c
        d_along = (delta * direction).sum(-1)
        d_perp  = (delta * perp).sum(-1)
        return np.tanh(a * d_along) * np.tanh(b * d_perp)

    # ── Dog: limit cycle (Hopf ring) ─────────────────────────────
    elif family == 'limit_cycle':
        r_star = max(params.get('r_star', 0.5), 0.05)
        sigma  = max(params.get('sigma', 0.2), 0.03)
        diff = X - c
        r = np.sqrt((diff**2).sum(-1) + 1e-8)
        return np.exp(-((r - r_star)**2) / (2*sigma**2))

    # ── Pig: cusp (catastrophe fold) ─────────────────────────────
    elif family == 'cusp':
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

    # ── Composite: product of two primitives ─────────────────────
    elif family == 'product':
        phi_a = eval_primitive(params['family_a'], params['params_a'], X)
        phi_b = eval_primitive(params['family_b'], params['params_b'], X)
        return phi_a * phi_b

    return np.zeros(N)


def phi_matrix(primitives: list, X: np.ndarray) -> np.ndarray:
    """
    Build feature matrix Φ where Φ[:,k] = φₖ(X).

    Parameters
    ----------
    primitives : list of dicts
        Each dict has keys 'family' and 'params'.
    X : np.ndarray, shape (N, d)

    Returns
    -------
    Phi : np.ndarray, shape (N, K)
    """
    if not primitives:
        return np.zeros((len(X), 1))
    return np.stack(
        [eval_primitive(p['family'], p['params'], X) for p in primitives],
        axis=1
    )
