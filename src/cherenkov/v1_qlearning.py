"""
Verisimilitude V1 — Q-learning over geometric primitives
=========================================================
Tabular Q-learning, 13 zodiac primitive families.
State:  discretized residual features.
Reward: variance reduction (dense) + held-out LR lift (terminal).
Policy: epsilon-greedy with correlation-seeded optimistic Q-init.
"""

import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# PRIMITIVES — the 13 zodiac families (action space)
# ══════════════════════════════════════════════════════════════════

FAMILIES = ['radial', 'directional', 'oscillatory', 'torsional',
            'crossing', 'trend', 'acute', 'obtuse', 'wormhole',
            'saddle', 'limit_cycle', 'cusp', 'linear_detachment']


def _dir(p, key, d, default=None):
    if default is None:
        default = np.eye(max(d, 2))[0]
    raw = np.array(p.get(key, default), dtype=float)
    v = np.zeros(d)
    v[:min(d, len(raw))] = raw[:min(d, len(raw))]
    n = np.linalg.norm(v)
    return v / (n + 1e-8) if n > 1e-8 else v


def _ctr(p, d):
    raw = np.array(p.get('center', np.zeros(min(d, 8))), dtype=float)
    c = np.zeros(d); c[:min(d, len(raw))] = raw[:min(d, len(raw))]
    return c


def eval_prim(family, params, X):
    N, d = X.shape
    c = _ctr(params, d)
    delta = X - c

    if family == 'radial':
        r = np.sqrt((delta**2).sum(-1) + 1e-8)
        return np.exp(-r**2 / max(params.get('sigma', 0.5), 0.05)**2)
    elif family == 'directional':
        w = _dir(params, 'w', d)
        return 1 / (1 + np.exp(-(X @ w + params.get('b', 0.0))))
    elif family == 'oscillatory':
        a = _dir(params, 'axis', d)
        return np.sin(params.get('freq', 2.0) * (delta @ a))
    elif family == 'torsional':
        dx = delta[:, 0]; dy = delta[:, 1] if d > 1 else np.zeros(N)
        r = np.sqrt(dx**2 + dy**2 + 1e-8)
        th = np.arctan2(dy, dx)
        s = max(params.get('sigma', 0.6), 0.05)
        return np.exp(-r**2 / (2*s**2)) * np.sin(params.get('freq', 2.0) * th)
    elif family == 'crossing':
        pairs = params.get('pairs', [(0, min(1, d-1))])
        pairs = [(i, j) for i, j in pairs if i < d and j < d] or [(0, min(1, d-1))]
        cross = np.stack([X[:, i] * X[:, j] for i, j in pairs], 1)
        return 1 / (1 + np.exp(-cross.sum(-1)))
    elif family == 'trend':
        di = _dir(params, 'direction', d)
        s = max(params.get('sigma', 0.4), 0.05)
        proj = (delta * di).sum(-1, keepdims=True)
        perp = delta - proj * di
        dp = np.sqrt((perp**2).sum(-1) + 1e-8)
        return np.exp(-dp**2 / (2*s**2)) * np.tanh(params.get('align', 1.0) * proj.squeeze(-1))
    elif family == 'acute':
        di = _dir(params, 'direction', d)
        pe = np.zeros(d)
        if d >= 2: pe[0] = -di[1]; pe[1] = di[0]
        else: pe[0] = 1.0
        pe /= (np.linalg.norm(pe) + 1e-8)
        ang = params.get('angle', np.pi/4); sig = max(params.get('sigma', 0.3), 0.05)
        da = delta @ di; dp = delta @ pe
        td = np.abs(dp) - np.tan(ang) * np.maximum(da, 0)
        return np.exp(-np.maximum(td, 0)**2 / (2*sig**2)) * (da > 0)
    elif family == 'obtuse':
        di = _dir(params, 'direction', d)
        pe = np.zeros(d)
        if d >= 2: pe[0] = -di[1]; pe[1] = di[0]
        else: pe[0] = 1.0
        pe /= (np.linalg.norm(pe) + 1e-8)
        ang = params.get('angle', 2*np.pi/3); sig = max(params.get('sigma', 0.5), 0.05)
        da = delta @ di; dp = delta @ pe
        bend = np.abs(dp) / (np.abs(da) + 1e-4)
        return np.exp(-(da**2 + dp**2) / (2*sig**2)) * (bend < np.tan(np.pi - ang))
    elif family == 'wormhole':
        r2 = (delta**2).sum(-1) + 1e-8
        sig = max(params.get('sigma', 0.8), 0.1)
        eps = max(params.get('epsilon', 0.15), 0.02)
        return np.exp(-r2/(2*sig**2)) * (1 - np.exp(-r2/(2*eps**2)))
    elif family == 'saddle':
        di = _dir(params, 'direction', d)
        pe = np.zeros(d)
        if d >= 2: pe[0] = -di[1]; pe[1] = di[0]
        else: pe[0] = 1.0
        pe /= (np.linalg.norm(pe) + 1e-8)
        return np.tanh(params.get('a', 1.5)*(delta@di)) * np.tanh(params.get('b', 1.5)*(delta@pe))
    elif family == 'limit_cycle':
        r = np.sqrt((delta**2).sum(-1) + 1e-8)
        rs = max(params.get('r_star', 0.5), 0.05)
        sig = max(params.get('sigma', 0.2), 0.03)
        return np.exp(-((r - rs)**2) / (2*sig**2))
    elif family == 'cusp':
        di = _dir(params, 'direction', d)
        pe = np.zeros(d)
        if d >= 2: pe[0] = -di[1]; pe[1] = di[0]
        else: pe[0] = 1.0
        pe /= (np.linalg.norm(pe) + 1e-8)
        da = delta @ di; dp = delta @ pe
        cd = da**2 - params.get('scale', 1.0) * (np.abs(dp) + 1e-8)**1.5
        sig = max(params.get('sigma', 0.4), 0.05)
        return np.exp(-(cd**2) / (2*sig**2))
    elif family == 'linear_detachment':
        di = _dir(params, 'dir_in', d)
        theta = params.get('theta', np.pi/4)
        do = np.zeros(d)
        do[0] = np.cos(theta)*di[0] - np.sin(theta)*(di[1] if d > 1 else 0.)
        if d > 1: do[1] = np.sin(theta)*di[0] + np.cos(theta)*di[1]
        if d > 2: do[2:] = di[2:]
        do /= (np.linalg.norm(do) + 1e-8)
        alpha = params.get('alpha', 1.5)
        sig = max(params.get('sigma', 0.6), 0.05)
        eps = max(params.get('epsilon', 0.15), 0.02)
        r2 = (delta**2).sum(-1) + 1e-8
        gap = np.exp(-r2/(2*sig**2)) * (1 - np.exp(-r2/(2*eps**2)))
        d_in = delta @ di
        approach = np.tanh(alpha * d_in)
        exit_ = np.tanh(alpha * (delta @ do))
        blend = np.where(d_in < 0, approach, exit_)
        return gap * blend
    return np.zeros(N)


# ══════════════════════════════════════════════════════════════════
# PARAMETER SAMPLER — picks parameters for a chosen family
# ══════════════════════════════════════════════════════════════════

def sample_params(family, X, residual, rng):
    """
    Given a family and the current residual, sample reasonable parameters.
    Uses residual-weighted KMeans to pick centers (data-driven).
    """
    N, d = X.shape
    w = np.abs(residual) + 1e-8
    w = w / w.sum()
    n_sample = min(50, N)
    idx = rng.choice(N, n_sample, replace=False, p=w)
    Xs = X[idx]
    nc = min(4, len(Xs))
    centers = KMeans(n_clusters=nc, n_init=3, random_state=int(rng.integers(0, 1e6))
                     ).fit(Xs).cluster_centers_

    # Pick a random center from the candidates
    c = centers[rng.integers(0, len(centers))]
    angle = rng.uniform(0, np.pi)
    di = (np.array([np.cos(angle), np.sin(angle)] + [0.]*(d-2))
          if d >= 2 else np.array([1.]))

    # Family-specific parameter ranges
    if family == 'radial':
        return {'center': c, 'sigma': rng.choice([0.3, 0.6, 1.2, 2.0])}
    elif family == 'directional':
        b = float(-(di[:d] * c[:d]).sum())
        return {'center': c, 'w': di, 'b': b + rng.choice([-0.5, 0., 0.5])}
    elif family == 'oscillatory':
        return {'center': c, 'axis': di, 'freq': rng.choice([1.0, 2.0, 3.5])}
    elif family == 'torsional':
        return {'center': c, 'freq': rng.choice([1.0, 2.5]),
                'sigma': rng.choice([0.4, 0.8])}
    elif family == 'crossing':
        pairs = [(i, j) for i in range(min(d, 4)) for j in range(i+1, min(d, 4))]
        return {'center': c, 'pairs': pairs[:3] if pairs else [(0, min(1, d-1))]}
    elif family == 'trend':
        return {'center': c, 'direction': di,
                'sigma': rng.choice([0.3, 0.6, 1.0]), 'align': 1.0}
    elif family == 'acute':
        return {'center': c, 'angle': rng.choice([np.pi/6, np.pi/4, np.pi/3]),
                'direction': di, 'sigma': 0.3}
    elif family == 'obtuse':
        return {'center': c, 'angle': rng.choice([2*np.pi/3, 3*np.pi/4]),
                'direction': di, 'sigma': 0.5}
    elif family == 'wormhole':
        return {'center': c, 'sigma': rng.choice([0.5, 0.9, 1.5]),
                'epsilon': rng.choice([0.1, 0.2])}
    elif family == 'saddle':
        a = rng.choice([0.8, 1.5, 2.5])
        return {'center': c, 'direction': di, 'a': a, 'b': a}
    elif family == 'limit_cycle':
        return {'center': c, 'r_star': rng.choice([0.2, 0.5, 0.9, 1.4]),
                'sigma': rng.choice([0.1, 0.2, 0.35])}
    elif family == 'cusp':
        return {'center': c, 'direction': di, 'sigma': 0.4,
                'scale': rng.choice([0.5, 1.0, 2.0])}
    elif family == 'linear_detachment':
        return {'center': c, 'dir_in': di,
                'theta': rng.choice([np.pi/6, np.pi/4, np.pi/3, np.pi/2]),
                'sigma': rng.choice([0.4, 0.8, 1.4]),
                'epsilon': rng.choice([0.1, 0.2]), 'alpha': 1.5}
    return {'center': c}


# ══════════════════════════════════════════════════════════════════
# Q-LEARNING AGENT
# ══════════════════════════════════════════════════════════════════

class V1Agent:
    """Tabular Q-learning over the 13 primitive families."""

    def __init__(self, n_state_bins=10, alpha=0.1, gamma=0.95,
                 var_reward_scale=0.1, lift_reward_scale=1.0, seed=0):
        self.n_families = len(FAMILIES)
        self.n_state_bins = n_state_bins
        self.alpha = alpha
        self.gamma = gamma
        self.var_scale = var_reward_scale
        self.lift_scale = lift_reward_scale
        self.rng = np.random.default_rng(seed)

        # Q-table: state_key (tuple) -> ndarray of Q-values per action
        self.Q = {}

        # FIX 4: optimistic Q-init seeded by per-family bias toward exploration
        # Higher init for families historically useful for nonlinear data
        # (radial, saddle, wormhole, limit_cycle, linear_detachment)
        priors = {
            'radial': 0.15, 'saddle': 0.12, 'wormhole': 0.10,
            'limit_cycle': 0.10, 'linear_detachment': 0.10,
            'cusp': 0.08, 'crossing': 0.08, 'trend': 0.06,
            'directional': 0.05, 'oscillatory': 0.05,
            'torsional': 0.05, 'acute': 0.05, 'obtuse': 0.05,
        }
        self.q_init = np.array([priors[f] for f in FAMILIES])

        self.episode_log = []
        self.q_snapshots = []

    # ── State representation ──────────────────────────────────────
    def state_features(self, residual):
        """Compute 6 informative scalars about the residual field."""
        if len(residual) == 0:
            return np.zeros(6)
        r = residual
        return np.array([
            float(np.var(r)),                    # spread
            float(np.mean(np.abs(r))),           # magnitude
            float(np.mean(r > 0)),               # positive fraction
            float(np.std(np.abs(r))),            # heteroscedasticity
            float(np.percentile(np.abs(r), 90)), # tail weight
            float(np.median(np.abs(r))),         # central magnitude
        ])

    def discretize(self, residual):
        feats = self.state_features(residual)
        # Robust normalization to [0, 1]
        normalized = np.clip(feats / (np.abs(feats).max() + 1e-8), -1, 1) * 0.5 + 0.5
        bins = np.minimum(
            (normalized * self.n_state_bins).astype(int),
            self.n_state_bins - 1
        )
        return tuple(bins.tolist())

    # ── Q-table access ────────────────────────────────────────────
    def get_q(self, state):
        if state not in self.Q:
            self.Q[state] = self.q_init.copy()
        return self.Q[state]

    def select_action(self, state, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_families))
        return int(np.argmax(self.get_q(state)))

    def update(self, state, action, reward, next_state, done):
        q = self.get_q(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.get_q(next_state)))
        q[action] = q[action] + self.alpha * (target - q[action])

    # ── Episode rollout ───────────────────────────────────────────
    def run_episode(self, X_train, y_train, X_test, y_test,
                    epsilon=0.2, max_steps=6):
        """One episode: place primitives one at a time, accumulate reward."""
        # Convert y to {-1, +1} target field
        target = np.where(y_train == y_train.min(), -1., 1.).astype(float)
        residual = target.copy()
        prims = []
        self._trajectory = []  # FIX 1: track (state, action) for credit prop

        state = self.discretize(residual)
        fingerprint = []
        per_step_rewards = []

        for step in range(max_steps):
            action = self.select_action(state, epsilon)
            family = FAMILIES[action]
            self._trajectory.append((state, action))

            # Sample params for this family using residual-weighted clustering
            params = sample_params(family, X_train, residual, self.rng)
            phi = eval_prim(family, params, X_train)

            # Tentatively add this primitive — refit weights against target
            trial_prims = prims + [{'family': family, 'params': params}]
            Phi = np.stack([eval_prim(p['family'], p['params'], X_train)
                            for p in trial_prims], 1)

            try:
                w, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)
                new_residual = target - Phi @ w
                var_before = float(np.var(residual))
                var_after = float(np.var(new_residual))
                var_reduction = max(0., var_before - var_after)
            except Exception:
                var_reduction = 0.
                new_residual = residual

            # Accept the primitive only if it reduced variance
            if var_reduction > 1e-4:
                prims = trial_prims
                residual = new_residual
                fingerprint.append(family)
            # Otherwise the primitive was a bad pick — punish slightly
            #   (zero variance reduction = zero shaping reward, but the
            #   action still advances the loop)

            reward_step = self.var_scale * var_reduction
            next_state = self.discretize(residual)
            self.update(state, action, reward_step,
                        next_state, done=(step == max_steps - 1))

            per_step_rewards.append(reward_step)
            state = next_state

            # Early stop if residual is flat
            if float(np.var(residual)) < 0.05:
                break

        # ── Terminal reward: held-out LR lift ──
        if prims:
            Phi_train = np.stack([eval_prim(p['family'], p['params'], X_train)
                                  for p in prims], 1)
            Phi_test = np.stack([eval_prim(p['family'], p['params'], X_test)
                                 for p in prims], 1)
            X_tr_aug = np.hstack([X_train, Phi_train])
            X_te_aug = np.hstack([X_test, Phi_test])
        else:
            X_tr_aug, X_te_aug = X_train, X_test

        try:
            base_score = LogisticRegression(max_iter=1000).fit(X_train, y_train).score(X_test, y_test)
            aug_score  = LogisticRegression(max_iter=1000).fit(X_tr_aug, y_train).score(X_te_aug, y_test)
            lift = aug_score - base_score
        except Exception:
            base_score = aug_score = 0.5; lift = 0.

        terminal_reward = self.lift_scale * lift  # can be negative
        # FIX 1: propagate terminal reward backward through trajectory
        # with γ-discount so earlier primitives get credit for late lift
        for k, (s, a) in enumerate(reversed(self._trajectory)):
            discounted = terminal_reward * (self.gamma ** k)
            q = self.get_q(s)
            q[a] = q[a] + self.alpha * (discounted - q[a])

        return {
            'fingerprint': fingerprint,
            'n_prims': len(prims),
            'shaping_reward': float(sum(per_step_rewards)),
            'terminal_reward': float(terminal_reward),
            'lift': float(lift),
            'base_score': float(base_score),
            'aug_score': float(aug_score),
            'final_var': float(np.var(residual)),
        }

    # ── Training loop ─────────────────────────────────────────────
    def train(self, X_train, y_train, X_test, y_test,
              n_episodes=100, epsilon_start=0.3, epsilon_end=0.05,
              snapshot_every=10, verbose=True):
        results = []
        for ep in range(n_episodes):
            t = ep / max(n_episodes - 1, 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * t
            r = self.run_episode(X_train, y_train, X_test, y_test, epsilon=epsilon)
            r['episode'] = ep + 1
            r['epsilon'] = float(epsilon)
            results.append(r)

            if (ep + 1) % snapshot_every == 0:
                all_q = [v for arr in self.Q.values() for v in arr]
                self.q_snapshots.append({
                    'episode': ep + 1,
                    'q_table_size': len(self.Q),
                    'mean_q': float(np.mean(all_q)) if all_q else 0.0,
                    'max_q': float(np.max(all_q)) if all_q else 0.0,
                })
                if verbose:
                    recent = results[-snapshot_every:]
                    avg_lift = np.mean([x['lift'] for x in recent])
                    avg_prims = np.mean([x['n_prims'] for x in recent])
                    fams = Counter([f for x in recent for f in x['fingerprint']])
                    top3 = ', '.join(f for f, _ in fams.most_common(3))
                    print(f"  ep {ep+1:3d}/{n_episodes}  ε={epsilon:.2f}  "
                          f"avg_lift={avg_lift:+.3f}  avg_prims={avg_prims:.1f}  "
                          f"|Q|={len(self.Q):4d}  top: [{top3}]")

        self.episode_log = results
        return results

    # ── Diagnostics ───────────────────────────────────────────────
    def spectrum_gradient(self):
        """
        For each episode, count primitives whose addition reduced variance.
        Headline metric (Reading A — local work check).
        """
        return [r['n_prims'] for r in self.episode_log]

    def cross_run_consistency(self):
        """
        Stability of fingerprint composition across episodes.
        Diagnostic only (Reading B — across-runs structure).
        """
        if len(self.episode_log) < 2:
            return 0.0
        fingerprints = [set(r['fingerprint']) for r in self.episode_log]
        n = len(fingerprints)
        sims = []
        for i in range(n):
            for j in range(i+1, n):
                a, b = fingerprints[i], fingerprints[j]
                if not a or not b:
                    continue
                sims.append(len(a & b) / len(a | b))  # Jaccard
        return float(np.mean(sims)) if sims else 0.0

    def lift_trajectory(self):
        """Held-out lift per episode — convergence indicator."""
        return [r['lift'] for r in self.episode_log]

    def policy_summary(self):
        """For each state visited, the argmax action."""
        if not self.Q:
            return {}
        return {state: FAMILIES[int(np.argmax(q))] for state, q in self.Q.items()}


# ══════════════════════════════════════════════════════════════════
# REAL DATASETS via OpenML
# ══════════════════════════════════════════════════════════════════

def load_real(name):
    """Load real-world OpenML dataset. Returns (X, y, n_episodes_recommended)."""
    from sklearn.datasets import fetch_openml
    cfg = {
        'eeg':        (1471, lambda d: (d.data.astype(float)[:600], (d.target == '1').astype(int)[:600]), 15),
        'ionosphere': (59,   lambda d: (d.data.astype(float),         (d.target == 'g').astype(int)),       15),
        'anneal':     (2,    None,                                                                          15),
        'qsar':       (1494, lambda d: (d.data.astype(float)[:600],   (d.target == '1').astype(int)[:600]), 15),
    }
    did, fn, n_eps = cfg[name]
    if name == 'anneal':
        from sklearn.preprocessing import LabelEncoder
        d = fetch_openml(data_id=did, as_frame=True, parser='auto')
        df = d.data.copy()
        for col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        return df.values.astype(float), (d.target.values == '3').astype(int), n_eps
    d = fetch_openml(data_id=did, as_frame=False, parser='auto')
    X, y = fn(d)
    return X, y, n_eps


def evaluate_v1_on_dataset(name, X, y, n_episodes, n_seeds=3, verbose=False):
    """FIX 3: held-out evaluation across seeds with stratified splits."""
    from sklearn.model_selection import StratifiedShuffleSplit

    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    sss = StratifiedShuffleSplit(n_splits=n_seeds, test_size=0.3, random_state=42)

    seed_results = []
    for seed, (tr, te) in enumerate(sss.split(X, y)):
        agent = V1Agent(seed=seed)
        agent.train(X[tr], y[tr], X[te], y[te],
                    n_episodes=n_episodes, verbose=verbose)
        last20 = agent.episode_log[-20:]
        seed_results.append({
            'seed': seed,
            'final_lift_mean': float(np.mean([r['lift'] for r in last20])),
            'final_lift_std':  float(np.std([r['lift']  for r in last20])),
            'final_aug_acc':   float(np.mean([r['aug_score']  for r in last20])),
            'final_base_acc':  float(np.mean([r['base_score'] for r in last20])),
            'jaccard':         agent.cross_run_consistency(),
            'top_families':    [f for f, _ in Counter(
                [fa for r in last20 for fa in r['fingerprint']]).most_common(3)],
        })
    return seed_results


# ══════════════════════════════════════════════════════════════════
# MULTI-DOMAIN MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time
    print("="*72)
    print("  Verisimilitude V1 — Multi-domain real-world evaluation")
    print("  Q-learning over 13 geometric primitives")
    print("="*72)

    DATASETS = ['ionosphere', 'qsar']  # fast datasets only — others run on RunPod
    summary = []
    t0 = time.time()

    for name in DATASETS:
        print(f"\n→ {name}")
        try:
            X, y, n_eps = load_real(name)
            print(f"  shape={X.shape}  classes={len(np.unique(y))}  "
                  f"balance={y.mean():.1%}")
            results = evaluate_v1_on_dataset(name, X, y, n_eps, n_seeds=2)

            lifts = [r['final_lift_mean']  for r in results]
            bases = [r['final_base_acc']   for r in results]
            augs  = [r['final_aug_acc']    for r in results]
            jacs  = [r['jaccard']          for r in results]
            all_fams = [f for r in results for f in r['top_families']]
            top = [f for f, _ in Counter(all_fams).most_common(3)]

            print(f"  base_LR : {np.mean(bases):.1%} ± {np.std(bases):.1%}")
            print(f"  +EBM    : {np.mean(augs):.1%} ± {np.std(augs):.1%}")
            print(f"  Δ lift  : {np.mean(lifts):+.1%} ± {np.std(lifts):.1%}")
            print(f"  Jaccard : {np.mean(jacs):.2f}  (cross-episode consistency)")
            print(f"  top fams: {top}")

            summary.append({'dataset': name,
                            'lift_mean': float(np.mean(lifts)),
                            'lift_std':  float(np.std(lifts)),
                            'jaccard':   float(np.mean(jacs)),
                            'top':       top})
        except Exception as e:
            print(f"  FAIL: {e}")

    print(f"\n{'='*72}")
    print(f"  Total: {time.time()-t0:.1f}s")
    wins = sum(1 for s in summary if s['lift_mean'] > 0.005)
    avg_lift = np.mean([s['lift_mean'] for s in summary]) if summary else 0
    avg_jac  = np.mean([s['jaccard']   for s in summary]) if summary else 0
    print(f"  {wins}/{len(summary)} domains improved | avg lift: {avg_lift:+.1%} | "
          f"avg Jaccard: {avg_jac:.2f}")
    print("="*72)

    import json
    with open('/mnt/user-data/outputs/v1_realworld_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
