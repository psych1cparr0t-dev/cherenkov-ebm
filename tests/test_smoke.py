"""
Quick smoke test — run this after pip install -e .
"""
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression

def normalize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)

def test_parsing_layer():
    from cherenkov import CherenkovParsingLayer

    X, y = make_circles(600, noise=0.05, factor=0.5, random_state=42)
    X = normalize(X)

    layer = CherenkovParsingLayer(n_shot=20, max_rounds=8, verbose=True)
    X_aug = layer.fit_transform(X, y)

    lr_raw = LogisticRegression(max_iter=1000).fit(X, y)
    lr_aug = LogisticRegression(max_iter=1000).fit(X_aug, y)

    acc_raw = (lr_raw.predict(X) == y).mean()
    acc_aug = (lr_aug.predict(X_aug) == y).mean()

    print(f"\nCircles — LR(X): {acc_raw:.1%}  LR(X_aug): {acc_aug:.1%}  "
          f"Δ={acc_aug-acc_raw:+.1%}")
    print(f"Fingerprint: {layer.fingerprint_}")
    print(f"Layer: {layer.describe()}")

    assert acc_aug > acc_raw + 0.20, \
        f"Parsing layer should improve by >20pp on circles, got {acc_aug-acc_raw:.1%}"
    print("\n✓ test_parsing_layer passed")


def test_sklearn_pipeline():
    from cherenkov import CherenkovParsingLayer
    from sklearn.pipeline import Pipeline

    X, y = make_moons(400, noise=0.15, random_state=42)
    X = normalize(X)

    pipe = Pipeline([
        ('parser', CherenkovParsingLayer(n_shot=20, max_rounds=6)),
        ('clf',    LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    print(f"\nMoons pipeline accuracy: {acc:.1%}")
    assert acc > 0.85, f"Expected >85% on moons, got {acc:.1%}"
    print("✓ test_sklearn_pipeline passed")


def test_primitive_eval():
    from cherenkov import eval_primitive
    import numpy as np

    X = np.random.randn(50, 2)
    for family in ['radial','directional','oscillatory','torsional',
                   'crossing','trend','acute','obtuse','wormhole',
                   'saddle','limit_cycle','cusp']:
        params = {
            'center': np.zeros(2),
            'w': np.array([1.,0.]),
            'direction': np.array([1.,0.]),
            'axis': np.array([1.,0.]),
        }
        phi = eval_primitive(family, params, X)
        assert phi.shape == (50,), f"{family}: wrong shape {phi.shape}"
        assert not np.any(np.isnan(phi)), f"{family}: NaN in output"
    print("✓ test_primitive_eval passed — all 12 families")


if __name__ == '__main__':
    print("Running Cherenkov EBM smoke tests...\n")
    test_primitive_eval()
    test_parsing_layer()
    test_sklearn_pipeline()
    print("\nAll tests passed.")
