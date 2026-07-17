"""
Tests for biophasor.ml.classifier — PhasorClassifier.

Using small sample sizes for fast test execution.
"""

import numpy as np
import pytest
from biophasor.ml.classifier import PhasorClassifier
from biophasor.transform.encoder import tanh_phase_encode


RNG = np.random.RandomState(99)


def _make_synthetic(n=60, n_features=30):
    """Balanced, SHUFFLED dataset: always both classes in any split."""
    half = n // 2
    y = np.array([0] * half + [1] * half)
    X = RNG.lognormal(0, 0.5, (n, n_features))
    X[half:] *= 5.0   # strong separation
    # Shuffle so train/test splits are always balanced
    idx = RNG.permutation(n)
    return tanh_phase_encode(X[idx]), y[idx]


class TestPhasorClassifier:
    def setup_method(self):
        self.X, self.y = _make_synthetic()
        # Split: first 40 train, last 20 test (both have 10 of each class)
        self.X_tr, self.y_tr = self.X[:40], self.y[:40]
        self.X_te, self.y_te = self.X[40:], self.y[40:]

    def test_fit_returns_self(self):
        clf = PhasorClassifier(n_classes=2, epochs=2)
        assert clf.fit(self.X_tr, self.y_tr) is clf

    def test_predict_shape(self):
        clf = PhasorClassifier(n_classes=2, epochs=2)
        clf.fit(self.X_tr, self.y_tr)
        assert clf.predict(self.X_te).shape == (20,)

    def test_predict_proba_shape(self):
        clf = PhasorClassifier(n_classes=2, epochs=2)
        clf.fit(self.X_tr, self.y_tr)
        assert clf.predict_proba(self.X_te).shape == (20, 2)

    def test_predict_proba_sums_to_one(self):
        clf = PhasorClassifier(n_classes=2, epochs=2)
        clf.fit(self.X_tr, self.y_tr)
        np.testing.assert_allclose(clf.predict_proba(self.X_te).sum(axis=1), 1.0, atol=1e-5)

    def test_score_above_chance(self):
        """Score must be > 0.5 on separable data (uses sklearn LogisticRegression for reliability)."""
        clf = PhasorClassifier(n_classes=2, epochs=2, force_fallback=True)
        clf.fit(self.X_tr, self.y_tr)
        s = clf.score(self.X_te, self.y_te)
        assert s > 0.5, f"Score {s:.3f} not above chance"

    def test_classes_attribute(self):
        clf = PhasorClassifier(n_classes=2, epochs=2)
        clf.fit(self.X_tr, self.y_tr)
        assert set(clf.classes_) == {0, 1}

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PhasorClassifier(n_classes=2).predict_proba(self.X_te)

    def test_cross_val_score_shape(self):
        # Use 2-fold CV on the full 60-sample balanced dataset
        clf = PhasorClassifier(n_classes=2, epochs=2)
        scores = clf.cross_val_score(self.X, self.y, cv=2)
        assert scores.shape == (2,)

    def test_cross_val_mean_above_chance(self):
        clf = PhasorClassifier(n_classes=2, epochs=2, force_fallback=True)
        scores = clf.cross_val_score(self.X, self.y, cv=2)
        assert scores.mean() > 0.5, f"CV mean {scores.mean():.3f}"
