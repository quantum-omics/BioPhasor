"""
biophasor.ml.classifier — PhasorClassifier (sklearn-compatible VPC wrapper).

Blueprint from Notebook 2.1 (CLL multi-omics classification):

    Pipeline:
        1. Encode each omics layer: φ = π·tanh((log1p(x)-μ)/σ)
        2. Concatenate → X_phase (n_samples × n_total_features)
        3. Wrap phasorflow.VPC in sklearn-compatible interface
        4. Evaluate with 5-fold stratified cross-validation (AUC-ROC)

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class PhasorClassifier:
    """
    sklearn-compatible classifier wrapping **phasorflow.VPC** (Variational
    Phasor Circuit).

    If phasorflow is not available, falls back to a logistic regression on
    the phase-encoded features (same pipeline, weaker model).

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    n_layers : int
        Number of VPC layers (equivalent to depth).
    lr : float
        Learning rate for VPC optimiser.
    epochs : int
        Training epochs.
    seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> clf = PhasorClassifier(n_classes=2, n_layers=4, epochs=80)
    >>> clf.fit(X_phase, y)
    >>> proba = clf.predict_proba(X_test)     # shape (n_test, n_classes)
    >>> auc = clf.score(X_test, y_test)        # AUC-ROC (binary) or accuracy
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_layers: int = 4,
        lr: float = 5e-3,
        epochs: int = 80,
        seed: int = 42,
        force_fallback: bool = False,
    ) -> None:
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.force_fallback = force_fallback
        self._model = None
        self._backend: str = "unknown"
        self.classes_: Optional[np.ndarray] = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhasorClassifier":
        """
        Fit the classifier on phase-encoded feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Phase-encoded omics features (output of tanh_phase_encode).
        y : np.ndarray, shape (n_samples,)
            Integer class labels.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if self.n_classes != n_classes:
            self.n_classes = n_classes

        if self.force_fallback:
            self._fit_fallback(X, y)
            self._backend = "logistic"
        else:
            try:
                self._fit_vpc(X, y)
                self._backend = "vpc"
            except ImportError:
                self._fit_fallback(X, y)
                self._backend = "logistic"

        return self

    def _fit_vpc(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using phasorflow.VPC."""
        import torch
        from phasorflow import VPC  # type: ignore[import]

        torch.manual_seed(self.seed)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        self._model = VPC(
            num_features=X.shape[1],
            num_layers=self.n_layers,
            num_classes=self.n_classes,
        )
        self._model.fit(X_t, y_t, epochs=self.epochs, lr=self.lr, verbose=False)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to sklearn LogisticRegression."""
        from sklearn.linear_model import LogisticRegression
        self._model = LogisticRegression(max_iter=500, random_state=self.seed)
        self._model.fit(X, y)

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        if self._backend == "vpc":
            import torch
            X_t = torch.tensor(X, dtype=torch.float32)
            proba = self._model.predict_proba(X_t)
            if isinstance(proba, torch.Tensor):
                proba = proba.detach().cpu().numpy()
            # Binary VPC may return (n_samples,) → reshape
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            return proba

        # Fallback
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute AUC-ROC (binary) or accuracy (multi-class).

        Falls back to accuracy when AUC is undefined (e.g. single-class test set).

        Returns
        -------
        float
        """
        import warnings
        from sklearn.metrics import roc_auc_score, accuracy_score
        proba = self.predict_proba(X)
        if self.n_classes == 2 and len(np.unique(y)) == 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    return roc_auc_score(y, proba[:, 1])
            except Exception:
                pass
        return accuracy_score(y, self.predict(X))

    # ── Cross-validation ──────────────────────────────────────────────────────

    def cross_val_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "roc_auc",
        seed: int = 42,
    ) -> np.ndarray:
        """
        Stratified k-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        cv : int   number of folds
        scoring : str  'roc_auc' or 'accuracy'
        seed : int

        Returns
        -------
        np.ndarray, shape (cv,)   per-fold scores
        """
        from sklearn.model_selection import StratifiedKFold

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        scores = []
        for train_idx, test_idx in kf.split(X, y):
            clone = PhasorClassifier(
                n_classes=self.n_classes,
                n_layers=self.n_layers,
                lr=self.lr,
                epochs=self.epochs,
                seed=self.seed,
                force_fallback=self.force_fallback,
            )
            clone.fit(X[train_idx], y[train_idx])
            scores.append(clone.score(X[test_idx], y[test_idx]))
        return np.array(scores)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PhasorClassifier(n_classes={self.n_classes}, n_layers={self.n_layers}, "
            f"epochs={self.epochs}, force_fallback={self.force_fallback}, backend={self._backend!r})"
        )
