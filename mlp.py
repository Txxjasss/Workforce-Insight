"""
A small, self-contained deep-learning model for the attrition task.

`TorchMLPClassifier` is a PyTorch multilayer perceptron wrapped in a
scikit-learn-compatible estimator (``fit`` / ``predict`` / ``predict_proba``)
so it drops straight into the same cross-validation, GridSearch and
joblib-persistence machinery the classical models use — and is served by the
Flask API through the identical ``predict_proba`` interface.

Design choices that matter on this dataset (1,470 rows, ~16% positive,
49 features):
  - Standardize inputs (NNs are scale-sensitive; trees are not).
  - Class-weighted BCE loss to counter the imbalance, mirroring the
    ``class_weight='balanced'`` used by the sklearn models.
  - Early stopping on a held-out validation slice to avoid overfitting such a
    small table.
  - Fully picklable (the net lives on CPU; numpy/torch state serializes), so
    ``joblib.dump`` / ``joblib.load`` works exactly like a sklearn model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-style wrapper around a PyTorch MLP for binary classification.

    Mixin precedes BaseEstimator so sklearn's tag system correctly identifies
    this as a classifier (required for ``scoring='roc_auc'`` in cross-validation).
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden=(128, 64),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 15,
        balance: bool = True,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.balance = balance
        self.random_state = random_state
        self.verbose = verbose

    # -- training -----------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)

        # Validation slice for early stopping.
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs, y, test_size=0.15, random_state=self.random_state, stratify=y
        )

        device = torch.device("cpu")
        self.model_ = _MLP(self.n_features_in_, tuple(self.hidden), self.dropout).to(device)

        pos_weight = None
        if self.balance:
            n_pos = float((y_tr == 1).sum())
            n_neg = float((y_tr == 0).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        Xt = torch.from_numpy(X_tr)
        yt = torch.from_numpy(y_tr)
        Xv = torch.from_numpy(X_val)
        yv = torch.from_numpy(y_val)

        best_val = float("inf")
        best_state = None
        wait = 0
        n = Xt.shape[0]

        for epoch in range(self.max_epochs):
            self.model_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                if idx.numel() < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                logits = self.model_(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xv), yv).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
            if self.verbose:
                print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    # -- inference ----------------------------------------------------------
    def _logits(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(Xs)).numpy()

    def predict_proba(self, X):
        p1 = 1.0 / (1.0 + np.exp(-self._logits(X)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # -- picklability -------------------------------------------------------
    # The torch module pickles fine via its state; nothing custom needed.
    # Keeping the net on CPU (above) guarantees portable joblib artifacts.
