

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from torch import nn

from churn.config import RANDOM_STATE, REPORTS_DIR
from churn.data import load_data, split_xy
from churn.evaluate import classification_report
from churn.train_models import make_pipeline


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """A 2-hidden-layer neural network: inputs -> 32 -> 16 -> 1 churn score."""

    def __init__(self, hidden=(32, 16), dropout=0.2, lr=1e-3, batch_size=64,
                 max_epochs=200, patience=15, balanced=True, random_state=RANDOM_STATE):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.balanced = balanced
        self.random_state = random_state

    def _build(self, n_inputs: int) -> nn.Sequential:
        layers, size = [], n_inputs
        for h in self.hidden:
            layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(self.dropout)]
            size = h
        layers.append(nn.Linear(size, 1))  # one output: the churn "logit"
        return nn.Sequential(*layers)

    @staticmethod
    def _to_tensor(X) -> torch.Tensor:
        X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X, y = self._to_tensor(X), torch.tensor(np.asarray(y), dtype=torch.float32)
        self.classes_ = np.array([0, 1])

        # Hold back 10% of the training rows to decide when to stop (early stopping)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, stratify=y, random_state=self.random_state
        )
        self.model_ = self._build(X.shape[1])
        # pos_weight is PyTorch's version of class_weight="balanced"
        pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum() if self.balanced else torch.tensor(1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        best_loss, best_state, bad_epochs = float("inf"), None, 0
        self.history_ = {"train_loss": [], "val_loss": []}
        for _ in range(self.max_epochs):
            # --- one epoch: show the network every training row once, in shuffled batches
            self.model_.train()
            order = torch.randperm(len(X_tr))
            epoch_loss = 0.0
            for start in range(0, len(X_tr), self.batch_size):
                idx = order[start:start + self.batch_size]
                optimiser.zero_grad()                       # 1. clear old gradients
                logits = self.model_(X_tr[idx]).squeeze(1)  # 2. forward pass: predict
                loss = loss_fn(logits, y_tr[idx])           # 3. how wrong were we?
                loss.backward()                             # 4. backprop: who is to blame?
                optimiser.step()                            # 5. nudge the weights
                epoch_loss += loss.item() * len(idx)

            # --- check on the held-back rows (no learning here)
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val).squeeze(1), y_val).item()
            self.history_["train_loss"].append(epoch_loss / len(X_tr))
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_loss - 1e-4:
                best_loss, bad_epochs = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:  # stopped improving: stop training
                    break

        self.model_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(self._to_tensor(X)).squeeze(1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main() -> None:
    X, y = split_xy(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Training the neural network with 5-fold CV (5 networks)...")
    mlp = make_pipeline(TorchMLPClassifier(), engineered=True)
    prob = cross_val_predict(mlp, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    mlp_cv = classification_report(y_train, prob)

    tuned_gb = json.loads((REPORTS_DIR / "tuning_results.json").read_text())["tuned_cv"]
    cols = ["roc_auc", "pr_auc", "recall", "precision", "churners_caught_top20pct"]
    table = pd.DataFrame({"gradient boosting (tuned)": tuned_gb, "neural network": mlp_cv}).T
    print("\n5-fold CV on the training set (both class-balanced):\n")
    print(table[cols].round(3).to_string())

    # Train once on all training data to plot the learning curve
    mlp.fit(X_train, y_train)
    hist = mlp.named_steps["clf"].history_
    plt.figure(figsize=(7, 4))
    plt.plot(hist["train_loss"], label="training loss")
    plt.plot(hist["val_loss"], label="validation loss")
    best_epoch = int(np.argmin(hist["val_loss"]))
    plt.axvline(best_epoch, ls="--", color="grey", label=f"best epoch = {best_epoch}")
    plt.xlabel("epoch")
    plt.ylabel("loss (lower is better)")
    plt.title("Neural network learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "figures" / "nn_learning_curve.png", dpi=120)
    print(f"\nStopped after {len(hist['val_loss'])} epochs (best: epoch {best_epoch}).")
    print("Saved reports/figures/nn_learning_curve.png")

    (REPORTS_DIR / "neural_net_results.json").write_text(json.dumps(mlp_cv, indent=2))


if __name__ == "__main__":
    main()
