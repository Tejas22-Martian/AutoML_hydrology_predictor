"""
LSTM rainfall-runoff model (PyTorch), wrapped in a scikit-learn style API.

WHY AN LSTM IS THE KEY UPGRADE
------------------------------
Streamflow has *memory*: today's flow depends on weeks of past rainfall stored
as soil moisture, snowpack and groundwater. Tree models (RF/XGBoost) only see
the lag features we hand-craft; they cannot learn arbitrary temporal dynamics.

A Long Short-Term Memory (LSTM) network is a recurrent neural network with
gated memory cells (input, forget and output gates) that learn *what to
remember and for how long*. In the hydrology literature the LSTM is the
state-of-the-art rainfall-runoff model: Kratzert et al. (2018, 2019) showed a
single LSTM trained on the CAMELS dataset beats calibrated conceptual
hydrological models, and beats them even for ungauged basins. Citing and
implementing this is exactly what lifts the project from "applied ML" to
"current hydrological ML research".

HOW THE GATES WORK (viva answer)
--------------------------------
At each timestep the cell maintains a state c_t.
  - forget gate f_t decides what fraction of the old state to keep,
  - input gate i_t decides what new information to write,
  - output gate o_t decides what to expose as the hidden state h_t.
Because the cell state is updated additively (c_t = f_t*c_{t-1} + i_t*g_t), the
gradient can flow over hundreds of timesteps without vanishing - that is why an
LSTM captures long catchment memory that a plain RNN or a feed-forward net cannot.

INTEGRATION NOTE
----------------
To slot into the existing pipeline we accept the standard 2-D feature matrix and
internally build overlapping sequences of `lookback` consecutive days, so the
LSTM sees a multivariate time window ending on the prediction day. (A "purist"
LSTM would consume raw forcings rather than pre-engineered lag features; we keep
the engineered features so all models share one preprocessing path, and we note
the trade-off in the report.)
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger("streamflow_automl.models.lstm")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class LSTMRegressor(RegressorMixin, BaseEstimator):
    """
    scikit-learn compatible LSTM regressor (fit / predict / get_params).

    Inheriting from BaseEstimator/RegressorMixin gives us get_params, set_params,
    clone() support and the estimator tags that sklearn's cross_val_score needs,
    so the LSTM works inside the AutoML machinery with no special-casing.
    """

    def __init__(self, lookback: int = 30, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2,
                 learning_rate: float = 1e-3, batch_size: int = 128,
                 max_epochs: int = 60, patience: int = 8,
                 random_state: int = 42):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    # ----- core -------------------------------------------------------------
    @staticmethod
    def _resolve_device():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _gather_windows(self, X_t, end_idx):
        """Build (b, lookback, f) windows on the fly for a batch of end positions.

        Windows are materialized lazily per batch (never all at once) so memory
        stays O(batch * lookback * features) instead of O(n * lookback * features).
        """
        import torch
        offsets = torch.arange(self.lookback, device=X_t.device)
        starts = end_idx - (self.lookback - 1)
        gather_idx = starts.unsqueeze(1) + offsets.unsqueeze(0)  # (b, lookback)
        return X_t[gather_idx]                                    # (b, lookback, f)

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, self._n_features = X.shape
        if n < self.lookback:
            raise ValueError(f"Not enough samples ({n}) for lookback={self.lookback}")

        self._device = self._resolve_device()
        X_t = torch.tensor(X, device=self._device)
        y_t = torch.tensor(y, device=self._device)

        # End positions of valid windows, split chronologically (val tail for
        # early stopping). No giant sequence array is ever allocated.
        ends = torch.arange(self.lookback - 1, n, device=self._device)
        n_val = max(1, int(0.1 * len(ends)))
        train_ends = ends[:-n_val]
        val_ends = ends[-n_val:]

        self._model = _LSTMNet(
            self._n_features, self.hidden_size, self.num_layers, self.dropout
        ).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_val = np.inf
        best_state = None
        bad_epochs = 0

        for _ in range(self.max_epochs):
            self._model.train()
            perm = train_ends[torch.randperm(len(train_ends), device=self._device)]
            for start in range(0, len(perm), self.batch_size):
                batch_ends = perm[start: start + self.batch_size]
                xb = self._gather_windows(X_t, batch_ends)
                yb = y_t[batch_ends].unsqueeze(1)
                optimizer.zero_grad()
                loss = loss_fn(self._model(xb), yb)
                loss.backward()
                optimizer.step()

            val_loss = self._batched_loss(X_t, y_t, val_ends, loss_fn)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break  # early stopping - prevents overfitting

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def _batched_loss(self, X_t, y_t, ends, loss_fn):
        import torch
        self._model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(ends), self.batch_size):
                be = ends[start: start + self.batch_size]
                pred = self._model(self._gather_windows(X_t, be))
                total += loss_fn(pred, y_t[be].unsqueeze(1)).item() * len(be)
                count += len(be)
        return total / max(count, 1)

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        device = getattr(self, "_device", self._resolve_device())
        X_t = torch.tensor(X, device=device)
        n = X_t.shape[0]
        ends = torch.arange(self.lookback - 1, n, device=device)

        self._model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(ends), self.batch_size):
                be = ends[start: start + self.batch_size]
                chunks.append(self._model(self._gather_windows(X_t, be)).cpu().numpy().ravel())
        preds = np.concatenate(chunks) if chunks else np.array([])
        # the first (lookback-1) rows have no full window; pad with the first
        # prediction so the output length matches the input length.
        pad = np.full(self.lookback - 1, preds[0]) if len(preds) else np.array([])
        return np.concatenate([pad, preds])


def _make_lstm_net(n_features, hidden_size, num_layers, dropout):
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)          # out: (batch, lookback, hidden)
            last = out[:, -1, :]           # take the final timestep's hidden state
            return self.head(last)

    return LSTMNet()


# lazily-built network class proxy so the module imports without torch present
def _LSTMNet(n_features, hidden_size, num_layers, dropout):
    return _make_lstm_net(n_features, hidden_size, num_layers, dropout)
