"""Shared test fixtures for StockLTSMTransformerQuantum.

CRITICAL: Environment variables and module mocks MUST be set before any
project imports. config.py runs validate_env() at import time, and models.py
imports pennylane and tensorflow at module level.
"""

import os
import sys

# ── Set env vars BEFORE any project imports ───────────────────────────
os.environ.setdefault("POLYGON_API_KEY", "test_polygon_key")
os.environ.setdefault("ALPHAVANTAGE_API_KEY", "test_alpha_key")
os.environ.setdefault("TICKERS", "AAPL,MSFT")

# ── Mock heavy dependencies that aren't installed in test env ─────────
from unittest.mock import MagicMock

# PennyLane
mock_pennylane = MagicMock()
mock_pennylane.device.return_value = MagicMock()
mock_pennylane.qnode = lambda dev: lambda fn: fn  # Decorator passthrough
mock_pennylane.RY = MagicMock()
mock_pennylane.RZ = MagicMock()
mock_pennylane.RX = MagicMock()
mock_pennylane.CNOT = MagicMock()
mock_pennylane.expval = MagicMock(return_value=0.5)
mock_pennylane.PauliZ = MagicMock()
mock_pennylane.AdamOptimizer = MagicMock()

# pennylane.numpy — needs to behave like numpy but with requires_grad
import numpy as _real_np
mock_qnp = MagicMock()
mock_qnp.random = MagicMock()
mock_qnp.random.random = lambda *args, **kwargs: _real_np.random.random(*args[:1] if args else ())
mock_qnp.array = lambda x, **kwargs: _real_np.array(x)
mock_qnp.mean = _real_np.mean
mock_qnp.save = MagicMock()
mock_qnp.load = MagicMock(return_value=_real_np.random.random(16))

sys.modules["pennylane"] = mock_pennylane
sys.modules["pennylane.numpy"] = mock_qnp

# TensorFlow / Keras
mock_tf = MagicMock()
mock_keras = MagicMock()
mock_keras_layers = MagicMock()
mock_keras_metrics = MagicMock()
mock_keras_models = MagicMock()
mock_keras_callbacks = MagicMock()

# Make metrics return MagicMock callables
mock_keras_metrics.MeanAbsoluteError = MagicMock
mock_keras_metrics.MeanAbsolutePercentageError = MagicMock

# Create a proper Callback base class so MetricsTrackingCallback inherits correctly
class _MockCallback:
    """Minimal Keras Callback stub for testing."""
    def __init__(self):
        self.model = None
    def set_model(self, model):
        self.model = model
    def on_epoch_end(self, epoch, logs=None):
        pass
    def on_train_begin(self, logs=None):
        pass
    def on_train_end(self, logs=None):
        pass

mock_keras_callbacks.Callback = _MockCallback
mock_keras_callbacks.ModelCheckpoint = MagicMock
mock_keras_callbacks.EarlyStopping = MagicMock

sys.modules["tensorflow"] = mock_tf
sys.modules["tensorflow.keras"] = mock_keras
sys.modules["tensorflow.keras.layers"] = mock_keras_layers
sys.modules["tensorflow.keras.metrics"] = mock_keras_metrics
sys.modules["tensorflow.keras.models"] = mock_keras_models
sys.modules["tensorflow.keras.callbacks"] = mock_keras_callbacks

# PyQt5
mock_pyqt = MagicMock()
sys.modules["PyQt5"] = mock_pyqt
sys.modules["PyQt5.QtCore"] = mock_pyqt
sys.modules["PyQt5.QtWidgets"] = mock_pyqt

# ta (technical analysis) — mock the momentum module
mock_ta = MagicMock()
mock_ta_momentum = MagicMock()
sys.modules["ta"] = mock_ta
sys.modules["ta.momentum"] = mock_ta_momentum
sys.modules["ta.trend"] = MagicMock()

# sklearn — provide real metric functions since MetricsTrackingCallback uses them
mock_sklearn = MagicMock()
mock_sklearn_preprocessing = MagicMock()
mock_sklearn_metrics = MagicMock()

def _mock_mae(y_true, y_pred):
    return float(_real_np.mean(_real_np.abs(_real_np.array(y_true).flatten() - _real_np.array(y_pred).flatten())))

def _mock_mape(y_true, y_pred):
    y_true, y_pred = _real_np.array(y_true).flatten(), _real_np.array(y_pred).flatten()
    return float(_real_np.mean(_real_np.abs((y_true - y_pred) / (y_true + 1e-10))))

def _mock_r2(y_true, y_pred):
    y_true, y_pred = _real_np.array(y_true).flatten(), _real_np.array(y_pred).flatten()
    ss_res = _real_np.sum((y_true - y_pred) ** 2)
    ss_tot = _real_np.sum((y_true - _real_np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))

mock_sklearn_metrics.mean_absolute_error = _mock_mae
mock_sklearn_metrics.mean_absolute_percentage_error = _mock_mape
mock_sklearn_metrics.r2_score = _mock_r2

sys.modules.setdefault("sklearn", mock_sklearn)
sys.modules.setdefault("sklearn.preprocessing", mock_sklearn_preprocessing)
sys.modules.setdefault("sklearn.metrics", mock_sklearn_metrics)

# matplotlib
mock_mpl = MagicMock()
sys.modules.setdefault("matplotlib", mock_mpl)
sys.modules.setdefault("matplotlib.pyplot", MagicMock())
sys.modules.setdefault("matplotlib.backends", MagicMock())
sys.modules.setdefault("matplotlib.backends.backend_qt5agg", MagicMock())

# tenacity — let it import naturally if available, otherwise mock
try:
    import tenacity
except ImportError:
    sys.modules["tenacity"] = MagicMock()

# yfinance
sys.modules.setdefault("yfinance", MagicMock())

# requests — let it import naturally if available
try:
    import requests
except ImportError:
    sys.modules["requests"] = MagicMock()

# ── Now safe to import standard test dependencies ─────────────────────
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def mock_scaler():
    """MinMaxScaler mock with n_features_in_ and inverse_transform."""
    scaler = MagicMock()
    scaler.n_features_in_ = 7
    scaler.scale_ = np.ones(7)

    def inverse_transform(x):
        return x * 100

    scaler.inverse_transform = inverse_transform
    return scaler


@pytest.fixture
def sample_predictions():
    """Monotonically increasing predictions for trading strategy tests."""
    return [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0]


@pytest.fixture
def flat_predictions():
    """Flat predictions (no trades expected)."""
    return [100.0] * 10


@pytest.fixture
def future_dates():
    """10 consecutive future dates starting from 2026-01-01."""
    base = datetime(2026, 1, 1)
    return [base + timedelta(days=i) for i in range(1, 11)]


@pytest.fixture
def last_date():
    """Base date for predictions."""
    return datetime(2026, 1, 1)


@pytest.fixture
def mock_stock_df():
    """DataFrame mimicking fetch_stock_data output with OHLCV columns."""
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    df = pd.DataFrame(
        {
            "Open": close - np.random.rand(200),
            "High": close + np.random.rand(200),
            "Low": close - np.random.rand(200),
            "Close": close,
            "Volume": np.random.randint(1000000, 10000000, 200),
        },
        index=dates,
    )
    return df


@pytest.fixture
def mock_keras_model():
    """Mock Keras model with predict method."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([[0.5]]))
    return model


@pytest.fixture
def sample_history():
    """Mock training history dict."""
    return {
        "loss": [0.1, 0.08, 0.06, 0.05],
        "val_loss": [0.12, 0.10, 0.08, 0.07],
        "train_mae": [0.05, 0.04, 0.03, 0.025],
        "val_mae": [0.06, 0.05, 0.04, 0.035],
        "val_mape": [2.0, 1.8, 1.5, 1.3],
        "val_r2": [0.85, 0.88, 0.90, 0.92],
        "val_sharpe": [0.5, 0.8, 1.0, 1.2],
    }
