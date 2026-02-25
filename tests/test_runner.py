"""Tests for training/runner.py — predict_future and MetricsTrackingCallback."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestPredictFuture:
    def test_output_length(self, mock_scaler):
        """predict_future should return exactly future_days predictions."""
        from training.runner import predict_future

        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([[0.5]]))

        last_sequence = np.random.rand(60, 7)
        result = predict_future(model, last_sequence, mock_scaler, future_days=10)
        assert len(result) == 10

    def test_predictions_non_negative(self, mock_scaler):
        """All predictions should be >= 0 (np.maximum applied)."""
        from training.runner import predict_future

        model = MagicMock()
        # Return negative values to test clamping
        model.predict = MagicMock(return_value=np.array([[-0.5]]))

        last_sequence = np.random.rand(60, 7)
        result = predict_future(model, last_sequence, mock_scaler, future_days=5)
        assert all(p >= 0 for p in result)

    def test_autoregressive_loop(self, mock_scaler):
        """Model.predict should be called once per future day."""
        from training.runner import predict_future

        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([[0.5]]))

        last_sequence = np.random.rand(60, 7)
        predict_future(model, last_sequence, mock_scaler, future_days=7)
        assert model.predict.call_count == 7

    def test_scaler_inverse_transform_called(self):
        """Scaler.inverse_transform should be called to convert back to price."""
        from training.runner import predict_future

        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([[0.5]]))
        scaler = MagicMock()
        scaler.scale_ = np.ones(7)
        scaler.inverse_transform = MagicMock(
            return_value=np.array([[150.0] + [0.0] * 6])
        )

        last_sequence = np.random.rand(60, 7)
        predict_future(model, last_sequence, scaler, future_days=1)
        scaler.inverse_transform.assert_called_once()


class TestMetricsTrackingCallback:
    def test_history_populated(self):
        """on_epoch_end should populate all history keys."""
        from training.runner import MetricsTrackingCallback

        n_train = 50
        n_val = 20
        X_train = np.random.rand(n_train, 60, 7)
        y_train = np.random.rand(n_train)
        X_val = np.random.rand(n_val, 60, 7)
        y_val = np.random.rand(n_val)

        callback = MetricsTrackingCallback(X_train, y_train, X_val, y_val)

        # Mock the model attribute
        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=[
            np.random.rand(n_train, 1),  # train prediction
            np.random.rand(n_val, 1),    # val prediction
        ])
        callback.model = mock_model

        callback.on_epoch_end(epoch=0, logs={})

        assert len(callback.history["train_mae"]) == 1
        assert len(callback.history["val_mae"]) == 1
        assert len(callback.history["val_mape"]) == 1
        assert len(callback.history["val_r2"]) == 1
        assert len(callback.history["val_sharpe"]) == 1

    def test_logs_updated_with_val_mae(self):
        """on_epoch_end should inject val_mae into Keras logs dict."""
        from training.runner import MetricsTrackingCallback

        n = 10
        callback = MetricsTrackingCallback(
            np.random.rand(n, 5, 3), np.random.rand(n),
            np.random.rand(n, 5, 3), np.random.rand(n),
        )

        mock_model = MagicMock()
        mock_model.predict = MagicMock(side_effect=[
            np.random.rand(n, 1),
            np.random.rand(n, 1),
        ])
        callback.model = mock_model

        logs = {}
        callback.on_epoch_end(epoch=0, logs=logs)
        assert "val_mae" in logs
        assert isinstance(logs["val_mae"], float)
