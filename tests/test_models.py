"""Tests for models.py — trading strategy, moving average, normalize_input, smooth_predictions."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from models import (
    moving_average,
    smooth_predictions,
    normalize_input,
    trading_strategy,
    add_noise_debug_overlay,
    compute_rsi_from_series,
)


class TestMovingAverage:
    def test_basic_window(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average(data, window=3)
        expected = np.convolve(data, np.ones(3) / 3, mode="valid")
        np.testing.assert_array_almost_equal(result, expected)

    def test_window_1_returns_original(self):
        data = [10.0, 20.0, 30.0]
        result = moving_average(data, window=1)
        np.testing.assert_array_almost_equal(result, data)

    def test_output_length(self):
        data = [1.0] * 10
        result = moving_average(data, window=3)
        assert len(result) == 10 - 3 + 1


class TestSmoothPredictions:
    def test_returns_same_length(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_predictions(preds, window=3)
        assert len(result) == len(preds)

    def test_smoothing_reduces_variance(self):
        np.random.seed(42)
        noisy = np.random.randn(50) * 10
        smoothed = smooth_predictions(noisy, window=5)
        # Interior points should have lower variance than original
        assert np.std(smoothed[5:-5]) < np.std(noisy[5:-5])


class TestNormalizeInput:
    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = normalize_input(x)
        assert len(result) == 4  # n_qubits = 4
        assert result.min() >= -np.pi - 0.01
        assert result.max() <= np.pi + 0.01

    def test_2d_input(self):
        x = np.random.rand(10, 7)
        result = normalize_input(x)
        assert len(result) == 4
        assert result.min() >= -np.pi - 0.01
        assert result.max() <= np.pi + 0.01

    def test_constant_input(self):
        """Constant input should not crash (division by near-zero handled)."""
        x = np.ones(7) * 5.0
        result = normalize_input(x)
        assert len(result) == 4
        assert np.all(np.isfinite(result))


class TestTradingStrategy:
    def test_returns_correct_tuple_format(self, sample_predictions, last_date, future_dates):
        positions, cash_str, go_no_go, stats = trading_strategy(
            sample_predictions, last_date, future_dates
        )
        assert isinstance(positions, list)
        assert isinstance(cash_str, str)
        assert cash_str.startswith("$")
        assert go_no_go in ("Go", "No Go")
        assert "Return %" in stats
        assert "Trades" in stats
        assert "Win Rate" in stats

    def test_no_trades_on_flat_predictions(self, flat_predictions, last_date, future_dates):
        positions, cash_str, go_no_go, stats = trading_strategy(
            flat_predictions, last_date, future_dates
        )
        # With flat prices, MA crossover should not trigger buys
        assert stats["Trades"] == 0

    def test_stop_loss_triggered(self, last_date, future_dates):
        """Prices that rise then crash should trigger stop-loss."""
        predictions = [100.0, 110.0, 111.0, 112.0, 80.0, 79.0, 78.0, 77.0, 76.0, 75.0]
        positions, cash_str, go_no_go, stats = trading_strategy(
            predictions, last_date, future_dates
        )
        # Should have at least one sell due to stop-loss
        sell_positions = [p for p in positions if "Sell" in p]
        if stats["Trades"] > 0:
            assert len(sell_positions) > 0

    def test_take_profit_triggered(self, last_date, future_dates):
        """Steadily rising prices should trigger take-profit."""
        predictions = [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0]
        positions, cash_str, go_no_go, stats = trading_strategy(
            predictions, last_date, future_dates
        )
        if stats["Trades"] > 0:
            assert stats["Wins"] > 0 or stats["Losses"] > 0

    def test_verbose_mode(self, sample_predictions, last_date, future_dates):
        """Verbose mode should not crash."""
        positions, cash_str, go_no_go, stats = trading_strategy(
            sample_predictions, last_date, future_dates, verbose=True
        )
        assert isinstance(positions, list)

    def test_rsi_filter(self, sample_predictions, last_date, future_dates):
        """RSI series should filter trades."""
        rsi_series = np.array([80.0] * 10)  # All above threshold[1]=70
        positions, cash_str, go_no_go, stats = trading_strategy(
            sample_predictions, last_date, future_dates,
            rsi_series=rsi_series, rsi_threshold=(30, 70)
        )
        # High RSI should prevent buying
        assert stats["Trades"] == 0

    def test_win_rate_calculation(self, last_date, future_dates):
        """Win rate should be wins/trades."""
        predictions = [100.0, 105.0, 112.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0]
        _, _, _, stats = trading_strategy(predictions, last_date, future_dates)
        if stats["Trades"] > 0:
            expected_rate = stats["Wins"] / stats["Trades"]
            assert stats["Win Rate"] == f"{expected_rate:.1%}"


class TestAddNoiseDebugOverlay:
    def test_creates_twin_axis(self):
        fig_mock = MagicMock()
        ax = MagicMock()
        ax2 = MagicMock()
        ax.twinx.return_value = ax2
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        dates = [datetime(2026, 1, i) for i in range(1, 6)]
        add_noise_debug_overlay(ax, predictions, dates)
        ax.twinx.assert_called_once()
        ax2.plot.assert_called_once()


class TestComputeRsiFromSeries:
    def test_basic_rsi(self):
        """compute_rsi_from_series should call RSIIndicator with correct params."""
        import pandas as pd
        import numpy as np
        from unittest.mock import MagicMock, patch

        # Mock RSIIndicator since ta library is mocked
        mock_rsi = MagicMock()
        mock_rsi.rsi.return_value.dropna.return_value.values = np.array([55.0, 60.0, 45.0])

        with patch("models.RSIIndicator", return_value=mock_rsi) as mock_cls:
            close = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                               46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03,
                               46.41, 46.22, 45.64])
            result = compute_rsi_from_series(close, period=14)
            mock_cls.assert_called_once()
            assert len(result) == 3
