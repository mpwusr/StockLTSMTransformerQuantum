"""Smoke tests for visualization/plots.py — verify plots don't crash."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


class TestPlotForecast:
    def test_plot_forecast_runs(self, mock_stock_df):
        """plot_forecast should not crash with valid inputs."""
        from visualization.plots import plot_forecast

        figure = MagicMock()
        ax = MagicMock()
        figure.add_subplot.return_value = ax
        canvas = MagicMock()

        data_key = ("AAPL", "yahoo", "2025-01-01", "2025-12-31", None)
        ticker_data = {data_key: {"data": mock_stock_df}}
        last_date = mock_stock_df.index[-1]

        pred_dict = {
            "Short": np.random.rand(30) * 100 + 100,
            "Medium": np.random.rand(90) * 100 + 100,
            "Long": np.random.rand(365) * 100 + 100,
        }

        plot_forecast(
            figure, canvas, ticker_data, data_key,
            "AAPL", last_date, pred_dict,
        )
        figure.clear.assert_called_once()
        figure.add_subplot.assert_called_once()
        canvas.draw.assert_called_once()

    def test_plot_forecast_with_comparison(self, mock_stock_df):
        """plot_forecast with comparison ticker should not crash."""
        from visualization.plots import plot_forecast

        figure = MagicMock()
        ax = MagicMock()
        figure.add_subplot.return_value = ax
        canvas = MagicMock()

        data_key1 = ("AAPL", "yahoo", "2025-01-01", "2025-12-31", None)
        data_key2 = ("MSFT", "yahoo", "2025-01-01", "2025-12-31", None)
        ticker_data = {
            data_key1: {"data": mock_stock_df},
            data_key2: {"data": mock_stock_df},
        }

        pred_dict = {
            "Short": np.random.rand(30) * 100 + 100,
            "Medium": np.random.rand(90) * 100 + 100,
            "Long": np.random.rand(365) * 100 + 100,
        }

        plot_forecast(
            figure, canvas, ticker_data, data_key1,
            "AAPL", mock_stock_df.index[-1], pred_dict,
            compare_forecast=pred_dict,
            compare_ticker="MSFT",
            compare_data_key=data_key2,
            compare_last_date=mock_stock_df.index[-1],
        )
        canvas.draw.assert_called_once()


class TestPlotTrainingCurves:
    def test_plot_training_curves_runs(self, sample_history):
        """plot_training_curves should not crash with valid history."""
        from visualization.plots import plot_training_curves

        figure = MagicMock()
        gs = MagicMock()
        figure.add_gridspec.return_value = gs
        ax_mock = MagicMock()
        figure.add_subplot.return_value = ax_mock
        canvas = MagicMock()

        result = plot_training_curves(figure, canvas, sample_history, label_prefix="AAPL")
        assert result is not None
        assert result["initialized"] is True
        canvas.draw.assert_called_once()

    def test_empty_history_returns_none(self):
        """Empty history should return early without plotting."""
        from visualization.plots import plot_training_curves

        figure = MagicMock()
        canvas = MagicMock()

        result = plot_training_curves(figure, canvas, {}, label_prefix="AAPL")
        assert result is None
        canvas.draw.assert_not_called()
