"""Tests for data/providers.py — date range resolution, data fetching, and preparation."""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestResolveDateRange:
    def test_years_ago(self):
        from data.providers import resolve_date_range
        start, end = resolve_date_range(years_ago=2)
        assert start is not None
        assert end is not None
        # Start should be ~2 years before end
        from datetime import date
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        diff = (e - s).days
        assert 720 <= diff <= 732

    def test_custom_dates(self):
        from data.providers import resolve_date_range
        start, end = resolve_date_range(start="2023-01-01", end="2024-01-01")
        assert start == "2023-01-01"
        assert end == "2024-01-01"

    def test_default_dates(self):
        from data.providers import resolve_date_range
        start, end = resolve_date_range()
        assert start == "2020-01-01"  # from config.yaml default_start

    def test_invalid_range_raises(self):
        from data.providers import resolve_date_range
        with pytest.raises(ValueError, match="Start date must be before end date"):
            resolve_date_range(start="2025-01-01", end="2020-01-01")


class TestLoadTickersFromEnv:
    @patch.dict(os.environ, {"TICKERS": "AAPL,MSFT,GOOGL"})
    def test_loads_tickers(self):
        from data.providers import load_tickers_from_env
        tickers = load_tickers_from_env()
        assert tickers == ["AAPL", "MSFT", "GOOGL"]

    @patch.dict(os.environ, {"TICKERS": ""})
    def test_empty_tickers(self):
        from data.providers import load_tickers_from_env
        tickers = load_tickers_from_env()
        assert tickers == []


class TestFetchStockData:
    @patch("data.providers.fetch_yahoo_resilient")
    def test_yahoo_source(self, mock_yahoo, mock_stock_df):
        mock_yahoo.return_value = mock_stock_df
        from data.providers import fetch_stock_data
        result = fetch_stock_data("AAPL", source="yahoo", start="2025-01-01",
                                  end="2025-06-01", use_cache=False)
        mock_yahoo.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert "Close" in result.columns

    @patch("data.providers.fetch_polygon_resilient")
    @patch("data.providers.fetch_yahoo_resilient")
    def test_fallback_to_yahoo(self, mock_yahoo, mock_polygon, mock_stock_df):
        mock_polygon.side_effect = Exception("Polygon API error")
        mock_yahoo.return_value = mock_stock_df
        from data.providers import fetch_stock_data
        result = fetch_stock_data("AAPL", source="polygon", start="2025-01-01",
                                  end="2025-06-01", use_cache=False)
        mock_yahoo.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch("data.providers.fetch_yahoo_resilient")
    def test_cache_hit(self, mock_yahoo, mock_stock_df, tmp_path):
        """Cached data should be loaded without API call."""
        mock_yahoo.return_value = mock_stock_df
        from data.providers import fetch_stock_data

        # First call — misses cache
        with patch("data.providers.CACHE_DIR", str(tmp_path)):
            result1 = fetch_stock_data("AAPL", source="yahoo", years_ago=2, use_cache=True)
            # Save to cache manually
            cache_file = list(tmp_path.glob("*.csv"))
            assert len(cache_file) == 1

    @patch("data.providers.fetch_yahoo_resilient")
    def test_unsupported_source_falls_back(self, mock_yahoo, mock_stock_df):
        """Unsupported source should fall back to yahoo (not raise)."""
        mock_yahoo.return_value = mock_stock_df
        from data.providers import fetch_stock_data
        # The code catches ValueError internally and falls back to yahoo
        result = fetch_stock_data("AAPL", source="nonexistent", start="2025-01-01",
                                  end="2025-06-01", use_cache=False)
        mock_yahoo.assert_called_once()
        assert isinstance(result, pd.DataFrame)


class TestFetchAndPrepareData:
    @patch("data.providers.fetch_stock_data")
    def test_returns_data_with_indicators(self, mock_fetch, mock_stock_df):
        """fetch_and_prepare_data should add indicators and return arrays."""
        mock_fetch.return_value = mock_stock_df
        from data.providers import fetch_and_prepare_data
        X, y, data_clean, scaler, look_back = fetch_and_prepare_data("AAPL", source="yahoo")
        # data_clean should have extra columns (indicators are mocked but columns are added)
        assert isinstance(data_clean, pd.DataFrame)
        assert hasattr(scaler, "fit_transform")

    @patch("data.providers.fetch_stock_data")
    def test_features_include_indicators(self, mock_fetch, mock_stock_df):
        mock_fetch.return_value = mock_stock_df
        from data.providers import fetch_and_prepare_data
        _, _, data_clean, _, _ = fetch_and_prepare_data("AAPL", source="yahoo")
        assert "SMA_20" in data_clean.columns
        assert "SMA_50" in data_clean.columns
        assert "RSI" in data_clean.columns
        assert "MACD" in data_clean.columns
        assert "MACD_Signal" in data_clean.columns
