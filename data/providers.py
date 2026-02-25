"""Data providers for stock market data with caching and fallback."""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import cfg
from utils.logging_config import setup_logging
from utils.resilient_clients import (
    fetch_yahoo_resilient,
    fetch_polygon_resilient,
    fetch_alphavantage_resilient,
)

logger = setup_logging("data_providers", "data_providers.log")

CACHE_DIR = cfg["data"]["cache_dir"]
os.makedirs(CACHE_DIR, exist_ok=True)


def load_tickers_from_env():
    """Load ticker symbols from the TICKERS environment variable."""
    from dotenv import load_dotenv
    load_dotenv()
    tickers_str = os.getenv("TICKERS", "")
    return [t.strip() for t in tickers_str.split(",") if t.strip()]


def resolve_date_range(start=None, end=None, years_ago=None):
    """Resolve date range from various input formats."""
    today = datetime.today().date()
    if years_ago:
        start_date = today - timedelta(days=365 * years_ago)
        end_date = today
    else:
        default_start = cfg["data"]["default_start"]
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else datetime.strptime(default_start, "%Y-%m-%d").date()
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else today
    if start_date > end_date:
        raise ValueError("Start date must be before end date.")
    return start_date.isoformat(), end_date.isoformat()


def fetch_stock_data(ticker, source="polygon", start=None, end=None, years_ago=None, use_cache=True):
    """Fetch stock data with caching and provider fallback.

    Tries the specified source first, falls back to Yahoo if it fails.
    Caches results as CSV for subsequent calls.
    """
    custom_range = (start is not None and end is not None and years_ago is None)
    start, end = resolve_date_range(start, end, years_ago)
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{source}_{start}_{end}.csv")

    if custom_range and os.path.exists(cache_file):
        logger.info(f"Custom range selected: removing old cache {cache_file}")
        os.remove(cache_file)

    if use_cache and os.path.exists(cache_file):
        logger.info(f"Loading {ticker} from cache: {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    try:
        if source == "polygon":
            df = fetch_polygon_resilient(ticker, start, end)
        elif source == "yahoo":
            df = fetch_yahoo_resilient(ticker, start, end)
        elif source == "alphavantage":
            df = fetch_alphavantage_resilient(ticker, start=start, end=end)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    except Exception as e:
        logger.warning(f"Primary source '{source}' failed for {ticker}: {e}")
        if source != "yahoo":
            logger.info("Falling back to Yahoo Finance")
            df = fetch_yahoo_resilient(ticker, start, end)
        else:
            raise

    if use_cache:
        df.to_csv(cache_file)
        logger.info(f"Saved {ticker} data to cache: {cache_file}")

    return df


def fetch_and_prepare_data(ticker, source="polygon", start=None, end=None, years_ago=None):
    """Fetch stock data and prepare it for model training.

    Adds technical indicators (SMA, RSI, MACD), scales features,
    and creates sliding window sequences.

    Returns:
        tuple: (X, y, data_clean, scaler, look_back)
    """
    from ta.trend import SMAIndicator, MACD
    from ta.momentum import RSIIndicator
    from sklearn.preprocessing import MinMaxScaler

    look_back = cfg["model"]["look_back"]

    data = fetch_stock_data(ticker, source=source, start=start, end=end, years_ago=years_ago)
    close_prices = data['Close'].squeeze()

    # Add technical indicators
    data['SMA_20'] = SMAIndicator(close_prices, window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
    data['RSI'] = RSIIndicator(close_prices, window=14).rsi()
    macd = MACD(close_prices)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    data_clean = data.dropna()
    logger.info(f"Prepared {len(data_clean)} rows for {ticker} with {len(data_clean.columns)} features")

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_clean)

    # Create sequences
    X, y = [], []
    close_index = data_clean.columns.get_loc('Close')
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, close_index])

    return np.array(X), np.array(y), data_clean, scaler, look_back
