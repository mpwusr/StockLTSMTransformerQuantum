"""Resilient API clients with tenacity retry logic."""

import logging
import time
import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from utils.logging_config import setup_logging

logger = setup_logging("resilient_clients", "resilient_clients.log")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def fetch_yahoo_resilient(ticker, start="2020-01-01", end=None, auto_adjust=True):
    """Fetch stock data from Yahoo Finance with retry logic."""
    logger.info(f"Fetching {ticker} from Yahoo Finance")
    df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    if df.empty:
        raise ValueError(f"No data from Yahoo for {ticker}")
    logger.info(f"Successfully fetched {len(df)} rows for {ticker} from Yahoo")
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def fetch_polygon_resilient(ticker, start, end, api_key=None):
    """Fetch stock data from Polygon.io with retry logic and pagination."""
    if api_key is None:
        api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY not found in environment")

    from config import cfg
    base_url = cfg["api"]["polygon_base_url"]
    limit = cfg["api"]["polygon_limit"]

    all_results = []
    current_start = start

    while True:
        url = f"{base_url}/{ticker}/range/1/day/{current_start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "limit": limit, "apiKey": api_key}

        logger.info(f"Fetching {ticker} from Polygon ({current_start} -> {end})")
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        batch = data.get("results", [])

        if not batch:
            break

        all_results.extend(batch)

        if len(batch) < limit:
            break

        last_ts = batch[-1]["t"]
        last_date = datetime.utcfromtimestamp(last_ts / 1000).date()
        current_start = (last_date + timedelta(days=1)).isoformat()

    if not all_results:
        raise ValueError(f"No data returned for {ticker} from Polygon")

    df = pd.DataFrame(all_results)[['o', 'h', 'l', 'c', 'v', 't']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 't']
    df['t'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('t', inplace=True)
    logger.info(f"Successfully fetched {len(df)} rows for {ticker} from Polygon")
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def fetch_alphavantage_resilient(ticker, start="2020-01-01", end=None, api_key=None):
    """Fetch stock data from Alpha Vantage with retry logic."""
    if api_key is None:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment")

    from config import cfg
    base_url = cfg["api"]["alpha_base_url"]

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": api_key,
    }
    logger.info(f"Fetching {ticker} from Alpha Vantage")
    resp = requests.get(base_url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "Error Message" in data:
        raise RuntimeError(f"AlphaVantage error: {data['Error Message']}")
    if "Time Series (Daily)" not in data:
        raise RuntimeError(f"Unexpected response format from AlphaVantage")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index").sort_index()
    df.index = pd.to_datetime(df.index)
    df.rename(columns={
        "1. open": "Open", "2. high": "High",
        "3. low": "Low", "4. close": "Close", "6. volume": "Volume",
    }, inplace=True)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today()
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

    logger.info(f"Successfully fetched {len(df)} rows for {ticker} from Alpha Vantage")
    return df[["Open", "High", "Low", "Close", "Volume"]]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((ConnectionError, requests.exceptions.Timeout)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def download_gdrive_resilient(gdrive_url, output_path="temp_tickers.csv"):
    """Download a file from Google Drive with retry logic."""
    if "spreadsheets/d/" in gdrive_url:
        file_id = gdrive_url.split("spreadsheets/d/")[-1].split("/")[0]
        download_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
    elif "id=" in gdrive_url:
        file_id = gdrive_url.split("id=")[-1].split("&")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    elif "file/d/" in gdrive_url:
        file_id = gdrive_url.split("file/d/")[-1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        raise ValueError("Invalid Google Drive or Google Sheets URL format.")

    logger.info(f"Downloading from Google Drive: {download_url}")
    response = requests.get(download_url, timeout=30)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info(f"File downloaded to {output_path}")
    return output_path
