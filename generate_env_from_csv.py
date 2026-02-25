"""Generate .env file with validated ticker symbols from a Google Sheets CSV."""

import csv
import os
import yfinance as yf
from dotenv import set_key, load_dotenv

from utils.logging_config import setup_logging
from utils.resilient_clients import download_gdrive_resilient

logger = setup_logging("generate_env", "generate_env.log")


def extract_first_column_tickers(file_path):
    """Extract ticker symbols from the first column of a CSV file."""
    symbols = set()
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].strip().isalpha():
                symbols.add(row[0].strip().upper())
    logger.info(f"Extracted {len(symbols)} raw tickers from {file_path}")
    return sorted(symbols)


def validate_tickers_yfinance(tickers):
    """Validate tickers against Yahoo Finance."""
    valid = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if "shortName" in info and info["shortName"]:
                valid.append(ticker)
                logger.debug(f"Validated: {ticker}")
        except Exception:
            logger.debug(f"Invalid ticker: {ticker}")
    logger.info(f"Validated {len(valid)}/{len(tickers)} tickers")
    return valid


def write_tickers_to_env(tickers, env_path=".env"):
    """Write validated tickers to .env file."""
    if not os.path.exists(env_path):
        open(env_path, "a").close()
    load_dotenv(env_path)
    tickers_str = ",".join(sorted(set(tickers)))
    set_key(env_path, "TICKERS", tickers_str)
    logger.info(f"TICKERS updated in {env_path}: {tickers_str}")


if __name__ == "__main__":
    url = input("Paste the Google Drive CSV shareable link: ").strip()
    try:
        local_csv = download_gdrive_resilient(url)
        raw_tickers = extract_first_column_tickers(local_csv)
        logger.info(f"Found {len(raw_tickers)} raw tickers. Validating...")
        clean_tickers = validate_tickers_yfinance(raw_tickers)
        logger.info(f"{len(clean_tickers)} tickers validated.")
        write_tickers_to_env(clean_tickers)
        os.remove(local_csv)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
