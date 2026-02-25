"""Centralized configuration loader for StockLTSMTransformerQuantum."""

import os
import yaml
from dotenv import load_dotenv

load_dotenv()

_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Environment variable validation
REQUIRED_ENV_VARS = []  # None strictly required for Yahoo-only mode
OPTIONAL_ENV_VARS = ["POLYGON_API_KEY", "ALPHAVANTAGE_API_KEY", "TICKERS"]

def validate_env():
    """Validate required environment variables exist."""
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Warn about optional vars
    for key in OPTIONAL_ENV_VARS:
        if not os.getenv(key):
            import logging
            logging.getLogger("config").warning(f"Optional env var {key} not set")

validate_env()

if __name__ == "__main__":
    import json
    print("Configuration loaded successfully:")
    print(json.dumps(cfg, indent=2, default=str))
    print(f"\nPOLYGON_API_KEY: {'set' if os.getenv('POLYGON_API_KEY') else 'not set'}")
    print(f"ALPHAVANTAGE_API_KEY: {'set' if os.getenv('ALPHAVANTAGE_API_KEY') else 'not set'}")
    print(f"TICKERS: {os.getenv('TICKERS', 'not set')}")
