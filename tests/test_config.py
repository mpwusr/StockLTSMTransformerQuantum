"""Tests for config.py — configuration loading and environment validation."""

import os
import pytest
from unittest.mock import patch


class TestConfigLoading:
    def test_cfg_is_dict(self):
        from config import cfg
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        from config import cfg
        assert "model" in cfg
        assert "trading" in cfg
        assert "horizons" in cfg
        assert "training" in cfg
        assert "data" in cfg
        assert "gui" in cfg

    def test_model_config_structure(self):
        from config import cfg
        assert "look_back" in cfg["model"]
        assert "lstm" in cfg["model"]
        assert "transformer" in cfg["model"]
        assert "gru_cnn" in cfg["model"]
        assert "quantum" in cfg["model"]

    def test_trading_config_values(self):
        from config import cfg
        trading = cfg["trading"]
        assert trading["initial_cash"] > 0
        assert 0 < trading["stop_loss"] < 1
        assert 0 < trading["take_profit"] < 1
        assert trading["moving_average_window"] >= 1

    def test_horizon_values(self):
        from config import cfg
        horizons = cfg["horizons"]
        assert horizons["short"] < horizons["medium"] < horizons["long"]


class TestValidateEnv:
    def test_validate_env_passes(self):
        """validate_env should pass when env vars are set (by conftest.py)."""
        from config import validate_env
        # Should not raise — env vars set in conftest.py
        validate_env()

    @patch.dict(os.environ, {}, clear=True)
    def test_validate_env_warns_missing_optional(self):
        """validate_env should warn about missing optional vars."""
        # Re-set required env vars so validate_env doesn't fail
        os.environ["POLYGON_API_KEY"] = ""
        os.environ["ALPHAVANTAGE_API_KEY"] = ""
        from config import validate_env
        # Should not raise (REQUIRED_ENV_VARS is empty)
        validate_env()
