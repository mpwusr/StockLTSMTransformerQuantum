# StockLTSMTransformerQuantum

**Author:** Michael P. Williams
**License:** MIT

A PyQt5 desktop application for forecasting stock prices using classical and quantum neural networks. Supports real-time training diagnostics, multi-horizon visualization, dual-stock comparison, and trading strategy simulation.

---

## Architecture

```
StockLTSMTransformerQuantum/
├── main.py                    # PyQt5 GUI entry point (StockTradingGUI)
├── models.py                  # LSTM, Transformer, GRU-CNN, QML model builders
│                              #   trading_strategy, moving_average, normalize_input
├── config.py                  # YAML + .env loader, validate_env() at import time
├── config.yaml                # All model, trading, horizon, and GUI settings
│
├── data/
│   └── providers.py           # Multi-source stock data fetching
│                              #   fetch_stock_data (Yahoo, Polygon, AlphaVantage)
│                              #   resolve_date_range, add_indicators (RSI, MACD, BB)
│
├── training/
│   └── runner.py              # ModelTrainerThread (QThread background training)
│                              #   predict_future (autoregressive multi-step)
│                              #   MetricsTrackingCallback (MAE, MAPE, R2, Sharpe)
│
├── visualization/
│   └── plots.py               # plot_forecast, plot_training_curves
│
├── utils/
│   ├── logging_config.py      # Centralized rotating-file logger
│   └── resilient_clients.py   # Retry-wrapped API calls with tenacity
│
├── tests/                     # pytest suite — 46 tests, no GPU/API required
│   ├── conftest.py            # Env setup + sys.modules mocks
│   ├── test_models.py         # trading_strategy, moving_average, RSI, etc.
│   ├── test_providers.py      # Data fetching, caching, fallback
│   ├── test_runner.py         # predict_future, MetricsTrackingCallback
│   ├── test_config.py         # YAML loading, env validation
│   └── test_visualization.py  # Smoke tests for plot functions
│
├── Makefile                   # make test, make test-cov, make clean
├── pytest.ini                 # pytest configuration
├── requirements.txt           # Production dependencies
└── requirements-test.txt      # Test-only dependencies
```

### Data Flow

```
User selects Ticker + Model + Horizon in GUI
        │
        ▼
  data/providers.py       fetch_stock_data() with automatic fallback
        │                 add RSI, MACD, Bollinger Band indicators
        │                 MinMaxScaler → sliding-window sequences
        ▼
  models.py               build_lstm_model() / build_transformer_model() /
        │                 build_gru_cnn_model() / optimize_quantum_weights()
        ▼
  training/runner.py      ModelTrainerThread trains on background QThread
        │                 MetricsTrackingCallback tracks MAE, MAPE, R2, Sharpe
        ▼
  models.py               predict_future() — autoregressive multi-step forecast
        │                 trading_strategy() — MA crossover + RSI + stop-loss
        ▼
  visualization/plots.py  plot_forecast() + plot_training_curves() → GUI canvas
```

### Models

| Model | Architecture | Config Key |
|-------|-------------|------------|
| **LSTM** | 2-layer LSTM (50 units each), dropout 0.2 | `model.lstm` |
| **Transformer** | Multi-head attention (4 heads, key_dim 64), dense layers | `model.transformer` |
| **GRU-CNN** | Conv1D (32 filters) + 2-layer GRU (64, 32 units) | `model.gru_cnn` |
| **QML** | 4-qubit PennyLane variational circuit, 4 layers, 300 iterations | `model.quantum` |

### Forecast Horizons

| Horizon | Days |
|---------|------|
| Short | 30 |
| Medium | 90 |
| Long | 365 |

---

## Configuration

All tunable settings live in **`config.yaml`**:

```yaml
model:
  look_back: 60                    # Sliding window size for input sequences
  lstm:
    units: [50, 50]                # Units per LSTM layer
    dropout: 0.2
  transformer:
    num_heads: 4                   # Multi-head attention heads
    key_dim: 64
    dense_units: [128, 100, 50]
    dropout: 0.1
  gru_cnn:
    conv_filters: 32
    kernel_size: 3
    gru_units: [64, 32]
    dropout: 0.2
  quantum:
    n_qubits: 4                    # Qubits in variational circuit
    layers: 4                      # Circuit depth
    iterations: 300                # Optimization steps
    step_size: 0.1

trading:
  initial_cash: 10000              # Starting capital for simulation
  stop_loss: 0.05                  # 5% stop-loss trigger
  take_profit: 0.1                 # 10% take-profit trigger
  moving_average_window: 3         # MA crossover window
  rsi_threshold: [30, 70]          # RSI buy/sell thresholds

horizons:
  short: 30
  medium: 90
  long: 365

training:
  epochs: 10
  batch_size: 32
  validation_split: 0.2
  patience: 3                      # Early stopping patience

data:
  default_start: "2020-01-01"      # Default historical start date
  providers: [polygon, yahoo, alphavantage]
  cache_dir: "data_cache"          # Local cache for fetched data
```

---

## Environment Variables

Create a **`.env`** file in the project root:

```bash
POLYGON_API_KEY=your_polygon_key        # Optional — falls back to Yahoo
ALPHAVANTAGE_API_KEY=your_alpha_key      # Optional — falls back to Yahoo
TICKERS=AAPL,TSLA,GOOGL                 # Comma-separated ticker list
```

**No API keys are strictly required.** If Polygon or AlphaVantage keys are missing or the source fails, the app automatically falls back to Yahoo Finance.

---

## Installation

**Requires Python 3.10+**

```bash
git clone https://github.com/mpwusr/StockLTSMTransformerQuantum.git
cd StockLTSMTransformerQuantum

python3 -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

> **Note:** If the `ta` package fails to build on macOS, install TA-Lib via Homebrew first:
> ```bash
> brew install ta-lib
> ```

---

## How to Run

1. Activate your virtual environment and configure `.env`
2. Launch the GUI:

```bash
python3 main.py
```

3. In the GUI:
   - Select a stock ticker and model (LSTM, Transformer, GRUCNN, QML)
   - Optionally enable dual-stock comparison with a second ticker
   - Choose a data source (Yahoo recommended if no API keys configured)
   - Click **Run** to start training and forecasting

### Output

| Output | Location |
|--------|----------|
| Forecast CSVs | `forecasts/{ticker}/` |
| Training metrics and logs | `training_logs/{ticker}/` |
| Debug trading logs | `logs/trade_debug.txt` |

---

## Testing

The test suite runs entirely offline with **no GPU, no API tokens, and no network access**. All heavy dependencies (PennyLane, TensorFlow/Keras, PyQt5, matplotlib, ta, sklearn) are stubbed via `sys.modules` mocks in `conftest.py`.

### Run with Make

```bash
make test            # Run all 46 tests with verbose output
make test-cov        # Run with coverage report (terminal + HTML)
make test-quick      # Stop on first failure
make test-verbose    # Full tracebacks with stdout capture disabled
make clean           # Remove __pycache__, .pytest_cache, coverage artifacts
make install-test    # Install test dependencies (pytest, pytest-cov, pytest-mock)
```

### Run Directly

```bash
pip install -r requirements-test.txt
python3 -m pytest tests/ -v
```

### Test Suite Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_models.py` | 17 | trading_strategy, moving_average, smooth_predictions, normalize_input, RSI, noise overlay |
| `test_providers.py` | 10 | resolve_date_range, load_tickers, fetch_stock_data (cache, fallback), fetch_and_prepare_data |
| `test_runner.py` | 6 | predict_future (autoregressive loop, inverse_transform), MetricsTrackingCallback |
| `test_config.py` | 7 | YAML loading, config structure, validate_env |
| `test_visualization.py` | 4 | plot_forecast, plot_training_curves (smoke tests) |
| **Total** | **46** | |

---

## Version Tracking

```bash
python3 versions.py
```

Writes Python version and all installed package versions to both the terminal and `versions.txt`.

---

## License

MIT License

---

**Contact:** [mw00066@vt.edu](mailto:mw00066@vt.edu)
