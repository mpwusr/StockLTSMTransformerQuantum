"""StockLTSMTransformerQuantum — PyQt5 GUI for stock price prediction.

Supports LSTM, Transformer, GRU-CNN, and Quantum ML models with
dual-stock comparison, training diagnostics, and trading strategy evaluation.
"""

import os
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTextEdit, QProgressBar,
    QTabWidget, QCheckBox, QLineEdit, QSizePolicy, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from config import cfg
from utils.logging_config import setup_logging
from data.providers import fetch_and_prepare_data, load_tickers_from_env
from training.runner import ModelTrainerThread, predict_future
from visualization.plots import plot_forecast, plot_training_curves
from models import (
    build_lstm_model, build_transformer_model, build_gru_cnn_model,
    optimize_quantum_weights, quantum_predict_future, trading_strategy,
)

logger = setup_logging("main", "main.log")


class StockTradingGUI(QMainWindow):
    """Main application window for stock trading analysis."""

    def __init__(self):
        super().__init__()
        self._init_state()
        self._init_ui()
        self._connect_signals()

    def _init_state(self):
        """Initialize application state variables."""
        self.training_axes_state = None
        self.compare_forecast = None
        self.compare_history = None
        self.compare_ticker = None
        self.compare_last_date = None
        self.compare_thread = None
        self.last_forecast = None
        self.last_history = None
        self.last_ticker = None
        self.last_date = None
        self.forecast_folder_path = None
        self.tickers = load_tickers_from_env()
        self.ticker_data = {}
        self.current_data_key = None
        self.compare_data_key = None
        self.output_texts = {}
        self.training_thread = None
        self.current_model = None

    def _init_ui(self):
        """Build the user interface."""
        gui_cfg = cfg["gui"]
        self.setWindowTitle(gui_cfg["window_title"])
        geom = gui_cfg["window_geometry"]
        self.setGeometry(*geom)

        main = QWidget(self)
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        # Ticker selection
        self.ticker_combo = QComboBox()
        self.ticker_combo.addItems(self.tickers)
        layout.addWidget(QLabel("Select Ticker:"))
        layout.addWidget(self.ticker_combo)

        # Comparison ticker
        self.compare_combo = QComboBox()
        self.compare_combo.addItems(self.tickers)
        layout.addWidget(QLabel("Compare With (Optional):"))
        layout.addWidget(self.compare_combo)

        self.compare_checkbox = QCheckBox("Enable Stock Comparison")
        layout.addWidget(self.compare_checkbox)

        # Data source selection
        self.source_combo = QComboBox()
        self.source_combo.addItems(cfg["data"]["providers"])
        layout.addWidget(QLabel("Select Data Source:"))
        layout.addWidget(self.source_combo)

        # Date range
        layout.addWidget(QLabel("Select Date Range:"))
        self.date_range_combo = QComboBox()
        self.date_range_combo.addItems([
            "Default (2020–Today)", "Last 2 years", "Last 3 years",
            "Last 5 years", "Custom Range",
        ])
        layout.addWidget(self.date_range_combo)

        self.start_input = QLineEdit()
        self.end_input = QLineEdit()
        self.start_input.setPlaceholderText("Start Date (YYYY-MM-DD)")
        self.end_input.setPlaceholderText("End Date (YYYY-MM-DD)")
        self.start_input.hide()
        self.end_input.hide()
        layout.addWidget(self.start_input)
        layout.addWidget(self.end_input)

        # Debug checkbox
        self.debug_checkbox = QCheckBox("Debug Trading Logs")
        layout.addWidget(self.debug_checkbox)

        # Model buttons
        button_layout = QHBoxLayout()
        self.lstm_btn = QPushButton("Run LSTM")
        self.trans_btn = QPushButton("Run Transformer")
        self.gru_cnn_btn = QPushButton("Run GRUCNN")
        self.q_btn = QPushButton("Run QML")
        for btn in [self.lstm_btn, self.trans_btn, self.gru_cnn_btn, self.q_btn]:
            button_layout.addWidget(btn)
        layout.addLayout(button_layout)

        # Save / View buttons
        button_row = QHBoxLayout()
        self.save_button = QPushButton("Save Forecast")
        self.save_button.setEnabled(False)
        self.view_folder_button = QPushButton("View Folder")
        self.view_folder_button.setEnabled(False)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.view_folder_button)
        layout.addLayout(button_row)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        # Tabs
        self.tabs = QTabWidget()
        self.forecast_tab = QWidget()
        self.training_tab = QWidget()
        self.tabs.addTab(self.forecast_tab, "Forecast")
        self.tabs.addTab(self.training_tab, "Training Diagnostics")
        layout.addWidget(self.tabs)

        # Forecast tab
        self.forecast_layout = QVBoxLayout(self.forecast_tab)
        output_layout = QHBoxLayout()
        for label in ["Short", "Medium", "Long"]:
            box = QTextEdit()
            box.setReadOnly(True)
            self.output_texts[label] = box
            output_layout.addWidget(box)
        self.forecast_layout.addLayout(output_layout)

        self.forecast_figure = plt.Figure()
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        self.forecast_layout.addWidget(self.forecast_canvas)

        # Training diagnostics tab
        self.training_layout = QVBoxLayout(self.training_tab)
        self.training_layout.setContentsMargins(0, 0, 0, 0)
        self.training_layout.setSpacing(0)
        self.training_layout.setAlignment(Qt.AlignTop)

        self.training_figure = plt.Figure(figsize=(12, 6), constrained_layout=True)
        self.training_canvas = FigureCanvas(self.training_figure)
        self.training_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_canvas.setMinimumHeight(600)
        self.training_layout.addWidget(self.training_canvas)

    def _connect_signals(self):
        """Connect all UI signals to handlers."""
        self.lstm_btn.clicked.connect(lambda: self.run("LSTM"))
        self.trans_btn.clicked.connect(lambda: self.run("Transformer"))
        self.gru_cnn_btn.clicked.connect(lambda: self.run("GRUCNN"))
        self.q_btn.clicked.connect(lambda: self.run("QML"))
        self.save_button.clicked.connect(self.save_forecast)
        self.view_folder_button.clicked.connect(self.open_forecast_folder)
        self.date_range_combo.currentIndexChanged.connect(self.toggle_custom_date_inputs)

    def toggle_custom_date_inputs(self):
        """Show/hide custom date inputs based on date range selection."""
        is_custom = self.date_range_combo.currentText() == "Custom Range"
        self.start_input.setVisible(is_custom)
        self.end_input.setVisible(is_custom)

    def save_forecast(self):
        """Save forecast predictions to CSV files."""
        if not self.last_forecast or not self.last_ticker or not self.current_model:
            return

        import pandas as pd
        from datetime import datetime

        output_dir = os.path.join("forecasts", self.last_ticker)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        for label, preds in self.last_forecast.items():
            dates = [
                self.ticker_data[self.current_data_key]['data'].index[-1] + timedelta(days=i)
                for i in range(1, len(preds) + 1)
            ]
            df = pd.DataFrame({"Date": dates, "Forecast": preds})
            fname = f"{self.current_model}_{label}_{timestamp}.csv"
            full_path = os.path.join(output_dir, fname)
            df.to_csv(full_path, index=False)
            saved_files.append(fname)

        self.save_button.setEnabled(False)
        QMessageBox.information(
            self, "Forecast Saved",
            f"Saved forecast CSVs:\n\n" + "\n".join(saved_files),
            QMessageBox.Ok,
        )
        self.view_folder_button.setEnabled(True)
        self.forecast_folder_path = output_dir
        logger.info(f"Forecast saved to {output_dir}")

    def open_forecast_folder(self):
        """Open the forecast output folder in the system file browser."""
        if not self.forecast_folder_path or not os.path.isdir(self.forecast_folder_path):
            return
        path = os.path.abspath(self.forecast_folder_path)
        if sys.platform.startswith("darwin"):
            os.system(f"open '{path}'")
        elif os.name == "nt":
            os.startfile(path)
        elif os.name == "posix":
            os.system(f"xdg-open '{path}'")

    def _parse_date_range(self):
        """Parse the selected date range into start, end, years_ago."""
        range_option = self.date_range_combo.currentText()
        start = end = None
        years_ago = None

        if "2" in range_option:
            years_ago = 2
        elif "3" in range_option:
            years_ago = 3
        elif "5" in range_option:
            years_ago = 5
        elif "Custom" in range_option:
            start = self.start_input.text().strip()
            end = self.end_input.text().strip()

        return start, end, years_ago

    def _get_or_fetch_data(self, ticker, source, start, end, years_ago):
        """Get cached data or fetch fresh data for a ticker."""
        key = (ticker, source, start, end, years_ago)
        if key not in self.ticker_data:
            X, y, data, scaler, look_back = fetch_and_prepare_data(
                ticker, source=source, start=start, end=end, years_ago=years_ago,
            )
            self.ticker_data[key] = {
                "X": X, "y": y, "data": data,
                "scaler": scaler, "look_back": look_back,
            }
        return key

    def run(self, mode):
        """Start model training and prediction for the selected mode."""
        self.progress.show()
        self.toggle_buttons(False, mode)
        self.current_model = mode
        self.training_axes_state = None
        self.forecast_figure.clear()
        self.training_figure.clear()

        ticker = self.ticker_combo.currentText()
        source = self.source_combo.currentText()
        start, end, years_ago = self._parse_date_range()

        # Fetch primary ticker data
        key = self._get_or_fetch_data(ticker, source, start, end, years_ago)
        self.current_data_key = key
        X, y, data, scaler, look_back = self.ticker_data[key].values()

        # Create trainer
        if mode == "QML":
            trainer = ModelTrainerThread(
                ticker, X, y, data, scaler, look_back,
                lambda _: None,
                type("QuantumWrap", (), {
                    "optimize": optimize_quantum_weights,
                    "predict": quantum_predict_future,
                }),
                use_quantum=True,
            )
        else:
            model_fn = {
                "LSTM": build_lstm_model,
                "Transformer": build_transformer_model,
                "GRUCNN": build_gru_cnn_model,
            }.get(mode, build_lstm_model)
            trainer = ModelTrainerThread(
                ticker, X, y, data, scaler, look_back,
                model_fn, predict_future,
            )

        trainer.finished.connect(self.on_complete)
        self.training_thread = trainer
        trainer.start()
        logger.info(f"Started {mode} training for {ticker}")

        # Handle comparison ticker
        compare_enabled = self.compare_checkbox.isChecked()
        compare_ticker = self.compare_combo.currentText()

        if compare_enabled and compare_ticker != ticker:
            compare_key = self._get_or_fetch_data(compare_ticker, source, start, end, years_ago)
            self.compare_data_key = compare_key
            X2, y2, data2, scaler2, look_back2 = self.ticker_data[compare_key].values()

            model_fn = {
                "LSTM": build_lstm_model,
                "Transformer": build_transformer_model,
                "GRUCNN": build_gru_cnn_model,
            }.get(mode, build_lstm_model)

            self.compare_thread = ModelTrainerThread(
                compare_ticker, X2, y2, data2, scaler2, look_back2,
                model_fn, predict_future,
            )
            self.compare_thread.finished.connect(self.on_compare_complete)
            self.compare_thread.start()
            logger.info(f"Started comparison {mode} training for {compare_ticker}")

    def on_compare_complete(self, compare_results, compare_ticker, last_date, history):
        """Handle comparison model training completion."""
        self.compare_forecast = compare_results
        self.compare_history = history
        self.compare_ticker = compare_ticker
        self.compare_last_date = last_date
        self.compare_thread = None

        if self.last_forecast is not None:
            self.training_axes_state = plot_training_curves(
                self.training_figure, self.training_canvas,
                self.compare_history, label_prefix=self.compare_ticker,
                axes_state=self.training_axes_state,
            )
        logger.info(f"Comparison complete for {compare_ticker}")

    def toggle_buttons(self, enable, running_label=""):
        """Enable/disable model buttons during training."""
        for btn, name in [
            (self.lstm_btn, "LSTM"),
            (self.trans_btn, "Transformer"),
            (self.gru_cnn_btn, "GRUCNN"),
            (self.q_btn, "QML"),
        ]:
            btn.setEnabled(enable)
            btn.setText("Running..." if name == running_label and not enable else f"Run {name}")

    def on_complete(self, results, ticker, last_date, history):
        """Handle primary model training completion."""
        self.progress.hide()
        self.toggle_buttons(True)

        if not results:
            logger.warning(f"Training returned no results for {ticker}")
            return

        self.last_forecast = results
        self.last_ticker = ticker
        self.last_history = history
        self.last_date = last_date
        self.save_button.setEnabled(True)

        # Update trading strategy output
        for label, preds in results.items():
            future_dates = [last_date + timedelta(days=i) for i in range(1, len(preds) + 1)]
            trades, cash, go, stats = trading_strategy(
                preds, last_date, future_dates,
                verbose=self.debug_checkbox.isChecked(),
            )
            summary = f"Initial ${cfg['trading']['initial_cash']:,}\n{label} Trades:\n"
            summary += "\n".join(trades) + f"\nCash: {cash}\nDecision: {go}\n"
            summary += "\n" + "\n".join(f"{k}: {v}" for k, v in stats.items())
            self.output_texts[label].setText(summary)

        # Update forecast plot
        plot_forecast(
            self.forecast_figure, self.forecast_canvas,
            self.ticker_data, self.current_data_key,
            self.last_ticker, last_date, self.last_forecast,
            compare_forecast=self.compare_forecast,
            compare_ticker=self.compare_ticker,
            compare_data_key=self.compare_data_key,
            compare_last_date=getattr(self, 'compare_last_date', None),
        )

        # Update training curves
        self.training_axes_state = plot_training_curves(
            self.training_figure, self.training_canvas,
            history, label_prefix=self.last_ticker,
            axes_state=None,
        )
        logger.info(f"Training complete for {ticker} — {self.current_model}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = StockTradingGUI()
    win.show()
    sys.exit(app.exec_())
