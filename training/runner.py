"""Training orchestration: model training thread, callbacks, and prediction."""

import os
import numpy as np
import pandas as pd
import pennylane.numpy as qnp
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from config import cfg
from utils.logging_config import setup_logging
from models import build_lstm_model, build_transformer_model, build_gru_cnn_model

logger = setup_logging("training_runner", "training_runner.log")


class MetricsTrackingCallback(Callback):
    """Custom Keras callback tracking MAE, MAPE, R², and Sharpe ratio per epoch."""

    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.history = {
            "train_mae": [],
            "val_mae": [],
            "val_mape": [],
            "val_r2": [],
            "val_sharpe": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0).squeeze()
        y_val_pred = self.model.predict(self.X_val, verbose=0).squeeze()

        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        val_mae = mean_absolute_error(self.y_val, y_val_pred)
        val_mape = mean_absolute_percentage_error(self.y_val, y_val_pred)
        val_r2 = r2_score(self.y_val, y_val_pred)

        daily_returns = np.diff(y_val_pred) / y_val_pred[:-1]
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) != 0 else 0.0

        self.history["train_mae"].append(train_mae)
        self.history["val_mae"].append(val_mae)
        self.history["val_mape"].append(val_mape)
        self.history["val_r2"].append(val_r2)
        self.history["val_sharpe"].append(sharpe)

        if logs is not None:
            logs["val_mae"] = val_mae

        logger.debug(f"Epoch {epoch+1}: MAE={val_mae:.4f}, MAPE={val_mape:.4f}, R²={val_r2:.4f}, Sharpe={sharpe:.4f}")


def predict_future(model, last_sequence, scaler, future_days):
    """Generate future price predictions using autoregressive loop."""
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = pred

    predictions = np.array(predictions).reshape(-1, 1)
    num_features = scaler.scale_.shape[0]
    padding = num_features - 1
    padded = np.concatenate((predictions, np.zeros((len(predictions), padding))), axis=1)
    return np.maximum(scaler.inverse_transform(padded)[:, 0], 0)


class ModelTrainerThread(QThread):
    """Background thread for model training and prediction."""

    finished = pyqtSignal(dict, str, object, dict)

    def __init__(self, ticker, X, y, data, scaler, look_back, model_builder, predictor, use_quantum=False):
        super().__init__()
        self.ticker = ticker
        self.X, self.y = X, y
        self.data = data
        self.scaler = scaler
        self.look_back = look_back
        self.model_builder = model_builder
        self.predictor = predictor
        self.use_quantum = use_quantum
        self.model_dir = f"models/{ticker}"
        os.makedirs(self.model_dir, exist_ok=True)
        self.history = {}

    def run(self):
        try:
            results = {}
            last_date = self.data.index[-1]
            train_size = int(len(self.X) * (1 - cfg["training"]["validation_split"]))
            X_train, X_val = self.X[:train_size], self.X[train_size:]
            y_train, y_val = self.y[:train_size], self.y[train_size:]

            if self.use_quantum:
                from models import optimize_quantum_weights, quantum_predict_future
                weights_path = f"{self.model_dir}/quantum_weights.npy"
                if os.path.exists(weights_path):
                    weights = qnp.load(weights_path)
                else:
                    weights = qnp.random.random(4, requires_grad=True)
                weights = self.predictor.optimize(X_train, y_train, iterations=cfg["model"]["quantum"]["iterations"], verbose=True)
                qnp.save(weights_path, weights)
                pred_func = lambda seq, days, horizon: self.predictor.predict(
                    seq, weights, self.scaler, self.look_back, days, horizon
                )
                self.history = {}
            else:
                model_key = (
                    "lstm_model" if self.model_builder == build_lstm_model else
                    "transformer_model" if self.model_builder == build_transformer_model else
                    "gru_cnn_model"
                )
                model_path = f"{self.model_dir}/{model_key}.keras"

                if os.path.exists(model_path):
                    try:
                        model = load_model(model_path)
                        logger.info(f"Loaded cached model from {model_path}")
                    except Exception as e:
                        logger.warning(f"Model load failed: {e}. Rebuilding.")
                        model = self.model_builder((self.X.shape[1], self.X.shape[2]))
                else:
                    model = self.model_builder((self.X.shape[1], self.X.shape[2]))

                checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss")
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=cfg["training"]["patience"],
                    restore_best_weights=True,
                )
                metrics_tracker = MetricsTrackingCallback(X_train, y_train, X_val, y_val)

                history = model.fit(
                    X_train, y_train,
                    epochs=cfg["training"]["epochs"],
                    batch_size=cfg["training"]["batch_size"],
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, early_stopping, metrics_tracker],
                    verbose=0,
                )
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")

                self.history = history.history
                self.history.update(metrics_tracker.history)

                pred_func = lambda seq, days, horizon: predict_future(model, seq, self.scaler, days)

            # Generate predictions for all horizons
            last_seq = self.X[-1]
            horizons = cfg["horizons"]
            for label, days in {"Short": horizons["short"], "Medium": horizons["medium"], "Long": horizons["long"]}.items():
                results[label] = pred_func(last_seq, days, label)

            self.finished.emit(results, self.ticker, last_date, self.history)

            # Save training log
            self._save_training_log()

        except Exception as e:
            logger.error(f"Training failed for {self.ticker}: {e}", exc_info=True)
            self.finished.emit({}, self.ticker, None, {})

    def _save_training_log(self):
        """Save training metrics to timestamped CSV."""
        if not self.history:
            return

        log_dir = os.path.join(cfg["data"]["log_dir"], self.ticker)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_tag = (
            "lstm" if self.model_builder == build_lstm_model else
            "transformer" if self.model_builder == build_transformer_model else
            "gru_cnn"
        )
        csv_path = os.path.join(log_dir, f"{model_tag}_training_log_{timestamp}.csv")
        pd.DataFrame(self.history).to_csv(csv_path, index=False)
        logger.info(f"Training log saved to {csv_path}")
