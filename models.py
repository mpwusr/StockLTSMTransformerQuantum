"""Model architectures and trading strategy for StockLTSMTransformerQuantum.

Contains LSTM, Transformer, GRU-CNN builders, quantum ML circuit,
and trading strategy evaluation.
"""

import warnings
import os
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention,
    LayerNormalization, Add, Conv1D, MaxPooling1D
)
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.models import Model
from ta.momentum import RSIIndicator

from config import cfg
from utils.logging_config import setup_logging

warnings.filterwarnings("ignore", category=UserWarning, module="keras")

logger = setup_logging("models", "models.log")

# --- Utility Functions ---

def moving_average(data, window=None):
    """Compute moving average using convolution."""
    if window is None:
        window = cfg["trading"]["moving_average_window"]
    return np.convolve(data, np.ones(window) / window, mode='valid')


def compute_rsi_from_series(close_series, period=14):
    """Compute RSI from a pandas Series."""
    rsi_indicator = RSIIndicator(close_series, window=period)
    return rsi_indicator.rsi().dropna().values


# --- Trading Strategy ---

def trading_strategy(predictions, last_date, future_dates, rsi_series=None,
                     rsi_threshold=None, verbose=False):
    """Evaluate trading strategy on predicted prices.

    Uses moving average crossover with RSI filter, stop-loss, and take-profit.

    Returns:
        tuple: (positions, final_cash_str, go_no_go, stats)
    """
    trading_cfg = cfg["trading"]
    if rsi_threshold is None:
        rsi_threshold = tuple(trading_cfg["rsi_threshold"])

    positions = []
    cash = trading_cfg["initial_cash"]
    initial_cash = cash
    shares = 0
    stop_loss = trading_cfg["stop_loss"]
    take_profit = trading_cfg["take_profit"]
    window = trading_cfg["moving_average_window"]
    entry_price = 0
    num_trades = 0
    wins = 0
    losses = 0

    ma_predictions = moving_average(predictions, window)
    ma_indices = range(window - 1, len(predictions))

    for i in ma_indices:
        pred_price = predictions[i]
        date_str = future_dates[i].strftime("%m%d%Y")
        ma_current = ma_predictions[i - (window - 1)]
        ma_previous = ma_predictions[i - window] if i > window - 1 else ma_predictions[0]

        rsi_ok = True
        if rsi_series is not None and i < len(rsi_series):
            rsi = rsi_series[i]
            rsi_ok = rsi < rsi_threshold[1]

        if shares == 0 and ma_current > ma_previous and rsi_ok and cash > pred_price > 0:
            shares_to_buy = max(1, int(cash / pred_price))
            shares += shares_to_buy
            cash -= shares_to_buy * pred_price
            entry_price = pred_price
            num_trades += 1
            positions.append(f"{date_str} Buy {shares_to_buy} @ ${pred_price:.2f}")
            if verbose:
                os.makedirs("logs", exist_ok=True)
                logger.debug(f"BUY on {date_str} at {pred_price:.2f}")

        elif shares > 0:
            if pred_price <= entry_price * (1 - stop_loss) or pred_price >= entry_price * (1 + take_profit):
                proceeds = shares * pred_price
                cash += proceeds
                profit = proceeds - (entry_price * shares)
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
                positions.append(f"{date_str} Sell {shares} @ ${pred_price:.2f} | {'Win' if profit > 0 else 'Loss'}")
                if verbose:
                    logger.debug(f"SELL on {date_str} at {pred_price:.2f} | {'Win' if profit > 0 else 'Loss'}")
                shares = 0

    if shares > 0:
        final_price = predictions[-1]
        proceeds = shares * final_price
        cash += proceeds
        profit = proceeds - (entry_price * shares)
        if profit > 0:
            wins += 1
        else:
            losses += 1
        positions.append(
            f"{future_dates[-1].strftime('%m%d%Y')} Final Sell {shares} @ ${final_price:.2f} | {'Win' if profit > 0 else 'Loss'}"
        )
        if verbose:
            logger.debug(f"FINAL SELL at {final_price:.2f}")

    return_pct = ((cash - initial_cash) / initial_cash) * 100
    go_no_go = "Go" if cash > initial_cash else "No Go"

    win_rate = wins / num_trades if num_trades > 0 else 0.0
    stats = {
        "Return %": f"{return_pct:.2f}%",
        "Trades": num_trades,
        "Wins": wins,
        "Losses": losses,
        "Win Rate": f"{win_rate:.1%}",
    }

    return positions, f"${cash:.2f}", go_no_go, stats


def add_noise_debug_overlay(ax, predictions, dates, label="Quantum Volatility", fill=False):
    """Add prediction volatility overlay to a plot axis."""
    diff = np.abs(np.diff(predictions))
    smoothed = np.convolve(diff, np.ones(5) / 5, mode="same")
    ax2 = ax.twinx()
    ax2.plot(dates[1:], smoothed, color="gray", alpha=0.3, label=label, linestyle="dotted")
    if fill:
        ax2.fill_between(dates[1:], smoothed, alpha=0.1, color="gray")
    ax2.set_ylabel("Prediction Volatility", fontsize=8)
    ax2.legend(loc="upper left", fontsize=8)


# --- Classical Model Builders ---

def build_lstm_model(input_shape):
    """Build LSTM model with Functional API."""
    lstm_cfg = cfg["model"]["lstm"]
    inputs = Input(shape=input_shape)
    x = LSTM(lstm_cfg["units"][0], return_sequences=True)(inputs)
    x = Dropout(lstm_cfg["dropout"])(x)
    x = LSTM(lstm_cfg["units"][1])(x)
    x = Dropout(lstm_cfg["dropout"])(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam', loss='mse',
        metrics=[MeanAbsoluteError(name='mae'), MeanAbsolutePercentageError(name='mape')],
        run_eagerly=False,
    )
    logger.info(f"Built LSTM model: {model.count_params()} parameters")
    return model


def build_transformer_model(input_shape):
    """Build Transformer model with self-attention."""
    trans_cfg = cfg["model"]["transformer"]
    inputs = Input(shape=input_shape)
    attention_output = MultiHeadAttention(
        num_heads=trans_cfg["num_heads"], key_dim=trans_cfg["key_dim"]
    )(inputs, inputs)
    attention_output = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(attention_output)

    ffn_output = Dense(trans_cfg["dense_units"][0], activation="relu")(x)
    ffn_output = Dense(input_shape[-1])(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = x[:, -1, :]
    x = Dense(trans_cfg["dense_units"][1], activation='relu')(x)
    x = Dropout(trans_cfg["dropout"])(x)
    x = Dense(trans_cfg["dense_units"][2], activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam', loss='mse',
        metrics=[MeanAbsoluteError(name='mae'), MeanAbsolutePercentageError(name='mape')],
        run_eagerly=False,
    )
    logger.info(f"Built Transformer model: {model.count_params()} parameters")
    return model


def build_gru_cnn_model(input_shape):
    """Build GRU-CNN hybrid model."""
    gru_cfg = cfg["model"]["gru_cnn"]
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=gru_cfg["conv_filters"], kernel_size=gru_cfg["kernel_size"], activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(gru_cfg["gru_units"][0], return_sequences=True)(x)
    x = GRU(gru_cfg["gru_units"][1])(x)
    x = Dropout(gru_cfg["dropout"])(x)
    x = Dense(50, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam', loss='mse',
        metrics=[MeanAbsoluteError(name='mae'), MeanAbsolutePercentageError(name='mape')],
        run_eagerly=False,
    )
    logger.info(f"Built GRU-CNN model: {model.count_params()} parameters")
    return model


# --- Quantum ML ---

quantum_cfg = cfg["model"]["quantum"]
n_qubits = quantum_cfg["n_qubits"]
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    """Variational quantum circuit for stock prediction."""
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i], wires=i)

    for layer in range(quantum_cfg["layers"]):
        for i in range(n_qubits):
            qml.RX(weights[layer * n_qubits + i], wires=i)
            qml.RZ(weights[layer * n_qubits + i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.expval(qml.PauliZ(0))


def normalize_input(x):
    """Normalize input features to angle range [-pi, pi] for quantum circuit."""
    if len(x.shape) == 1:
        raw = x[:n_qubits]
    else:
        raw = np.mean(x, axis=0)[:n_qubits]

    min_val = np.min(raw)
    max_val = np.max(raw)
    scaled = (raw - min_val) / (max_val - min_val + 1e-6)
    logger.debug(f"Normalized input (first {n_qubits}): {scaled}")
    return (scaled * 2 * np.pi) - np.pi


def smooth_predictions(preds, window=3):
    """Smooth predictions with moving average."""
    return np.convolve(preds, np.ones(window) / window, mode="same")


def optimize_quantum_weights(X, y, iterations=None, verbose=False):
    """Optimize quantum circuit weights using Adam optimizer."""
    if iterations is None:
        iterations = quantum_cfg["iterations"]

    weights = qnp.random.random(quantum_cfg["layers"] * n_qubits, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=quantum_cfg["step_size"])
    y = qnp.array(y, requires_grad=False)

    for step in range(iterations):
        def cost(w):
            preds = []
            for x in X[:, -1]:
                x_scaled = normalize_input(x)
                pred = quantum_circuit(x_scaled, w)
                preds.append(pred)
            preds = qnp.array(preds)
            if verbose and step % 10 == 0:
                logger.debug(f"[Step {step}] Sample preds: {preds[:5]}")
            return qnp.mean((preds - y) ** 2)

        weights = opt.step(cost, weights)

    logger.info(f"Quantum optimization complete after {iterations} iterations")
    return weights


def quantum_predict_future(last_sequence, weights, scaler, look_back, future_days, horizon="Short"):
    """Generate future predictions using quantum circuit."""
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_days):
        x_scaled = normalize_input(current_sequence)
        pred = quantum_circuit(x_scaled, weights)
        predictions.append(pred)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = pred

    predictions = np.array(predictions)
    logger.debug(f"[Horizon: {horizon}] Raw predictions (first 10): {predictions[:10]}")

    predictions = (predictions + 1) / 2  # Normalize to [0, 1]
    logger.debug(f"[Horizon: {horizon}] Normalized predictions (first 10): {predictions[:10]}")

    # Use scaler's feature count instead of magic number
    n_features = scaler.n_features_in_
    padding = n_features - 1
    padded = np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), padding))), axis=1)
    final_predictions = np.maximum(scaler.inverse_transform(padded)[:, 0], 0)
    logger.debug(f"[Horizon: {horizon}] Final predictions (first 10): {final_predictions[:10]}")

    return final_predictions
