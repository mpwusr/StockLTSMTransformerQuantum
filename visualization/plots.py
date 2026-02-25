"""Visualization functions for stock forecasts and training diagnostics."""

from datetime import timedelta
from config import cfg
from utils.logging_config import setup_logging

logger = setup_logging("visualization", "visualization.log")


def plot_forecast(figure, canvas, ticker_data, current_data_key, last_ticker,
                  last_date, pred_dict, compare_forecast=None, compare_ticker=None,
                  compare_data_key=None, compare_last_date=None):
    """Plot forecast results for primary and optional comparison ticker.

    Args:
        figure: matplotlib Figure object
        canvas: FigureCanvasQTAgg to draw on
        ticker_data: dict of cached ticker data
        current_data_key: key for primary ticker data
        last_ticker: primary ticker symbol
        last_date: last date of actual data
        pred_dict: dict with Short/Medium/Long predictions
        compare_forecast: optional comparison predictions
        compare_ticker: optional comparison ticker symbol
        compare_data_key: optional comparison data key
        compare_last_date: optional comparison last date
    """
    figure.clear()
    ax = figure.add_subplot(111)

    # Primary ticker actual prices
    real = ticker_data[current_data_key]["data"]
    ax.plot(real.index, real["Close"], label=f"{last_ticker} Actual", color="black")

    # Primary ticker forecasts
    base = [last_date + timedelta(days=i) for i in range(1, cfg["horizons"]["long"] + 1)]
    short_days = cfg["horizons"]["short"]
    medium_days = cfg["horizons"]["medium"]

    colors = cfg["gui"]["model_colors"]
    ax.plot(base[:short_days], pred_dict["Short"], label=f"{last_ticker} Short", color="blue")
    ax.plot(base[short_days:medium_days], pred_dict["Medium"][-(medium_days - short_days):],
            label=f"{last_ticker} Medium", color="orange")
    ax.plot(base[medium_days:], pred_dict["Long"][-(cfg['horizons']['long'] - medium_days):],
            label=f"{last_ticker} Long", color="green")

    # Comparison ticker
    if compare_forecast and compare_data_key:
        compare_real = ticker_data[compare_data_key]["data"]
        ax.plot(compare_real.index, compare_real["Close"],
                label=f"{compare_ticker} Actual", linestyle="--", color="gray")

        base2 = [compare_last_date + timedelta(days=i) for i in range(1, cfg["horizons"]["long"] + 1)]
        ax.plot(base2[:short_days], compare_forecast["Short"],
                label=f"{compare_ticker} Short", linestyle="--", color="purple")
        ax.plot(base2[short_days:medium_days], compare_forecast["Medium"][-(medium_days - short_days):],
                label=f"{compare_ticker} Medium", linestyle="--", color="brown")
        ax.plot(base2[medium_days:], compare_forecast["Long"][-(cfg['horizons']['long'] - medium_days):],
                label=f"{compare_ticker} Long", linestyle="--", color="red")

    ax.set_xlim(real.index.min(), base[-1])
    ax.set_title("Forecast Comparison")
    ax.legend(loc="best")
    ax.figure.autofmt_xdate()
    canvas.draw()
    logger.info(f"Forecast plot updated for {last_ticker}")


def plot_training_curves(figure, canvas, history, label_prefix="",
                         axes_state=None):
    """Plot training diagnostics: loss, MAE, and validation metrics.

    Args:
        figure: matplotlib Figure object
        canvas: FigureCanvasQTAgg to draw on
        history: training history dict
        label_prefix: prefix for legend labels (e.g. ticker symbol)
        axes_state: dict to track axes initialization state, or None to always create fresh

    Returns:
        dict: updated axes state
    """
    if not history:
        logger.warning("No training history available for plotting")
        return axes_state

    # Initialize axes if needed
    if axes_state is None or not axes_state.get("initialized"):
        figure.clear()
        figure.set_constrained_layout(False)
        figure.subplots_adjust(right=0.78)
        gs = figure.add_gridspec(3, 1)

        ax_loss = figure.add_subplot(gs[0, 0])
        ax_mae = figure.add_subplot(gs[1, 0])
        ax_val = figure.add_subplot(gs[2, 0])

        ax_loss.set_ylabel("MSE")
        ax_mae.set_ylabel("MAE")
        ax_val.set_ylabel("Metric Value")
        ax_val.set_ylim(bottom=-5, top=10)

        axes_state = {
            "initialized": True,
            "ax_loss": ax_loss,
            "ax_mae": ax_mae,
            "ax_val": ax_val,
        }

    ax_loss = axes_state["ax_loss"]
    ax_mae = axes_state["ax_mae"]
    ax_val = axes_state["ax_val"]

    def lbl(name):
        return f"{label_prefix} {name}" if label_prefix else name

    # Loss curves
    if "loss" in history and len(history["loss"]):
        ax_loss.plot(history["loss"], label=lbl("Train Loss"))
    if "val_loss" in history and len(history["val_loss"]):
        ax_loss.plot(history["val_loss"], label=lbl("Val Loss"))
    ax_loss.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    # MAE curves
    if "train_mae" in history and len(history["train_mae"]):
        ax_mae.plot(history["train_mae"], label=lbl("Train MAE"))
    if "val_mae" in history and len(history["val_mae"]):
        ax_mae.plot(history["val_mae"], label=lbl("Val MAE"))
    ax_mae.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    # Validation metrics
    if "val_mape" in history and len(history["val_mape"]):
        ax_val.plot(history["val_mape"], label=lbl("Val MAPE"))
    if "val_r2" in history and len(history["val_r2"]):
        ax_val.plot(history["val_r2"], label=lbl("Val R²"))
    if "val_sharpe" in history and len(history["val_sharpe"]):
        ax_val.plot(history["val_sharpe"], label=lbl("Sharpe Ratio"))
    ax_val.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    canvas.draw()
    logger.info(f"Training curves updated for {label_prefix}")
    return axes_state
