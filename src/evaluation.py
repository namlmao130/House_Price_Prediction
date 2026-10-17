"""Tính và báo cáo metric độc lập với cách fit model."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Metric trên giá gốc; MAPE theo phần trăm, yêu cầu giá thật dương."""
    if len(y_true) < 2 or len(y_true) != len(predictions):
        raise ValueError("Đánh giá cần ít nhất hai dòng và số dự đoán khớp y.")
    if not np.isfinite(y_true).all() or not np.isfinite(predictions).all():
        raise ValueError("Nhãn/dự đoán đánh giá phải hữu hạn.")
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if np.any(actual <= 0):
        raise ValueError("MAPE yêu cầu giá thật lớn hơn 0.")
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }


def report_metrics(label: str, metrics: dict) -> None:
    logger.info(
        "%s: R²=%.4f; MAE=%.4f; RMSE=%.4f tỷ; MAPE=%.2f%%",
        label, metrics["r2"], metrics["mae"], metrics["rmse"], metrics["mape"],
    )
