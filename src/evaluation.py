"""Tính và báo cáo metric độc lập với cách fit model."""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

PRICE_BAND_EDGES = (0.0, 3.0, 6.0, 10.0, 20.0, np.inf)
PRICE_BAND_LABELS = ("<3 tỷ", "3–6 tỷ", "6–10 tỷ", "10–20 tỷ", ">20 tỷ")
MIN_SEGMENT_ROWS = 200
HIT_RATE_THRESHOLDS = (0.10, 0.20)


def _validate(y_true, predictions) -> tuple[np.ndarray, np.ndarray]:
    if len(y_true) < 2 or len(y_true) != len(predictions):
        raise ValueError("Đánh giá cần ít nhất hai dòng và số dự đoán khớp y.")
    if not np.isfinite(y_true).all() or not np.isfinite(predictions).all():
        raise ValueError("Nhãn/dự đoán đánh giá phải hữu hạn.")
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    if np.any(actual <= 0):
        raise ValueError("MAPE yêu cầu giá thật lớn hơn 0.")
    return actual, predicted


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict:
    """Metric trên giá gốc, bổ sung sai số tương đối và metric thang log."""
    actual, predicted = _validate(y_true, predictions)
    ape = np.abs(actual - predicted) / actual
    log_actual = np.log1p(actual)
    log_predicted = np.log1p(np.clip(predicted, 0, None))
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(ape) * 100),
        "median_ape": float(np.median(ape) * 100),
        "hit_rate_10": float(np.mean(ape <= HIT_RATE_THRESHOLDS[0])),
        "hit_rate_20": float(np.mean(ape <= HIT_RATE_THRESHOLDS[1])),
        "bias": float(np.median(predicted / actual)),
        "r2_log": float(r2_score(log_actual, log_predicted)),
        "rmsle": float(np.sqrt(mean_squared_error(log_actual, log_predicted))),
    }


def _segment_row(label: str, actual: np.ndarray, predicted: np.ndarray) -> dict:
    ape = np.abs(actual - predicted) / actual
    return {
        "segment": str(label),
        "n": int(len(actual)),
        "median_price": float(np.median(actual)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "median_ape": float(np.median(ape) * 100),
        "hit_rate_20": float(np.mean(ape <= HIT_RATE_THRESHOLDS[1])),
        "bias": float(np.median(predicted / actual)),
    }


def evaluate_by_segment(
    y_true: pd.Series, predictions: np.ndarray, segments: pd.Series,
    min_rows: int = MIN_SEGMENT_ROWS,
) -> list[dict]:
    """Tính sai số theo nhóm, bỏ các nhóm quá ít mẫu."""
    actual, predicted = _validate(y_true, predictions)
    segments = pd.Series(np.asarray(segments, dtype=object)).fillna("Không rõ")
    rows = []
    for label in sorted(segments.unique(), key=str):
        mask = (segments == label).to_numpy()
        if mask.sum() >= min_rows:
            rows.append(_segment_row(label, actual[mask], predicted[mask]))
    return rows


def evaluate_by_price_band(y_true: pd.Series, predictions: np.ndarray) -> list[dict]:
    """Tính sai số theo các khoảng giá thật cố định."""
    actual, predicted = _validate(y_true, predictions)
    bands = pd.cut(
        actual, bins=list(PRICE_BAND_EDGES), labels=list(PRICE_BAND_LABELS),
        right=False,
    )
    rows = []
    for label in PRICE_BAND_LABELS:
        mask = np.asarray(bands == label)
        if mask.sum() >= MIN_SEGMENT_ROWS:
            rows.append(_segment_row(label, actual[mask], predicted[mask]))
    return rows


def report_metrics(label: str, metrics: dict) -> None:
    logger.info(
        "%s: R²=%.4f; MAE=%.4f; RMSE=%.4f tỷ; MAPE=%.2f%%; "
        "median APE=%.2f%%; ±20%%=%.1f%%; R² log=%.4f",
        label, metrics["r2"], metrics["mae"], metrics["rmse"], metrics["mape"],
        metrics["median_ape"], metrics["hit_rate_20"] * 100, metrics["r2_log"],
    )
