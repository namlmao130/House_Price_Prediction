"""CV, chọn model, đánh giá test và refit; preprocessing độc lập từng lần fit."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import N_SPLITS, RANDOM_STATE
from evaluation import evaluate_regression, report_metrics
from preprocessing import HousePricePreprocessor
from validation import align_series, align_target, ensure_disjoint_groups

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Các kết quả cùng thuộc một thí nghiệm, tránh tuple dài dễ nhầm vị trí."""

    best_name: str
    metrics: dict
    audit: dict
    predictions: pd.DataFrame
    fold_assignment: pd.Series
    model: object
    preprocessor: HousePricePreprocessor


def get_models() -> dict:
    """Giữ nguyên estimator và hyperparameter; chỉ RF dùng nhiều lõi như trước."""
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()), ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def cross_validate(
    X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series,
    models: dict,
) -> tuple[dict, list[dict], pd.Series]:
    """Mỗi outer-fold fit preprocessing mới; dùng chung feature fold cho các model."""
    y_train = align_target(y_train, X_train.index)
    groups_train = align_series(groups_train, X_train.index, "groups_train")
    if groups_train.nunique() < N_SPLITS:
        raise ValueError(f"CV cần ít nhất {N_SPLITS} group train sau deduplicate.")
    splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_assignment = pd.Series(index=X_train.index, dtype="Int64", name="cv_fold")
    scores = {name: [] for name in models}
    audits = []
    for number, (train_pos, valid_pos) in enumerate(
        splitter.split(X_train, groups=groups_train), 1,
    ):
        ensure_disjoint_groups(
            groups_train.iloc[train_pos], groups_train.iloc[valid_pos],
        )
        preprocessor = HousePricePreprocessor()
        X_fit, y_fit = preprocessor.fit_transform(
            X_train.iloc[train_pos], y_train.iloc[train_pos], groups_train.iloc[train_pos],
        )
        X_valid = preprocessor.transform(X_train.iloc[valid_pos])
        y_valid = y_train.iloc[valid_pos]
        if not X_valid.index.equals(y_valid.index):
            raise RuntimeError("Validation bị thay đổi index/số dòng sau transform.")
        fold_assignment.loc[X_valid.index] = number
        audits.append({
            "fold": number,
            "train_rows_before_filter": len(train_pos),
            "train_rows_after_filter": len(X_fit),
            "validation_rows": len(valid_pos),
            "outlier_bounds": preprocessor.outlier_bounds_,
            "train_outlier_filter": "none",
            "numeric_medians": preprocessor.numeric_medians_,
            "group_overlap": 0,
        })
        # Chỉ giữ ma trận của một fold trong RAM, không giữ cả năm fold.
        for name, estimator in models.items():
            logger.info("CV %s: fold %s/%s", name, number, N_SPLITS)
            model = clone(estimator).fit(X_fit, np.log1p(y_fit))
            metrics = evaluate_regression(y_valid, np.expm1(model.predict(X_valid)))
            scores[name].append(metrics)
            report_metrics(f"CV {name} fold {number}", metrics)
    summary = {
        name: {
            "cv_r2_mean": float(np.mean([score["r2"] for score in fold_scores])),
            "cv_r2_std": float(np.std([score["r2"] for score in fold_scores])),
            "cv_mae": float(np.mean([score["mae"] for score in fold_scores])),
            "cv_rmse": float(np.mean([score["rmse"] for score in fold_scores])),
            "cv_mape": float(np.mean([score["mape"] for score in fold_scores])),
            "fold_scores": fold_scores,
        }
        for name, fold_scores in scores.items()
    }
    return summary, audits, fold_assignment


def train_and_evaluate(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, groups_train: pd.Series,
) -> EvaluationResult:
    """Chọn bằng CV trên train trước khi fit/evaluate trên tập test cố định."""
    y_train = align_target(y_train, X_train.index)
    y_test = align_target(y_test, X_test.index)
    groups_train = align_series(groups_train, X_train.index, "groups_train")
    if groups_train.duplicated().any():
        raise ValueError("Train còn duplicate group; hãy deduplicate sau split.")
    if len(y_test) < 2:
        raise ValueError("Test cần ít nhất hai mẫu để tính R².")
    models = get_models()
    summary, folds, assignment = cross_validate(X_train, y_train, groups_train, models)
    best_name = max(summary, key=lambda name: summary[name]["cv_r2_mean"])
    logger.info("Chọn bằng CV trước khi đánh giá test: %s", best_name)
    preprocessor = HousePricePreprocessor()
    X_fit, y_fit = preprocessor.fit_transform(X_train, y_train, groups_train)
    X_test_encoded = preprocessor.transform(X_test)
    predictions = pd.DataFrame({"price_ty": y_test})
    best_model = None
    for name, estimator in models.items():
        logger.info("Fit %s để đánh giá toàn bộ %s dòng test", name, len(y_test))
        model = clone(estimator).fit(X_fit, np.log1p(y_fit))
        predictions[name] = np.expm1(model.predict(X_test_encoded))
        metrics = evaluate_regression(y_test, predictions[name].to_numpy())
        summary[name].update({f"test_{key}": value for key, value in metrics.items()})
        report_metrics(f"Test {name}", metrics)
        if name == best_name:
            best_model = model
    baseline = evaluate_regression(y_test, np.full(len(y_test), float(y_train.median())))
    audit = {
        "folds": folds,
        "train_rows_before_filter": len(X_train),
        "train_rows_after_filter": len(X_fit),
        "test_rows": len(X_test),
        "test_rows_removed_by_outlier_filter": 0,
        "test_outlier_filter": "none",
        "train_outlier_filter": "none",
        "outlier_bounds": preprocessor.outlier_bounds_,
        "numeric_medians": preprocessor.numeric_medians_,
        "baseline_train_median_test_mae": baseline["mae"],
    }
    return EvaluationResult(
        best_name, summary, audit, predictions, assignment, best_model, preprocessor,
    )


def refit_final_model(
    best_name: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
) -> tuple[object, HousePricePreprocessor]:
    """Refit sau đánh giá trên dữ liệu khử trùng; không báo lại test score."""
    logger.info("Refit model triển khai: %s", best_name)
    preprocessor = HousePricePreprocessor()
    X_fit, y_fit = preprocessor.fit_transform(X, y, groups)
    model = get_models()[best_name].fit(X_fit, np.log1p(y_fit))
    return model, preprocessor


def get_feature_importance(model: object, columns: list[str]) -> pd.Series:
    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model
    values = (
        inner.feature_importances_ if hasattr(inner, "feature_importances_")
        else np.abs(inner.coef_)
    )
    return pd.Series(values, index=columns).sort_values(ascending=False)
