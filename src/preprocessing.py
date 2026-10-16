"""Preprocessor có state: fit chỉ trên train, transform chỉ đọc state."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold

from config import (
    CATEGORICAL_COLUMNS,
    ENCODING_COLUMNS,
    IMPUTE_NUMERIC_COLUMNS,
    INPUT_COLUMNS,
    MISSING_INDICATOR_COLUMNS,
    N_SPLITS,
    NUMERIC_COLUMNS,
    PRICE_UNIT,
    RANDOM_STATE,
)
from validation import (
    align_series,
    align_target,
    ensure_disjoint_groups,
    require_columns,
    require_unique_index,
)


def fit_numeric_imputer(X_train: pd.DataFrame) -> dict[str, float]:
    """Fit median trên train; báo lỗi nếu cả cột không có số hữu hạn."""
    require_columns(X_train, IMPUTE_NUMERIC_COLUMNS)
    numeric = X_train[IMPUTE_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    medians = numeric.replace([np.inf, -np.inf], np.nan).median()
    missing = medians.index[medians.isna()].tolist()
    if missing:
        raise ValueError(f"Train không có giá trị hữu hạn cho: {', '.join(missing)}")
    return {column: float(value) for column, value in medians.items()}


def fill_numeric_missing(X: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Tạo indicator từ đầu vào rồi fill bằng median đã fit; không học lại."""
    require_columns(X, IMPUTE_NUMERIC_COLUMNS)
    filled = X.copy()
    for column in IMPUTE_NUMERIC_COLUMNS:
        if column not in medians or not np.isfinite(medians[column]):
            raise ValueError(f"Thiếu median hữu hạn đã fit cho {column}.")
        numeric = pd.to_numeric(filled[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan,
        )
        filled[f"{column}_missing"] = numeric.isna().astype(int)
        filled[column] = numeric.fillna(medians[column])
    return filled


def fit_target_statistics(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Học median giá quận/phường từ đúng phần train được cung cấp."""
    require_columns(X_train, ["district_name", "ward_name"])
    require_unique_index(X_train)
    encoded = X_train[["district_name", "ward_name"]].copy()
    encoded["price_ty"] = align_target(y_train, X_train.index)
    if encoded.empty:
        raise ValueError("Không thể fit target statistics trên train rỗng.")
    return {
        "districts": encoded.groupby("district_name")["price_ty"].median().to_dict(),
        "ward_medians": (
            encoded.dropna(subset=["ward_name"])
            .groupby(["district_name", "ward_name"])["price_ty"].median().to_dict()
        ),
        "global_median_price": float(encoded["price_ty"].median()),
    }


def transform_target_encoding(X: pd.DataFrame, statistics: dict) -> pd.DataFrame:
    """Fallback phường → quận → toàn train; không nhận y của tập cần biến đổi."""
    district_prices = X["district_name"].map(statistics["districts"]).fillna(
        statistics["global_median_price"]
    )
    ward_prices = [
        statistics["ward_medians"].get((district, ward), district_price)
        for district, ward, district_price in zip(
            X["district_name"], X["ward_name"], district_prices,
        )
    ]
    return pd.DataFrame(
        {
            "district_median_price": district_prices.astype(float),
            "ward_median_price": pd.Series(ward_prices, index=X.index, dtype=float),
        },
        index=X.index,
    )


def make_oof_target_encoding(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Mỗi group chỉ nhận target statistics của các inner-train group khác."""
    require_unique_index(X_train)
    y_train = align_target(y_train, X_train.index)
    groups_train = align_series(groups_train, X_train.index, "groups_train")
    split_count = min(n_splits, groups_train.nunique())
    if split_count < 2:
        raise ValueError("Cần ít nhất hai group train để tạo target encoding OOF.")
    splitter = GroupKFold(
        n_splits=split_count, shuffle=True, random_state=random_state,
    )
    encoded = pd.DataFrame(index=X_train.index, columns=ENCODING_COLUMNS, dtype=float)
    for fit_pos, holdout_pos in splitter.split(X_train, groups=groups_train):
        fit_index, holdout_index = X_train.index[fit_pos], X_train.index[holdout_pos]
        ensure_disjoint_groups(
            groups_train.loc[fit_index], groups_train.loc[holdout_index],
        )
        statistics = fit_target_statistics(
            X_train.loc[fit_index], y_train.loc[fit_index],
        )
        encoded.loc[holdout_index] = transform_target_encoding(
            X_train.loc[holdout_index], statistics,
        )
    if encoded.isna().any().any():
        raise RuntimeError("Target encoding OOF còn giá trị thiếu.")
    return encoded


class HousePricePreprocessor:
    """Giữ thống kê train; fit_transform trả (X OOF, y đã align), giữ mọi dòng."""

    def __init__(
        self, n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits phải >= 2.")
        self.n_splits = n_splits
        self.random_state = random_state
        self._is_fitted = False

    def _validate_inputs(self, X: pd.DataFrame) -> None:
        require_columns(X, INPUT_COLUMNS)
        require_unique_index(X)
        if not np.isfinite(X["area"].to_numpy(dtype=float)).all():
            raise ValueError("area phải hữu hạn; hãy clean dữ liệu trước khi fit.")

    def _fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        self._is_fitted = False
        self._validate_inputs(X_train)
        if X_train.empty:
            raise ValueError("Không thể fit trên train rỗng.")
        y_train = align_target(y_train, X_train.index)
        groups_train = align_series(groups_train, X_train.index, "groups_train")
        self.numeric_medians_ = fit_numeric_imputer(X_train)
        filled_train = fill_numeric_missing(X_train, self.numeric_medians_)
        # Giữ key metadata cũ; không còn học bounds hoặc lọc phân vị.
        self.outlier_bounds_ = None
        retained = filled_train
        target = y_train.loc[retained.index]
        retained_groups = groups_train.loc[retained.index]
        if len(retained) < 2 or retained_groups.nunique() < 2:
            raise ValueError("Cần ít nhất hai group train để tạo OOF.")
        self.target_statistics_ = fit_target_statistics(retained, target)
        self.district_medians_ = self.target_statistics_["districts"]
        self.ward_medians_ = self.target_statistics_["ward_medians"]
        self.global_median_price_ = self.target_statistics_["global_median_price"]
        self.categories_ = {
            column: sorted(retained[column].dropna().unique().tolist())
            for column in CATEGORICAL_COLUMNS
        }
        self.feature_cols_ = [
            *NUMERIC_COLUMNS, *MISSING_INDICATOR_COLUMNS, *ENCODING_COLUMNS,
            *[
                f"{column}_{value}"
                for column, categories in self.categories_.items()
                for value in categories
            ],
        ]
        self.retained_index_ = retained.index.copy()
        self.fit_rows_before_filter_ = len(X_train)
        self.fit_rows_after_filter_ = len(retained)
        self._is_fitted = True
        return retained, target, retained_groups

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series,
    ) -> "HousePricePreprocessor":
        """Học state chỉ từ train; dùng fit_transform để tạo feature train OOF."""
        self._fit(X_train, y_train, groups_train)
        return self

    def _assemble_features(
        self, filled: pd.DataFrame, target_encoding: pd.DataFrame,
    ) -> pd.DataFrame:
        numeric = filled[[*NUMERIC_COLUMNS, *MISSING_INDICATOR_COLUMNS]].astype(float)
        one_hot = pd.DataFrame(
            {
                f"{column}_{category}": filled[column].eq(category).fillna(False).astype(int)
                for column, categories in self.categories_.items()
                for category in categories
            },
            index=filled.index,
        )
        features = pd.concat([numeric, target_encoding, one_hot], axis=1)
        return features.loc[:, self.feature_cols_]

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit train rồi tạo OOF, giữ mọi dòng; không encode bằng nhãn chính nó."""
        retained, target, groups = self._fit(X_train, y_train, groups_train)
        encoded = make_oof_target_encoding(
            retained, target, groups, self.n_splits, self.random_state,
        )
        return self._assemble_features(retained, encoded), target

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Chỉ dùng state train; giữ mọi dòng validation/test và không nhận y."""
        if not self._is_fitted:
            raise NotFittedError("Hãy fit preprocessor bằng train trước khi transform.")
        self._validate_inputs(X)
        filled = fill_numeric_missing(X, self.numeric_medians_)
        encoded = transform_target_encoding(X, self.target_statistics_)
        return self._assemble_features(filled, encoded)

    def to_metadata(self) -> dict:
        """Xuất state tại ranh giới lưu artifact; giữ các key mà giao diện sử dụng."""
        if not self._is_fitted:
            raise NotFittedError("Preprocessor chưa được fit.")
        wards_by_district = {}
        for district, ward in self.ward_medians_:
            wards_by_district.setdefault(district, []).append(ward)
        return deepcopy({
            **self.target_statistics_,
            "numeric_medians": self.numeric_medians_,
            "outlier_bounds": self.outlier_bounds_,
            "train_outlier_filter": "none",
            "categories": self.categories_,
            "feature_cols": self.feature_cols_,
            "wards_by_district": {
                district: sorted(wards) for district, wards in wards_by_district.items()
            },
            "fit_rows_before_filter": self.fit_rows_before_filter_,
            "fit_rows_after_filter": self.fit_rows_after_filter_,
            "price_unit": PRICE_UNIT,
        })
