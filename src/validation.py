"""Kiểm tra schema và index trước khi fit; không sửa dữ liệu đầu vào."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")
    if not frame.columns.is_unique:
        raise ValueError("Tên cột phải duy nhất.")


def require_unique_index(frame: pd.DataFrame | pd.Series) -> None:
    if not frame.index.is_unique:
        raise ValueError("Index phải duy nhất để align X, y và groups.")


def align_series(
    values: pd.Series, index: pd.Index, name: str,
) -> pd.Series:
    require_unique_index(values)
    if not index.isin(values.index).all():
        raise ValueError(f"{name} thiếu index tương ứng với X.")
    aligned = values.loc[index]
    if aligned.isna().any():
        raise ValueError(f"{name} có giá trị thiếu.")
    return aligned


def align_target(y: pd.Series, index: pd.Index) -> pd.Series:
    target = align_series(y, index, "y")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("y phải có giá trị hữu hạn.")
    return target


def ensure_disjoint_groups(train: pd.Series, test: pd.Series) -> None:
    if set(train) & set(test):
        raise RuntimeError("Group train và validation/test bị overlap.")

