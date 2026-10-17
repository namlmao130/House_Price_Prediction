"""Quy tắc cố định theo từng dòng; không fit trên toàn dataset."""

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_COLUMNS, INPUT_COLUMNS, NUMERIC_COLUMNS, PRICE_SCALE,
    MAX_PRICE_TY, MIN_AREA, MAX_AREA,
    MIN_BEDROOM_COUNT, MAX_BEDROOM_COUNT,
    MIN_BATHROOM_COUNT, MAX_BATHROOM_COUNT,
    MIN_FLOOR_COUNT, MAX_FLOOR_COUNT,
    MIN_FRONTAGE_WIDTH, MAX_FRONTAGE_WIDTH,
    MIN_PRICE_PER_M2, MAX_PRICE_PER_M2,
)
from validation import require_columns, require_unique_index


def data_entry_error_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn cờ các lỗi nhập liệu rõ ràng bằng ngưỡng cố định theo từng dòng."""
    require_columns(
        df,
        [
            "price", "area", "bedroom_count", "bathroom_count",
            "floor_count", "frontage_width",
        ],
    )
    price = pd.to_numeric(df["price"], errors="coerce") / PRICE_SCALE
    area = pd.to_numeric(df["area"], errors="coerce")
    floors = pd.to_numeric(df["floor_count"], errors="coerce")
    bedrooms = pd.to_numeric(df["bedroom_count"], errors="coerce")
    bathrooms = pd.to_numeric(df["bathroom_count"], errors="coerce")
    frontage = pd.to_numeric(df["frontage_width"], errors="coerce")
    return pd.DataFrame(
        {
            "price_non_positive": price.le(0),
            "price_above_10000_ty": price.gt(MAX_PRICE_TY),
            "area_below_min": area.lt(MIN_AREA),
            "area_above_max": area.gt(MAX_AREA),
            "bedroom_count_outside_bounds": (
                bedrooms.lt(MIN_BEDROOM_COUNT) | bedrooms.gt(MAX_BEDROOM_COUNT)
            ),
            "bathroom_count_outside_bounds": (
                bathrooms.lt(MIN_BATHROOM_COUNT) | bathrooms.gt(MAX_BATHROOM_COUNT)
            ),
            "floor_count_outside_bounds": (
                floors.lt(MIN_FLOOR_COUNT) | floors.gt(MAX_FLOOR_COUNT)
            ),
            "frontage_width_outside_bounds": (
                frontage.lt(MIN_FRONTAGE_WIDTH) | frontage.gt(MAX_FRONTAGE_WIDTH)
            ),
            "frontage_width_above_area": frontage.gt(area),
            "price_per_m2_below_min": (
                area.gt(0) & price.div(area).lt(MIN_PRICE_PER_M2)
            ),
            "price_per_m2_above_max": (
                area.gt(0) & price.div(area).gt(MAX_PRICE_PER_M2)
            ),
        },
        index=df.index,
    ).fillna(False)


def clean_listings(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa và lọc theo dòng; giữ NaN số tùy chọn cho imputation sau split."""
    require_columns(raw_df, ["price", *INPUT_COLUMNS])
    require_unique_index(raw_df)
    cleaned = raw_df.copy()
    numeric_columns = ["price", *NUMERIC_COLUMNS]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[numeric_columns] = cleaned[numeric_columns].replace(
        [np.inf, -np.inf], np.nan,
    )
    for column in [*CATEGORICAL_COLUMNS, "ward_name"]:
        cleaned[column] = (
            cleaned[column].astype("string").str.normalize("NFKC").str.strip()
            .str.replace(r"\s+", " ", regex=True).replace("", pd.NA)
        )
    required = ["price", "area", *CATEGORICAL_COLUMNS]
    valid = (
        cleaned[required].notna().all(axis=1)
        & ~cleaned["property_type_name"].str.casefold().eq("đất").fillna(False)
        & cleaned["price"].gt(0)
        & cleaned["area"].gt(0)
        & ~cleaned[NUMERIC_COLUMNS].lt(0).any(axis=1)
        & ~data_entry_error_flags(cleaned).any(axis=1)
    )
    cleaned = cleaned.loc[valid].copy()
    cleaned["price_ty"] = cleaned["price"] / PRICE_SCALE
    cleaned["price_per_m2"] = cleaned["price_ty"] * 1000 / cleaned["area"]
    return cleaned



def dataset_summary(raw: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    """Ngày hỗn hợp có/không có phần giây lẻ đều được đọc; lỗi không loại mẫu."""
    require_columns(raw, ["published_at"])
    dates = pd.to_datetime(raw["published_at"], format="mixed", errors="coerce")
    entry_flags = data_entry_error_flags(raw)
    return {
        "raw_rows": len(raw),
        "eligible_rows": len(eligible),
        "rows_removed_by_fixed_checks": len(raw) - len(eligible),
        "published_at_min": dates.min().isoformat() if dates.notna().any() else None,
        "published_at_max": dates.max().isoformat() if dates.notna().any() else None,
        "unparsed_date_rows": int(dates.isna().sum()),
        "date_rows_removed": 0,
        "time_features": False,
        "data_entry_error_rows_removed": int(entry_flags.any(axis=1).sum()),
        "data_entry_error_counts": {
            column: int(entry_flags[column].sum()) for column in entry_flags
        },
        "data_entry_error_rules": {
            "max_price_ty": MAX_PRICE_TY,
            "min_area": MIN_AREA,
            "max_area": MAX_AREA,
            "min_bedroom_count": MIN_BEDROOM_COUNT,
            "max_bedroom_count": MAX_BEDROOM_COUNT,
            "min_bathroom_count": MIN_BATHROOM_COUNT,
            "max_bathroom_count": MAX_BATHROOM_COUNT,
            "min_floor_count": MIN_FLOOR_COUNT,
            "max_floor_count": MAX_FLOOR_COUNT,
            "min_frontage_width": MIN_FRONTAGE_WIDTH,
            "max_frontage_width": MAX_FRONTAGE_WIDTH,
            "min_price_per_m2": MIN_PRICE_PER_M2,
            "max_price_per_m2": MAX_PRICE_PER_M2,
        },
    }


def deduplicate(frame: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Giữ dòng nguồn đầu tiên cho mỗi group, không đổi thứ tự hoặc index."""
    from splitting import build_group_key

    require_unique_index(frame)
    if groups is None:
        groups = build_group_key(frame)
    elif not groups.index.equals(frame.index) or groups.isna().any():
        raise ValueError("groups phải khớp index của dữ liệu cần khử trùng.")
    return frame.loc[~groups.duplicated(keep="first")].copy()


