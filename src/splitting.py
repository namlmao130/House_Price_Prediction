"""Chia group trước, khử trùng từng tập sau; giữ index nguồn xuyên suốt."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import DEDUP_GROUP_COLUMNS, INPUT_COLUMNS, NUMERIC_COLUMNS
from config import RANDOM_STATE, TEST_SIZE
from validation import ensure_disjoint_groups, require_columns, require_unique_index


def build_group_key(frame: pd.DataFrame) -> pd.Series:
    """Fingerprint đặc trưng và giá; theo quy ước đồ án, khác giá là khác nhóm."""
    require_columns(frame, DEDUP_GROUP_COLUMNS)
    require_unique_index(frame)
    values = frame[DEDUP_GROUP_COLUMNS].copy()
    for column in [*NUMERIC_COLUMNS, "price_ty"]:
        values[column] = pd.to_numeric(values[column], errors="coerce").astype(float)
    for column in ["district_name", "ward_name", "property_type_name"]:
        values[column] = (
            values[column].astype("string").str.normalize("NFKC").str.casefold()
            .str.strip().str.replace(r"\s+", " ", regex=True).fillna("<missing>")
        )
    return pd.util.hash_pandas_object(values, index=False).rename("group_id")


def split_train_test(
    cleaned_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """GroupShuffleSplit trên dữ liệu chưa dedup; trả hai tập và group nguồn."""
    if cleaned_df.empty:
        raise ValueError("Không còn dữ liệu sau cleaning.")
    groups = build_group_key(cleaned_df)
    if groups.nunique() < 2:
        raise ValueError("Cần ít nhất hai group để chia train/test.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    train_pos, test_pos = next(splitter.split(cleaned_df, groups=groups))
    train_df = cleaned_df.iloc[train_pos].copy()
    test_df = cleaned_df.iloc[test_pos].copy()
    ensure_disjoint_groups(groups.loc[train_df.index], groups.loc[test_df.index])
    return train_df, test_df, groups


def build_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    require_columns(frame, [*INPUT_COLUMNS, "price_ty"])
    require_unique_index(frame)
    if frame.empty:
        raise ValueError("Không thể tạo X/y từ dữ liệu rỗng.")
    return frame[INPUT_COLUMNS].copy(), frame["price_ty"].copy()

