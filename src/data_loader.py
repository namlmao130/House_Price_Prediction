"""Đọc đúng 10 shard, lọc Hà Nội và chỉ nạp cột cần dùng."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from config import (
    DATA_DIR, PARQUET_FILE_COUNT, PROVINCE_NAME, RANDOM_STATE, RAW_COLUMNS,
)

logger = logging.getLogger(__name__)


def parquet_files(data_dir: Path = DATA_DIR) -> tuple[Path, ...]:
    return tuple(
        Path(data_dir) / f"shard_{index:04d}.parquet"
        for index in range(PARQUET_FILE_COUNT)
    )


def load_raw_data(
    data_dir: Path = DATA_DIR,
    sample_size: int | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Lọc theo tỉnh khi đọc Parquet; có thể lấy mẫu trước khi chuyển sang pandas."""
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size phải là số nguyên dương.")
    files = parquet_files(data_dir)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy shard: {path}")
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    missing = set([*RAW_COLUMNS, "province_name"]) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Parquet thiếu cột: {', '.join(sorted(missing))}")
    province_filter = ds.field("province_name") == PROVINCE_NAME
    sampled = False
    if sample_size is None:
        table = dataset.to_table(columns=RAW_COLUMNS, filter=province_filter)
    else:
        row_count = dataset.count_rows(filter=province_filter)
        take_count = min(sample_size, row_count)
        sampled = take_count < row_count
        indices = np.sort(
            np.random.default_rng(random_state).choice(
                row_count, size=take_count, replace=False,
            )
        )
        table = dataset.take(
            indices, columns=RAW_COLUMNS, filter=province_filter,
        )
    raw_df = table.to_pandas()
    logger.info(
        "Đọc %s shard: %s dòng Hà Nội%s",
        len(files), f"{len(raw_df):,}", " (mẫu)" if sampled else "",
    )
    return raw_df

