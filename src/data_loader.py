"""Đọc đúng 10 shard, lọc Hà Nội và chỉ nạp cột cần dùng."""

import logging
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from config import DATA_DIR, PARQUET_FILE_COUNT, PROVINCE_NAME, RAW_COLUMNS

logger = logging.getLogger(__name__)


def parquet_files(data_dir: Path = DATA_DIR) -> tuple[Path, ...]:
    return tuple(
        Path(data_dir) / f"shard_{index:04d}.parquet"
        for index in range(PARQUET_FILE_COUNT)
    )


def load_raw_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Lọc theo tỉnh khi đọc Parquet; chưa học bất kỳ thống kê nào."""
    files = parquet_files(data_dir)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy shard: {path}")
    dataset = ds.dataset([str(path) for path in files], format="parquet")
    missing = set([*RAW_COLUMNS, "province_name"]) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Parquet thiếu cột: {', '.join(sorted(missing))}")
    table = dataset.to_table(
        columns=RAW_COLUMNS,
        filter=ds.field("province_name") == PROVINCE_NAME,
    )
    raw_df = table.to_pandas()
    logger.info("Đọc %s shard: %s dòng Hà Nội", len(files), f"{len(raw_df):,}")
    return raw_df

