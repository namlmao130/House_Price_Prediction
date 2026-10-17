"""Kiểm tra đọc đầy đủ và lấy mẫu trực tiếp từ các shard Parquet."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import RAW_COLUMNS
from data_loader import load_raw_data


class DataLoaderTests(unittest.TestCase):
    def test_sample_is_filtered_bounded_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            for shard in range(10):
                frame = pd.DataFrame({
                    "province_name": ["Hà Nội"] * 4 + ["TP HCM"] * 2,
                    "published_at": ["2026-01-01"] * 6,
                    "price": [10e9] * 6,
                    "area": [50.] * 6,
                    "bedroom_count": [3.] * 6,
                    "bathroom_count": [2.] * 6,
                    "floor_count": [4.] * 6,
                    "frontage_width": [4.] * 6,
                    "property_type_name": ["Nhà"] * 6,
                    "district_name": [f"Quận {shard}"] * 6,
                    "ward_name": [f"Phường {row}" for row in range(6)],
                })
                pq.write_table(
                    pa.Table.from_pandas(frame, preserve_index=False),
                    data_dir / f"shard_{shard:04d}.parquet",
                )

            full = load_raw_data(data_dir)
            first = load_raw_data(data_dir, sample_size=15, random_state=7)
            second = load_raw_data(data_dir, sample_size=15, random_state=7)

            self.assertEqual(len(full), 40)
            self.assertEqual(len(first), 15)
            self.assertEqual(first.columns.tolist(), RAW_COLUMNS)
            assert_frame_equal(first, second)
            with self.assertRaisesRegex(ValueError, "số nguyên dương"):
                load_raw_data(data_dir, sample_size=0)


if __name__ == "__main__":
    unittest.main()
