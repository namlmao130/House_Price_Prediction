"""Tab đánh giá hiển thị metric mới và vẫn tương thích artifact cũ."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import artifacts
import cleaning
import config
import data_loader

BASE_METRICS = {
    "test_r2": .7, "cv_r2_mean": .65, "cv_r2_std": .04,
    "cv_mae": 5., "test_mae": 5., "cv_rmse": 20., "test_rmse": 20.,
    "cv_mape": 25., "test_mape": 24.,
}
EXTRA_METRICS = {
    "test_median_ape": 16.5, "test_hit_rate_10": .31,
    "test_hit_rate_20": .57, "test_r2_log": .81, "test_bias": .97,
}


class Model:
    def predict(self, frame):
        return np.log1p([10.])


class Preprocessor:
    def transform(self, frame):
        return frame[["area"]].copy()


def make_bundle(with_extra: bool) -> dict:
    metrics = {**BASE_METRICS, **(EXTRA_METRICS if with_extra else {})}
    evaluation = {
        "train_rows_before_filter": 64, "test_rows": 16, "unique_groups": 80,
        "train_test_group_overlap": 0, "baseline_train_median_test_mae": 10.,
    }
    if with_extra:
        row = {
            "n": 500, "median_price": 8., "mae": 3., "median_ape": 15.,
            "hit_rate_20": .6, "bias": .95,
        }
        evaluation["segment_metrics_model"] = "Test model"
        evaluation["segment_metrics"] = {
            "price_band": [{"segment": "<3 tỷ", **row}],
            "property_type": [{"segment": "Nhà", **row}],
            "district": [],
        }
    return {
        "model": Model(), "preprocessor": Preprocessor(),
        "target_transform": "log1p", "districts": {"A": 10.},
        "wards_by_district": {"A": ["W1"]},
        "best_model_name": "Test model", "best_model_mae": 5.,
        "cv_summary": {"Test model": metrics},
        "artifact_version": {
            "version_id": "v", "created_at_utc": "2026-01-01T00:00:00+00:00",
            "data": {"raw_rows": 10, "eligible_rows": 9,
                     "deduplicated_rows": 8, "test_rows": 2},
            "evaluation": evaluation,
        },
    }


def run_app(bundle: dict) -> AppTest:
    frame = pd.DataFrame({
        "area": [30., 40.], "bedroom_count": [2., 3.],
        "bathroom_count": [1., 2.], "floor_count": [2., 3.],
        "frontage_width": [3., 4.], "price_ty": [5., 6.],
        "district_name": ["A", "A"], "ward_name": ["W1", "W1"],
        "property_type_name": ["Nhà", "Nhà"],
    })
    st.cache_resource.clear()
    st.cache_data.clear()
    with patch.object(artifacts, "load_artifacts", return_value=bundle), \
         patch.object(config, "MODEL_PATH", ROOT / "README.md"), \
         patch.object(config, "PREPROCESSOR_PATH", ROOT / "README.md"), \
         patch.object(data_loader, "parquet_files", return_value=()), \
         patch.object(data_loader, "load_raw_data", return_value=frame), \
         patch.object(cleaning, "clean_listings", side_effect=lambda data: data):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


class EvaluationTabTests(unittest.TestCase):
    def test_new_metrics_and_segment_tables_are_shown(self):
        app = run_app(make_bundle(with_extra=True))
        self.assertFalse(app.exception)
        values = {item.label: item.value for item in app.metric}
        self.assertEqual(values["Dự đoán lệch ≤ 20%"], "57.0%")
        self.assertEqual(values["Dự đoán lệch ≤ 10%"], "31.0%")
        self.assertEqual(values["Sai lệch trung vị (Median APE)"], "16.50%")
        self.assertEqual(values["R² trên thang log"], "0.810")
        self.assertTrue(any("Sai số theo nhóm" in item.value for item in app.subheader))

    def test_old_artifact_without_new_metrics_still_renders(self):
        app = run_app(make_bundle(with_extra=False))
        self.assertFalse(app.exception)
        self.assertTrue(any("train_model.py" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
