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
    "test_r2": 0.7,
    "cv_r2_mean": 0.65,
    "cv_r2_std": 0.04,
    "cv_mae": 5.0,
    "test_mae": 5.0,
    "cv_rmse": 20.0,
    "test_rmse": 20.0,
    "cv_mape": 25.0,
    "test_mape": 24.0,
}
EXTRA_METRICS = {
    "test_median_ape": 16.5,
    "test_hit_rate_10": 0.31,
    "test_hit_rate_20": 0.57,
    "test_r2_log": 0.81,
    "test_bias": 0.97,
}


class Model:
    def predict(self, frame):
        return np.log1p([10.0])


class Preprocessor:
    def transform(self, frame):
        return frame[["area"]].copy()


def make_bundle(with_extra: bool) -> dict:
    metrics = {**BASE_METRICS, **(EXTRA_METRICS if with_extra else {})}
    evaluation = {
        "train_rows_before_filter": 64,
        "test_rows": 16,
        "unique_groups": 80,
        "train_test_group_overlap": 0,
        "baseline_train_median_test_mae": 10.0,
    }
    if with_extra:
        row = {
            "n": 500,
            "median_price": 8.0,
            "mae": 3.0,
            "median_ape": 15.0,
            "hit_rate_20": 0.6,
            "bias": 0.95,
        }
        evaluation["segment_metrics_model"] = "Test model"
        evaluation["segment_metrics"] = {
            "price_band": [{"segment": "<3 tỷ", **row}],
            "property_type": [{"segment": "Nhà", **row}],
            "district": [],
        }
    return {
        "model": Model(),
        "preprocessor": Preprocessor(),
        "target_transform": "log1p",
        "districts": {"A": 10.0},
        "wards_by_district": {"A": ["W1"]},
        "best_model_name": "Test model",
        "best_model_mae": 5.0,
        "cv_summary": {
            "Weak model": {**metrics, "cv_r2_mean": 0.40},
            "Test model": metrics,
        },
        "artifact_version": {
            "version_id": "v",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "data": {
                "raw_rows": 10,
                "eligible_rows": 9,
                "deduplicated_rows": 8,
                "test_rows": 2,
            },
            "evaluation": evaluation,
        },
    }


def run_app(bundle: dict) -> AppTest:
    frame = pd.DataFrame(
        {
            "area": [30.0, 40.0],
            "bedroom_count": [2.0, 3.0],
            "bathroom_count": [1.0, 2.0],
            "floor_count": [2.0, 3.0],
            "frontage_width": [3.0, 4.0],
            "price_ty": [5.0, 6.0],
            "district_name": ["A", "A"],
            "ward_name": ["W1", "W1"],
            "property_type_name": ["Nhà", "Nhà"],
        }
    )
    st.cache_resource.clear()
    st.cache_data.clear()
    with patch.object(artifacts, "load_artifacts", return_value=bundle), patch.object(
        config, "MODEL_PATH", ROOT / "README.md"
    ), patch.object(config, "PREPROCESSOR_PATH", ROOT / "README.md"), patch.object(
        data_loader, "parquet_files", return_value=()
    ), patch.object(
        data_loader, "load_raw_data", return_value=frame
    ), patch.object(
        cleaning, "clean_listings", side_effect=lambda data: data
    ):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()


class EvaluationTabTests(unittest.TestCase):
    def test_summary_cards_show_only_headline_metrics(self):
        app = run_app(make_bundle(with_extra=True))
        self.assertFalse(app.exception)
        values = {item.label: item.value for item in app.metric}
        self.assertEqual(values["Test R²"], "0.700")
        self.assertEqual(values["Test MAE"], "5.00 tỷ")
        self.assertEqual(values["Sai lệch trung vị"], "16.50%")
        self.assertEqual(values["Dự đoán lệch ≤ 20%"], "57.00%")
        for dropped in (
            "Dự đoán lệch ≤ 10%",
            "R² trên thang log",
            "Test MAPE",
            "Sai lệch trung vị (Median APE)",
        ):
            self.assertNotIn(dropped, values)

    def test_main_table_is_compact_sorted_and_marks_selected_model(self):
        app = run_app(make_bundle(with_extra=True))
        tables = [item.value for item in app.dataframe]
        main = next(t for t in tables if "CV R² (± std)" in t.columns)
        self.assertEqual(
            list(main.columns),
            [
                "Mô hình",
                "CV R² (± std)",
                "Test R²",
                "Test MAE (tỷ)",
                "Median APE",
                "Lệch ≤ 20%",
            ],
        )
        self.assertEqual(list(main["Mô hình"]), ["✅ Test model", "Weak model"])
        self.assertEqual(main.iloc[0]["CV R² (± std)"], "0.650 ± 0.040")

    def test_full_metrics_are_in_expander_table(self):
        app = run_app(make_bundle(with_extra=True))
        self.assertTrue(
            any("Xem toàn bộ chỉ số" in item.label for item in app.expander)
        )
        tables = [item.value for item in app.dataframe]
        full = next(t for t in tables if "Test R² log" in t.columns)
        self.assertIn("Test Median APE (%)", full.columns)
        self.assertIn("Test lệch ≤ 10%", full.columns)
        self.assertEqual(full.iloc[0]["Test lệch ≤ 10%"], "31.00%")

    def test_segment_tables_are_shown(self):
        app = run_app(make_bundle(with_extra=True))
        self.assertTrue(any("Sai số theo nhóm" in item.value for item in app.subheader))
        tables = [item.value for item in app.dataframe]
        self.assertTrue(any("Khoảng giá" in t.columns for t in tables))
        self.assertTrue(any("Loại BĐS" in t.columns for t in tables))

    def test_old_artifact_without_new_metrics_still_renders(self):
        app = run_app(make_bundle(with_extra=False))
        self.assertFalse(app.exception)
        self.assertTrue(any("train_model.py" in item.value for item in app.info))
        main = next(
            item.value
            for item in app.dataframe
            if "CV R² (± std)" in item.value.columns
        )
        self.assertIn("Test MAPE", main.columns)
        self.assertNotIn("Median APE", main.columns)


if __name__ == "__main__":
    unittest.main()
