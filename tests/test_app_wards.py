"""Kiểm tra lựa chọn phường và đầu vào dự đoán bằng dữ liệu nhỏ, xác định."""
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
import cleaning
import artifacts
import config
import data_loader


class RecordingModel:
    def predict(self, frame):
        self.last_input = frame.copy()
        return np.array([float(frame["ward_median_price"].iloc[0])])


class WardSelectionTests(unittest.TestCase):
    def test_ward_selection_unknown_and_district_change(self):
        model = RecordingModel()
        bundle = {
            "model": model,
            "feature_cols": ["area", "district_median_price", "ward_median_price"],
            "districts": {"A": 10., "B": 30.},
            "ward_medians": {("A", "W1"): 14., ("B", "W2"): 35.},
            "wards_by_district": {"A": ["W1"], "B": ["W2"]},
            "global_median_price": 20.,
            "best_model_name": "Test model",
            "best_model_mae": 2.,
        }
        frame = pd.DataFrame({
            "area": [30., 40., 50., 60.],
            "bedroom_count": [2., 3., 4., 5.],
            "bathroom_count": [1., 2., 3., 4.],
            "floor_count": [2., 3., 4., 5.],
            "frontage_width": [3., 4., 5., 6.],
            "price_ty": [8., 12., 25., 35.],
            "district_name": ["A", "A", "B", "B"],
            "ward_name": ["W1", "W1", "W2", "W2"],
            "property_type_name": ["Nhà"] * 4,
        })
        st.cache_resource.clear()
        st.cache_data.clear()
        with patch.object(artifacts, "load_artifacts", return_value=bundle), \
             patch.object(config, "MODEL_PATH", ROOT / "README.md"), \
             patch.object(config, "PREPROCESSOR_PATH", ROOT / "README.md"), \
             patch.object(data_loader, "load_raw_data", return_value=frame), \
             patch.object(cleaning, "clean_listings", side_effect=lambda data: data):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("2025–2026" in c.value for c in app.caption))
            self.assertEqual(app.selectbox[1].options, ["Không rõ", "W1"])
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(model.last_input["ward_median_price"].iloc[0], 10.)
            self.assertNotIn("month", model.last_input.columns)
            self.assertNotIn("year", model.last_input.columns)
            self.assertIn("Biên sai số tham khảo ± MAE", app.success[0].value)
            app.selectbox[1].select("W1").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(model.last_input["ward_median_price"].iloc[0], 14.)
            app.selectbox[0].select("B").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox[1].options, ["Không rõ", "W2"])
            self.assertEqual(app.selectbox[1].value, "Không rõ")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(model.last_input["ward_median_price"].iloc[0], 30.)
            app.selectbox[1].select("W2").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(model.last_input["ward_median_price"].iloc[0], 35.)
            with patch.object(model, "predict", return_value=np.array([-3.1])):
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.success)
                self.assertTrue(any("Chưa ước lượng được" in item.value for item in app.warning))
                self.assertTrue(any("Giải thích chỉ số" in item.value for item in app.subheader))
        st.cache_resource.clear()
        st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
