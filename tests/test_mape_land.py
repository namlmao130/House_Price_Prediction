"""Regression checks for percentage metrics and excluding land before splitting."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleaning import clean_listings
from evaluation import evaluate_regression
from preprocessing import HousePricePreprocessor
from splitting import build_group_key, build_xy


class MapeLandTests(unittest.TestCase):
    def test_mape_is_percent_on_original_prices(self):
        result = evaluate_regression(pd.Series([2., 10.]), np.array([1., 12.]))
        self.assertAlmostEqual(result["mape"], 35.)
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))

    def test_mape_does_not_clip_large_errors_or_drop_small_prices(self):
        result = evaluate_regression(pd.Series([0.001, 1.]), np.array([1., 1.]))
        self.assertAlmostEqual(result["mape"], 49950.)
        for prices in [[0., 1.], [-1., 1.]]:
            with self.assertRaisesRegex(ValueError, "lớn hơn 0"):
                evaluate_regression(pd.Series(prices), np.array([1., 1.]))

    def test_clean_excludes_normalized_land_and_keeps_other_types(self):
        raw = pd.DataFrame({
            "price": 5e9, "area": 50., "bedroom_count": 3.,
            "bathroom_count": 2., "floor_count": 4., "frontage_width": 4.,
            "district_name": "A", "ward_name": "W",
            "property_type_name": ["Đất", " ĐẤT ", "đất", "Nhà", "Căn hộ", None],
        })
        before = raw.copy(deep=True)
        cleaned = clean_listings(raw)
        self.assertEqual(cleaned.index.tolist(), [3, 4])
        assert_frame_equal(raw, before)
        X, y = build_xy(cleaned)
        prep = HousePricePreprocessor().fit(X, y, build_group_key(cleaned))
        self.assertEqual(prep.categories_["property_type_name"], ["Căn hộ", "Nhà"])
        self.assertNotIn("property_type_name_Đất", prep.feature_cols_)


if __name__ == "__main__":
    unittest.main()
