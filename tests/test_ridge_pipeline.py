"""Ridge chạy trên đặc trưng log1p và giữ nguyên contract feature."""

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import HousePricePreprocessor
from training import LOG_FEATURE_COLUMNS, get_feature_importance, get_models


def make_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    area = np.clip(rng.lognormal(np.log(60), 0.8, n), 10, 2500)
    X = pd.DataFrame({
        "area": area, "bedroom_count": rng.integers(1, 6, n).astype(float),
        "bathroom_count": rng.integers(1, 5, n).astype(float),
        "floor_count": rng.integers(1, 8, n).astype(float),
        "frontage_width": rng.uniform(2, 8, n),
        "district_name": rng.choice(["A", "B"], n), "ward_name": "W",
        "property_type_name": rng.choice(["Nhà", "Căn hộ chung cư"], n),
    })
    log_price = 1 + 0.9 * np.log(area / 60) + 0.08 * X["floor_count"]
    y = pd.Series(np.exp(log_price + rng.normal(0, .3, n)), index=X.index)
    return X, y, pd.Series(np.arange(n), index=X.index)


class RidgePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y, groups = make_data()
        cls.preprocessor = HousePricePreprocessor()
        cls.X_fit, y_fit = cls.preprocessor.fit_transform(X, y, groups)
        cls.model = get_models()["Ridge Regression"].fit(cls.X_fit, np.log1p(y_fit))
        cls.y_fit = y_fit

    def test_extreme_area_stays_finite(self):
        X, _, _ = make_data(n=50, seed=1)
        X.loc[X.index[:3], "area"] = [1000., 2000., 2500.]
        prediction = np.expm1(self.model.predict(self.preprocessor.transform(X)))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertLess(prediction.max(), 10 * self.y_fit.max())

    def test_input_frame_is_not_modified(self):
        X, _, _ = make_data(n=20, seed=2)
        encoded = self.preprocessor.transform(X)
        before = encoded.copy()
        self.model.predict(encoded)
        pd.testing.assert_frame_equal(encoded, before)

    def test_importance_labels_match_feature_columns(self):
        importance = get_feature_importance(self.model, list(self.X_fit.columns))
        self.assertEqual(set(importance.index), set(self.X_fit.columns))
        self.assertTrue(set(LOG_FEATURE_COLUMNS) <= set(self.X_fit.columns))

    def test_model_is_picklable(self):
        restored = pickle.loads(pickle.dumps(self.model))
        np.testing.assert_allclose(
            restored.predict(self.X_fit.head(5)), self.model.predict(self.X_fit.head(5)),
        )


if __name__ == "__main__":
    unittest.main()
