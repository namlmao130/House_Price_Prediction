"""Verify log targets while preprocessing and metrics remain on original prices."""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from artifacts import RunContext, load_artifacts, save_artifacts
from config import N_SPLITS, RANDOM_STATE
from evaluation import evaluate_regression
from preprocessing import HousePricePreprocessor
from training import get_models, refit_final_model, train_and_evaluate


class LogTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = pd.DataFrame({
            "area": rng.uniform(30, 150, 100), "bedroom_count": 3.,
            "bathroom_count": 2., "floor_count": 4., "frontage_width": 4.,
            "district_name": ["A", "B"] * 50, "ward_name": "W",
            "property_type_name": "Nhà",
        })
        cls.y = pd.Series(np.expm1(1 + cls.X.area / 100), index=cls.X.index)
        cls.groups = pd.Series(np.arange(100), index=cls.X.index)
        cls.models = get_models()
        for model in cls.models.values():
            if "n_estimators" in model.get_params():
                model.set_params(n_estimators=5)
        with patch("training.get_models", return_value=cls.models):
            cls.result = train_and_evaluate(
                cls.X.iloc[:80], cls.y.iloc[:80], cls.X.iloc[80:], cls.y.iloc[80:],
                cls.groups.iloc[:80],
            )

    def test_all_models_fit_log_and_test_metrics_use_original_prices(self):
        prep = HousePricePreprocessor()
        features, y_fit = prep.fit_transform(
            self.X.iloc[:80], self.y.iloc[:80], self.groups.iloc[:80],
        )
        test = prep.transform(self.X.iloc[80:])
        for name, estimator in self.models.items():
            expected = np.expm1(clone(estimator).fit(features, np.log1p(y_fit)).predict(test))
            np.testing.assert_allclose(self.result.predictions[name], expected)
            for metric, value in evaluate_regression(self.y.iloc[80:], expected).items():
                self.assertAlmostEqual(self.result.metrics[name][f"test_{metric}"], value)
        self.assertGreater(prep.global_median_price_, float(np.log1p(y_fit).median()))

    def test_cv_metrics_on_original_scale(self):
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        train, valid = next(splitter.split(self.X.iloc[:80], groups=self.groups.iloc[:80]))
        prep = HousePricePreprocessor()
        features, y_fit = prep.fit_transform(
            self.X.iloc[train], self.y.iloc[train], self.groups.iloc[train],
        )
        for name, estimator in self.models.items():
            predictions = np.expm1(clone(estimator).fit(features, np.log1p(y_fit)).predict(
                prep.transform(self.X.iloc[valid]),
            ))
            expected = evaluate_regression(self.y.iloc[valid], predictions)
            for metric, value in expected.items():
                self.assertAlmostEqual(self.result.metrics[name]["fold_scores"][0][metric], value)

    def test_refit_all_models_uses_log_target(self):
        for name, estimator in self.models.items():
            with patch("training.get_models", return_value=self.models):
                model, prep = refit_final_model(name, self.X, self.y, self.groups)
            features, y_fit = HousePricePreprocessor().fit_transform(self.X, self.y, self.groups)
            reference = clone(estimator).fit(features, np.log1p(y_fit))
            np.testing.assert_allclose(model.predict(features), reference.predict(features))

    def test_saved_reports_and_bundles_record_log_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "results/runs/test"
            run_dir.mkdir(parents=True)
            run = RunContext(root, root / "data", run_dir, "data-hash", "code-hash", "v1")
            with patch("artifacts.dataset_sha256", return_value="data-hash"), \
                 patch("artifacts.source_sha256", return_value="code-hash"), \
                 patch("artifacts.source_files", return_value=[]), \
                 patch("artifacts.parquet_files", return_value=[]):
                save_artifacts(
                    run, {}, self.result, self.result.model, self.result.preprocessor,
                    self.groups, self.X.iloc[80:].index,
                )
            for filename in ["results/metrics.json", "results/runs/test/results.json"]:
                self.assertEqual(json.loads((root / filename).read_text(
                    encoding="utf-8"))["target_transform"], "log1p")
            for path in [root / "artifacts/model.pkl", root / "artifacts/preprocessor.pkl",
                         run_dir / "model.pkl", run_dir / "evaluation_model.pkl"]:
                with path.open("rb") as source:
                    self.assertEqual(pickle.load(source)["target_transform"], "log1p")
            bundle = load_artifacts(root / "artifacts")
            predicted = np.expm1(bundle["model"].predict(
                bundle["preprocessor"].transform(self.X.iloc[80:]),
            ))
            np.testing.assert_allclose(predicted, self.result.predictions[self.result.best_name])


if __name__ == "__main__":
    unittest.main()
