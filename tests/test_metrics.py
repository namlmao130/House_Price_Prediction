"""Kiểm tra metric sai số tương đối, thang log và sai số theo nhóm."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import (
    MIN_SEGMENT_ROWS, evaluate_by_price_band, evaluate_by_segment,
    evaluate_regression,
)


class RegressionMetricTests(unittest.TestCase):
    def test_relative_error_metrics_have_known_values(self):
        actual = pd.Series([10., 10., 10., 10.])
        predicted = np.array([10., 11., 12., 15.])
        metrics = evaluate_regression(actual, predicted)
        self.assertAlmostEqual(metrics["mape"], 20.0)
        self.assertAlmostEqual(metrics["median_ape"], 15.0)
        self.assertAlmostEqual(metrics["hit_rate_10"], 0.5)
        self.assertAlmostEqual(metrics["hit_rate_20"], 0.75)
        self.assertAlmostEqual(metrics["bias"], 1.15)

    def test_median_ape_is_robust_to_one_extreme_error(self):
        metrics = evaluate_regression(
            pd.Series([10.] * 9 + [1.]), np.array([10.] * 9 + [100.]),
        )
        self.assertAlmostEqual(metrics["median_ape"], 0.0)
        self.assertGreater(metrics["mape"], 900.0)

    def test_perfect_prediction_gives_perfect_scores(self):
        actual = pd.Series([2., 5., 9., 30.])
        metrics = evaluate_regression(actual, actual.to_numpy())
        for key in ("r2", "r2_log", "hit_rate_10", "hit_rate_20"):
            self.assertAlmostEqual(metrics[key], 1.0)
        for key in ("mae", "rmse", "mape", "median_ape", "rmsle"):
            self.assertAlmostEqual(metrics[key], 0.0)
        self.assertAlmostEqual(metrics["bias"], 1.0)

    def test_log_metrics_use_log1p_scale(self):
        actual = pd.Series([1., 3., 7., 15.])
        predicted = np.array([2., 3., 5., 20.])
        expected = np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))
        self.assertAlmostEqual(evaluate_regression(actual, predicted)["rmsle"], expected)

    def test_invalid_inputs_still_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_regression(pd.Series([1., 0.]), np.array([1., 1.]))
        with self.assertRaises(ValueError):
            evaluate_regression(pd.Series([1., 2.]), np.array([1., np.nan]))
        with self.assertRaises(ValueError):
            evaluate_regression(pd.Series([1., 2.]), np.array([1.]))


class SegmentMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n = MIN_SEGMENT_ROWS * 40
        self.actual = pd.Series(rng.uniform(1, 40, self.n))
        self.predicted = self.actual.to_numpy() * rng.uniform(0.8, 1.2, self.n)

    def test_price_band_counts_cover_all_rows_and_keep_order(self):
        rows = evaluate_by_price_band(self.actual, self.predicted)
        self.assertEqual(sum(row["n"] for row in rows), self.n)
        medians = [row["median_price"] for row in rows]
        self.assertEqual(medians, sorted(medians))

    def test_small_segments_are_dropped(self):
        labels = pd.Series(["A"] * (self.n - 10) + ["B"] * 10)
        rows = evaluate_by_segment(self.actual, self.predicted, labels)
        self.assertEqual([row["segment"] for row in rows], ["A"])

    def test_missing_segment_label_becomes_unknown(self):
        labels = pd.Series([None] * self.n, dtype="string")
        rows = evaluate_by_segment(self.actual, self.predicted, labels)
        self.assertEqual(rows[0]["segment"], "Không rõ")
        self.assertEqual(rows[0]["n"], self.n)


if __name__ == "__main__":
    unittest.main()
