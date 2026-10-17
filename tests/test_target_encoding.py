"""Regression checks for train-only preprocessing and price-aware groups."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import INPUT_COLUMNS, MISSING_INDICATOR_COLUMNS
from cleaning import clean_listings as clean, dataset_summary, data_entry_error_flags
from cleaning import deduplicate
from splitting import build_group_key, split_train_test
from preprocessing import (
    HousePricePreprocessor, fill_numeric_missing, fit_numeric_imputer,
)
from training import cross_validate, train_and_evaluate


def fit_preprocessor(X, y, groups=None):
    if groups is None:
        groups = pd.Series(np.arange(len(X)), index=X.index)
    preprocessor = HousePricePreprocessor()
    features, target = preprocessor.fit_transform(X, y, groups)
    return features, target, preprocessor


def prepare_cv_fold(X, y, train_pos, validation_pos):
    features, target, preprocessor = fit_preprocessor(
        X.iloc[train_pos], y.iloc[train_pos]
    )
    return features, target, preprocessor.transform(X.iloc[validation_pos]), preprocessor



def example_rows(count=100):
    return pd.DataFrame({
        "area": np.linspace(40, 100, count), "bedroom_count": 3.,
        "bathroom_count": 2., "floor_count": 3., "frontage_width": 4.,
        "district_name": "A", "ward_name": "W1", "property_type_name": "Nhà",
    }, index=pd.Index(np.arange(count) * 3 + 10))


class PreprocessingTests(unittest.TestCase):
    def test_optional_numeric_missing_is_fit_only_on_train(self):
        X = example_rows(10)
        X.area = 50.
        y = pd.Series(10., index=X.index)
        optional = [
            "bedroom_count", "bathroom_count", "floor_count", "frontage_width"
        ]
        expected = {
            "bedroom_count": 3., "bathroom_count": 2.,
            "floor_count": 3., "frontage_width": 4.,
        }
        for row, column in enumerate(optional):
            X.loc[X.index[row], column] = np.nan
            X.loc[X.index[8], column] = np.nan
            X.loc[X.index[9], column] = 9999.

        train, _, valid, meta = prepare_cv_fold(
            X, y, np.arange(8), np.arange(8, 10)
        )
        self.assertEqual(meta.to_metadata()["numeric_medians"], expected)
        self.assertTrue(set(MISSING_INDICATOR_COLUMNS).issubset(meta.to_metadata()["feature_cols"]))
        for row, column in enumerate(optional):
            indicator = f"{column}_missing"
            self.assertEqual(train.loc[X.index[row], column], expected[column])
            self.assertEqual(train.loc[X.index[row], indicator], 1)
            self.assertEqual(valid.loc[X.index[8], column], expected[column])
            self.assertEqual(valid.loc[X.index[8], indicator], 1)
            self.assertEqual(valid.loc[X.index[9], indicator], 0)

        medians = fit_numeric_imputer(X.iloc[:8])
        filled = fill_numeric_missing(X.iloc[8:], medians)
        self.assertEqual(medians, expected)
        self.assertEqual(filled.loc[X.index[8], "bedroom_count"], 3.)

    def test_clean_keeps_optional_missing_but_drops_required_missing(self):
        raw = example_rows(8)
        raw["price"] = 10e9
        optional = [
            "bedroom_count", "bathroom_count", "floor_count", "frontage_width"
        ]
        for row, column in enumerate(optional):
            raw.loc[raw.index[row], column] = np.nan
        raw.loc[raw.index[4], "price"] = np.nan
        raw.loc[raw.index[5], "area"] = np.nan
        raw.loc[raw.index[6], "district_name"] = pd.NA
        raw.loc[raw.index[7], "property_type_name"] = pd.NA
        eligible = clean(raw)
        self.assertEqual(eligible.index.tolist(), raw.index[:4].tolist())

    def test_validation_labels_cannot_change_any_fitted_preprocessing(self):
        X = example_rows()
        y = pd.Series(np.linspace(2, 40, len(X)), index=X.index)
        before = prepare_cv_fold(X, y, np.arange(80), np.arange(80, 100))
        changed = y.copy()
        changed.iloc[80:] = 1e12
        after = prepare_cv_fold(X, changed, np.arange(80), np.arange(80, 100))
        assert_frame_equal(before[0], after[0])
        assert_series_equal(before[1], after[1])
        assert_frame_equal(before[2], after[2])
        self.assertEqual(before[3].to_metadata(), after[3].to_metadata())

    def test_preprocessing_keeps_extreme_train_and_validation_rows(self):
        X = example_rows()
        y = pd.Series(np.linspace(2, 40, len(X)), index=X.index)
        X.loc[X.index[-1], "area"] = 1e9
        y.iloc[-1] = 1e12
        X.loc[X.index[0], "area"] = 1e8
        y.iloc[0] = 1e10
        train, target, valid, meta = prepare_cv_fold(X, y, np.arange(80), np.arange(80, 100))
        self.assertEqual(len(train), 80)
        self.assertEqual(train.index.tolist(), X.index[:80].tolist())
        assert_series_equal(target, y.iloc[:80])
        self.assertEqual(train.iloc[0].area, 1e8)
        self.assertEqual(len(valid), 20)
        self.assertEqual(valid.index.tolist(), X.index[80:].tolist())
        self.assertEqual(valid.iloc[-1].area, 1e9)
        self.assertIsNone(meta.to_metadata()["outlier_bounds"])
        self.assertEqual(meta.to_metadata()["train_outlier_filter"], "none")
        self.assertEqual(meta.to_metadata()["fit_rows_after_filter"], len(target))

    def test_ward_median_is_district_specific_and_missing_falls_back(self):
        X = example_rows(6)
        X.area = 50.
        X["district_name"] = ["A", "A", "A", "A", "B", "B"]
        X["ward_name"] = "Shared ward"
        y = pd.Series([2., 2., 6., 6., 10., 10.], index=X.index)
        _, _, meta = fit_preprocessor(X, y)
        encoded_full = meta.transform(X)
        np.testing.assert_allclose(encoded_full.ward_median_price, [4., 4., 4., 4., 10., 10.])
        unseen = X.iloc[:3].copy()
        unseen["district_name"] = ["A", "B", "C"]
        unseen["ward_name"] = [None, "Shared ward", "New ward"]
        valid = meta.transform(unseen)
        np.testing.assert_allclose(valid.ward_median_price, [4., 10., 6.])

    def test_category_vocabulary_only_uses_fitted_train(self):
        X = example_rows()
        y = pd.Series(np.linspace(2, 40, len(X)), index=X.index)
        X.loc[X.index[80:], "district_name"] = "TEST_ONLY"
        train, _, valid, meta = prepare_cv_fold(X, y, np.arange(80), np.arange(80, 100))
        self.assertNotIn("district_name_TEST_ONLY", train.columns)
        self.assertNotIn("TEST_ONLY", meta.to_metadata()["categories"]["district_name"])
        self.assertTrue(valid["district_name_A"].eq(0).all())

    def test_price_defines_group_but_date_does_not(self):
        X = example_rows(20)
        X.iloc[1] = X.iloc[0]
        X.loc[X.index[1], "district_name"] = "  a  "
        X.loc[X.index[1], "ward_name"] = "w1"
        X["price_ty"] = np.arange(20, dtype=float)
        X.loc[X.index[1], "price_ty"] = X.loc[X.index[0], "price_ty"]
        X["published_at"] = ["2025-06-01", "2026-03-01"] * 10
        groups = build_group_key(X)
        self.assertEqual(groups.iloc[0], groups.iloc[1])

        changed_price = X.copy()
        changed_price.loc[X.index[1], "price_ty"] = 999.
        changed_groups = build_group_key(changed_price)
        self.assertNotEqual(changed_groups.iloc[0], changed_groups.iloc[1])

        changed = X.copy()
        changed["published_at"] = "unknown"
        assert_series_equal(groups, build_group_key(changed))
        tr, te, split_groups = split_train_test(X)
        self.assertFalse(set(split_groups.loc[tr.index]) & set(split_groups.loc[te.index]))

    def test_deduplicate_keeps_one_row_per_key_and_keeps_different_price(self):
        X = example_rows(6)
        X.iloc[1] = X.iloc[0]
        X["price_ty"] = [5., 5., 7., 8., 9., 10.]
        X["published_at"] = ["2025-01-01", "2026-01-01", None, None, None, None]
        result = deduplicate(X)
        self.assertEqual(len(result), 5)
        self.assertIn(X.index[0], result.index)
        self.assertNotIn(X.index[1], result.index)
        changed = X.copy()
        changed.loc[X.index[1], "price_ty"] = 6.
        self.assertEqual(len(deduplicate(changed)), 6)

    def test_training_target_encoding_is_out_of_fold(self):
        X = example_rows(20)
        X["ward_name"] = [f"W{i}" for i in range(len(X))]
        y = pd.Series(2.0 ** np.arange(len(X)), index=X.index)
        encoded, target, _ = fit_preprocessor(X, y)
        self.assertTrue(encoded.index.equals(target.index))
        self.assertFalse(np.isclose(encoded.ward_median_price, target).any())

    def test_mixed_dates_and_unknown_dates_do_not_remove_rows_or_create_features(self):
        raw = example_rows(8)
        raw["price"] = [2e9, 2e9, 6e9, 6e9, 10e9, 10e9, 15e9, 15e9]
        raw["published_at"] = [
            "2025-06-18T10:57:06", "2025-06-18T10:57:06.123456",
            "2026-03-30", "2025-06-20T15:23:36", "2026-03-30T23:58:16.386000",
            "2025-06-25T19:45:45", None, "invalid",
        ]
        eligible = clean(raw)
        self.assertEqual(len(eligible), len(raw))
        info = dataset_summary(raw, eligible)
        self.assertEqual(info["unparsed_date_rows"], 2)
        self.assertEqual(info["date_rows_removed"], 0)
        encoded, _, meta = fit_preprocessor(eligible[INPUT_COLUMNS], eligible.price_ty)
        self.assertFalse({"month", "year", "published_at"} & set(encoded.columns))
        self.assertFalse({"month", "year", "published_at"} & set(meta.to_metadata()["feature_cols"]))

    def test_clean_does_not_learn_quantiles_from_other_rows(self):
        raw = example_rows(100)
        raw["price"] = np.linspace(2e9, 20e9, len(raw))
        raw.loc[raw.index[-1], "price"] = 100e9
        full = clean(raw)
        self.assertIn(raw.index[-1], full.index)
        self.assertEqual(len(full), len(raw))
        assert_frame_equal(full.loc[raw.index[:2]], clean(raw.iloc[:2]))

    def test_clear_data_entry_errors_are_removed_by_fixed_row_rules(self):
        raw = example_rows(8)
        raw["area"] = 100.
        raw["price"] = [
            10e9, 595000e9, 52999e9, 20e9, 99e9, 4050e9, 20e9, 20e9,
        ]
        raw["floor_count"] = [5., 5., 4., 71., 10., 32., 10., 50.]
        raw["bedroom_count"] = [4., 3., 4., 4., 102., 1., 102., 4.]
        raw["bathroom_count"] = [4., 3., 5., 4., 1., 1., 20., 4.]
        flags = data_entry_error_flags(raw)
        self.assertEqual(int(flags.price_above_10000_ty.sum()), 2)
        self.assertEqual(int(flags.floor_count_outside_bounds.sum()), 3)
        self.assertEqual(int(flags.bedroom_count_outside_bounds.sum()), 2)
        self.assertEqual(clean(raw).index.tolist(), [raw.index[0]])

    def test_price_per_m2_below_3_million_is_removed(self):
        raw = example_rows(3)
        raw["area"] = [100., 100., 100.]
        raw["price"] = [299_999_999., 300_000_000., 301_000_000.]
        flags = data_entry_error_flags(raw)
        self.assertEqual(
            flags.price_per_m2_below_min.tolist(), [True, False, False]
        )
        self.assertEqual(clean(raw).index.tolist(), raw.index[1:].tolist())

    def test_area_bounds_are_inclusive(self):
        raw = example_rows(4)
        raw["area"] = [9.9, 10., 2500., 2500.1]
        raw["price"] = 10e9
        flags = data_entry_error_flags(raw)
        self.assertEqual(flags.area_below_min.tolist(), [True, False, False, False])
        self.assertEqual(flags.area_above_max.tolist(), [False, False, False, True])
        self.assertEqual(clean(raw).index.tolist(), raw.index[1:3].tolist())

    def test_optional_numeric_bounds_and_frontage_area_rule(self):
        raw = example_rows(10)
        raw["area"] = 100.
        raw["price"] = 10e9
        raw.loc[raw.index[0], "bedroom_count"] = 0.
        raw.loc[raw.index[1], "bedroom_count"] = 51.
        raw.loc[raw.index[2], "bathroom_count"] = 0.
        raw.loc[raw.index[3], "bathroom_count"] = 51.
        raw.loc[raw.index[4], "floor_count"] = 0.
        raw.loc[raw.index[5], "floor_count"] = 31.
        raw.loc[raw.index[6], "frontage_width"] = 0.
        raw.loc[raw.index[7], ["area", "frontage_width"]] = [200., 101.]
        raw.loc[raw.index[8], ["area", "frontage_width"]] = [10., 11.]
        flags = data_entry_error_flags(raw)
        self.assertEqual(int(flags.bedroom_count_outside_bounds.sum()), 2)
        self.assertEqual(int(flags.bathroom_count_outside_bounds.sum()), 2)
        self.assertEqual(int(flags.floor_count_outside_bounds.sum()), 2)
        self.assertEqual(int(flags.frontage_width_outside_bounds.sum()), 2)
        self.assertEqual(int(flags.frontage_width_above_area.sum()), 1)
        self.assertEqual(clean(raw).index.tolist(), [raw.index[9]])

    def test_full_evaluation_uses_five_folds_and_keeps_test_outliers(self):
        X = example_rows(120)
        y = pd.Series(np.linspace(2, 50, len(X)), index=X.index)
        y.iloc[-1] = 1000.
        with patch("training.get_models", return_value={"Linear": LinearRegression()}), \
             patch("training.cross_validate", wraps=cross_validate) as validate:
            evaluation = train_and_evaluate(
                X.iloc[:100], y.iloc[:100], X.iloc[100:], y.iloc[100:],
                pd.Series(np.arange(100), index=X.index[:100]),
            )
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(len(evaluation.predictions), 20)
        self.assertEqual(evaluation.audit["test_rows_removed_by_outlier_filter"], 0)
        self.assertEqual(evaluation.audit["train_rows_before_filter"], 100)
        self.assertEqual(evaluation.audit["train_rows_after_filter"], 100)
        for fold in evaluation.audit["folds"]:
            self.assertEqual(fold["train_rows_before_filter"], fold["train_rows_after_filter"])
            self.assertIsNone(fold["outlier_bounds"])
        self.assertTrue(np.isfinite(evaluation.metrics[evaluation.best_name]["test_mae"]))
        self.assertEqual(len(evaluation.fold_assignment.dropna()), 100)
        self.assertEqual(len(evaluation.metrics[evaluation.best_name]["fold_scores"]), 5)


if __name__ == "__main__":
    unittest.main()
