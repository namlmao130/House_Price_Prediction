"""Entry point: python src/train_model.py."""

import logging
import sys

from artifacts import save_artifacts, start_run
from cleaning import clean_listings, dataset_summary
from data_loader import load_raw_data
from cleaning import deduplicate
from splitting import build_xy, split_train_test
from training import refit_final_model, train_and_evaluate
from validation import ensure_disjoint_groups


def main() -> None:
    run = start_run()
    raw_df = load_raw_data()
    cleaned_df = clean_listings(raw_df)
    data_summary = dataset_summary(raw_df, cleaned_df)
    del raw_df

    train_df, test_df, groups = split_train_test(cleaned_df)
    train_df = deduplicate(train_df, groups.loc[train_df.index])
    test_df = deduplicate(test_df, groups.loc[test_df.index])
    X_train, y_train = build_xy(train_df)
    X_test, y_test = build_xy(test_df)
    groups_train = groups.loc[X_train.index]
    ensure_disjoint_groups(groups_train, groups.loc[X_test.index])
    data_summary.update({
        "deduplicated_rows": len(train_df) + len(test_df),
        "duplicate_rows_removed": len(cleaned_df) - len(train_df) - len(test_df),
        "deduplication_keep": "first source row per group within each split",
    })
    del train_df, test_df
    evaluation = train_and_evaluate(X_train, y_train, X_test, y_test, groups_train)

    # Giữ thứ tự nguồn khi refit, để thứ tự train/test không đổi thứ tự OOF cuối.
    retained = cleaned_df.index.isin(X_train.index.union(X_test.index))
    X_final, y_final = build_xy(cleaned_df.loc[retained])
    groups_final = groups.loc[X_final.index]
    del cleaned_df, groups, X_train, y_train, X_test, y_test
    model, preprocessor = refit_final_model(
        evaluation.best_name, X_final, y_final, groups_final,
    )
    save_artifacts(
        run, data_summary, evaluation, model, preprocessor, groups_final,
        evaluation.predictions.index,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
