"""Version hóa và lưu kết quả; metadata dictionary chỉ dùng ở ranh giới này."""

import hashlib
import json
import logging
import pickle
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import sklearn

from config import DATA_DIR, DEDUP_GROUP_COLUMNS, N_SPLITS, PROJECT_ROOT
from config import RANDOM_STATE, RESULTS_DIR, TARGET_TRANSFORM, TEST_SIZE
from data_loader import parquet_files
from preprocessing import HousePricePreprocessor
from training import EvaluationResult, get_feature_importance

logger = logging.getLogger(__name__)


def dataset_sha256(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in parquet_files(data_dir):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    return sorted([
        *root.joinpath("src").glob("*.py"), *root.joinpath("tests").glob("*.py"),
        root / "app.py", root / "requirements.txt", root / "README.md",
    ])


def source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RunContext:
    project_root: Path
    data_dir: Path
    run_dir: Path
    data_hash: str
    code_hash: str
    version_id: str


def start_run(
    project_root: Path = PROJECT_ROOT, data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
) -> RunContext:
    """Snapshot tất cả source modules trước khi đọc dữ liệu/fit model."""
    data_hash = dataset_sha256(data_dir)
    code_hash = source_sha256(project_root)
    version_id = f"{data_hash[:8]}-{code_hash[:8]}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = results_dir / "runs" / f"{timestamp}-{version_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for path in source_files(project_root):
        destination = run_dir / "source" / path.relative_to(project_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return RunContext(project_root, data_dir, run_dir, data_hash, code_hash, version_id)


def build_report(
    run: RunContext, data_summary: dict, evaluation: EvaluationResult,
    preprocessor: HousePricePreprocessor, groups: pd.Series,
) -> dict:
    metadata = preprocessor.to_metadata()
    return {
        "schema_version": 4,
        "target_transform": TARGET_TRANSFORM,
        "version_id": run.version_id,
        "run_id": run.run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            **data_summary,
            "data_directory": str(run.data_dir.relative_to(run.project_root)),
            "files": [path.name for path in parquet_files(run.data_dir)],
            "sha256": run.data_hash,
            "train_rows": evaluation.audit["train_rows_before_filter"],
            "test_rows": evaluation.audit["test_rows"],
            "deployment_fit_rows": preprocessor.fit_rows_after_filter_,
        },
        "source": {
            "sha256": run.code_hash,
            "files": [str(path.relative_to(run.project_root))
                      for path in source_files(run.project_root)],
        },
        "environment": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "evaluation": {
            "split": "GroupShuffleSplit first; then deduplicate train and test separately",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_splits": N_SPLITS,
            "group_columns": DEDUP_GROUP_COLUMNS,
            "group_strategy": "normalized feature/location/price fingerprint; not a property ID",
            "unique_groups": int(groups.nunique()),
            "largest_group_rows": int(groups.value_counts().max()),
            "train_test_group_overlap": 0,
            "cross_validation": (
                "GroupKFold; independent train-only imputation/category fits; "
                "no percentile outlier filtering; "
                "inner GroupKFold OOF for train target encoding; "
                "validation uses outer-train statistics only"
            ),
            "best_model": evaluation.best_name,
            "models": evaluation.metrics,
            **evaluation.audit,
        },
        "deployment": {
            "fit_rows_before_filter": metadata["fit_rows_before_filter"],
            "fit_rows_after_filter": metadata["fit_rows_after_filter"],
            "outlier_bounds": metadata["outlier_bounds"],
            "train_outlier_filter": metadata["train_outlier_filter"],
            "numeric_medians": metadata["numeric_medians"],
            "feature_cols": metadata["feature_cols"],
            "metrics_note": (
                "Test metrics describe the training-split model; "
                "deployment is refit after evaluation."
            ),
        },
        "run_directory": str(run.run_dir.relative_to(run.project_root)),
    }


def save_artifacts(
    run: RunContext, data_summary: dict, evaluation: EvaluationResult,
    model: object, preprocessor: HousePricePreprocessor, groups: pd.Series,
    test_index: pd.Index,
) -> dict:
    """Lưu model và state cùng phiên bản; chỉ dọn run cũ sau khi xuất thành công."""
    if run.data_hash != dataset_sha256(run.data_dir):
        raise RuntimeError("Dữ liệu đã đổi trong lúc train; không công bố artifact.")
    if run.code_hash != source_sha256(run.project_root):
        raise RuntimeError("Mã nguồn đã đổi trong lúc train; không công bố artifact.")
    report = build_report(run, data_summary, evaluation, preprocessor, groups)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    assignment = pd.DataFrame({
        "group_id": groups, "split": "train",
        "cv_fold": evaluation.fold_assignment.reindex(groups.index),
    })
    assignment.loc[test_index, "split"] = "test"
    assignment["retained_for_evaluation_fit"] = assignment.index.isin(
        evaluation.preprocessor.retained_index_,
    )
    assignment.to_csv(run.run_dir / "splits.csv", index_label="source_row_index")
    evaluation.predictions.to_csv(
        run.run_dir / "test_predictions.csv", index_label="source_row_index",
    )
    bundle = {
        "target_transform": TARGET_TRANSFORM,
        "model": model, "preprocessor": preprocessor,
        "best_model_name": evaluation.best_name,
        "best_model_mae": evaluation.metrics[evaluation.best_name]["test_mae"],
        "cv_summary": evaluation.metrics,
        "feature_importance": get_feature_importance(model, preprocessor.feature_cols_),
        "artifact_version": report, **preprocessor.to_metadata(),
    }
    evaluation_bundle = {
        "target_transform": TARGET_TRANSFORM,
        "model": evaluation.model, "preprocessor": evaluation.preprocessor,
        **evaluation.preprocessor.to_metadata(),
    }
    for filename, contents in [
        ("model.pkl", bundle), ("evaluation_model.pkl", evaluation_bundle),
    ]:
        with (run.run_dir / filename).open("wb") as destination:
            pickle.dump(contents, destination)
    (run.run_dir / "results.json").write_text(report_text, encoding="utf-8")
    artifacts_dir = run.project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Both payloads carry a version to reject mixed files after interrupted writes.
    model_payload = {key: value for key, value in bundle.items()
                     if key not in {"preprocessor", *preprocessor.to_metadata()}}
    model_payload["version_id"] = run.version_id
    state_payload = {"preprocessor": preprocessor, "version_id": run.version_id,
                     "target_transform": TARGET_TRANSFORM,
                     **preprocessor.to_metadata()}
    for filename, payload in [("model.pkl", model_payload),
                              ("preprocessor.pkl", state_payload)]:
        temporary = artifacts_dir / (filename + ".tmp")
        with temporary.open("wb") as destination:
            pickle.dump(payload, destination)
        temporary.replace(artifacts_dir / filename)
    results_dir = run.run_dir.parent.parent
    temporary_report = results_dir / "metrics.json.tmp"
    temporary_report.write_text(report_text, encoding="utf-8")
    temporary_report.replace(results_dir / "metrics.json")
    temporary_predictions = results_dir / "evaluation.csv.tmp"
    evaluation.predictions.to_csv(temporary_predictions, index_label="source_row_index")
    temporary_predictions.replace(results_dir / "evaluation.csv")
    for old_run in run.run_dir.parent.iterdir():
        # Không xóa ra ngoài results/runs, kể cả khi có junction/symlink.
        if (old_run != run.run_dir and old_run.is_dir()
                and old_run.resolve().parent == run.run_dir.parent.resolve()):
            shutil.rmtree(old_run)
    for old_report in results_dir.glob("model_results_*.json"):
        old_report.unlink()
    logger.info("Đã lưu phiên bản %s vào %s", run.version_id, run.run_dir)
    return report


def load_artifacts(artifacts_dir: Path) -> dict:
    """Load a matching model/state pair saved by this pipeline (trusted files only)."""
    payloads = []
    for filename in ["model.pkl", "preprocessor.pkl"]:
        with (artifacts_dir / filename).open("rb") as source:
            payloads.append(pickle.load(source))
    model_payload, state_payload = payloads
    if (not model_payload.get("version_id")
            or model_payload["version_id"] != state_payload.get("version_id")):
        raise ValueError("Model và preprocessor không cùng phiên bản; hãy train lại.")
    return {**model_payload, **state_payload}
