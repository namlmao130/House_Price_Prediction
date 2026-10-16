"""Cấu hình dữ liệu và ML dùng chung; không có thống kê học từ dataset."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "parquet_dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
PROVINCE_NAME = "Hà Nội"
PARQUET_FILE_COUNT = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
OUTLIER_QUANTILES = (0.01, 0.99)
PRICE_SCALE = 1_000_000_000
PRICE_UNIT = "ty VND"
TARGET_TRANSFORM = "log1p"

IMPUTE_NUMERIC_COLUMNS = [
    "bedroom_count", "bathroom_count", "floor_count", "frontage_width",
]
NUMERIC_COLUMNS = ["area", *IMPUTE_NUMERIC_COLUMNS]
MISSING_INDICATOR_COLUMNS = [f"{name}_missing" for name in IMPUTE_NUMERIC_COLUMNS]
CATEGORICAL_COLUMNS = ["property_type_name", "district_name"]
INPUT_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["ward_name"]
DEDUP_GROUP_COLUMNS = NUMERIC_COLUMNS + [
    "district_name", "ward_name", "property_type_name", "price_ty",
]
ENCODING_COLUMNS = ["district_median_price", "ward_median_price"]
RAW_COLUMNS = ["published_at", "price", *INPUT_COLUMNS]
MAX_PRICE_TY = 10_000.0
MAX_FLOOR_COUNT = 70.0
MIN_ROOMS_FOR_RATIO_CHECK = 20.0
MIN_BATHROOM_PER_BEDROOM = 0.1
MAX_BATHROOM_PER_BEDROOM = 3
MAX_PRICE_PER_M2 = 4

