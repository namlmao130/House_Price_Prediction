# Dự đoán giá nhà Hà Nội — dữ liệu 2025–2026

Dự đoán **giá chào bán**, đơn vị tỷ VNĐ. Ngày đăng chỉ mô tả dữ liệu; không
dùng làm feature và không đánh giá theo thời gian.

## Chạy project

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src\train_model.py
streamlit run app.py
```

Đặt đủ `shard_0000.parquet` đến `shard_0009.parquet` trong
`data/parquet_dataset/`. Loader lọc Hà Nội ngay trong PyArrow và chỉ đọc 10 cột
cần dùng. Không tự tải dữ liệu hoặc thêm feature.

## Cấu trúc

```text
app.py                    Giao diện hiện hữu; đọc model bundle
artifacts/
  model.pkl               Model triển khai, tạo sau khi train
  preprocessor.pkl        Preprocessor đã fit và metadata
src/
  __init__.py             Đánh dấu package
  config.py               Đường dẫn, feature, ngưỡng, seed và số fold
  data_loader.py          Đọc các shard Parquet
  cleaning.py             Chuẩn hóa, kiểm tra lỗi và deduplicate
  splitting.py            Group key, chia group, tạo X/y
  preprocessing.py        HousePricePreprocessor và các phép biến đổi
  evaluation.py           R², MAE, RMSE và định dạng log metric
  training.py             CV, chọn model, test evaluation, final refit
  artifacts.py            Hash, snapshot source, lưu model/state/kết quả
  validation.py           Kiểm tra schema và align index
  train_model.py          main() điều phối pipeline
tests/                    Kiểm thử leakage, edge case, artifact và giao diện
docs/REFACTOR.md           Báo cáo thay đổi, trước/sau và checklist leakage
docs/CODE_REFERENCE.md    Toàn bộ source theo từng file
results/metrics.json      Kết quả của lần train hoàn tất gần nhất
results/evaluation.csv    Giá thật và dự đoán trên test
```

## Flow

```text
load raw parquet → clean theo dòng → build group key → GroupShuffleSplit
→ deduplicate riêng train/test → align groups_train theo index còn lại
→ tạo X/y → CV chỉ trên train → chọn model theo CV R² trung bình
→ fit preprocessing/model trên train → transform và evaluate toàn bộ test
→ refit model được chọn trên dữ liệu đã khử trùng → save model + state
```

Mỗi lượt fit preprocessing thực hiện:

```text
fit numeric median trên train → tạo missing indicators → fill train
→ fit outlier bounds train → lọc train → học categories/statistics train
→ inner GroupKFold → tạo target encoding OOF cho train
```

CV tạo **preprocessor mới trong từng outer-fold**, chạy các model trên cùng
feature của fold rồi giải phóng fold trước. Không dùng preprocessor fit trên
toàn bộ train để biến đổi validation trong CV.

## API preprocessing

```python
from preprocessing import HousePricePreprocessor

preprocessor = HousePricePreprocessor()
X_fit, y_fit = preprocessor.fit_transform(X_train, y_train, groups_train)
X_test_encoded = preprocessor.transform(X_test)
model.fit(X_fit, y_fit)
predictions = model.predict(X_test_encoded)
```

`fit()` trả `self`. `fit_transform()` trả **(X đã lọc + OOF, y đã align)** vì
lọc ngoại lệ thay đổi số dòng train; đây không phải sklearn TransformerMixin
để ghép tùy ý vào Pipeline. Dữ liệu train phải dùng `fit_transform()` để nhận
OOF. `transform()` dùng thống kê toàn train, dành cho validation/test và suy luận;
không nhận y, không fit và không bỏ dòng.

State gồm `numeric_medians_`, `district_medians_`, `ward_medians_`,
`global_median_price_`, `categories_`, `feature_cols_`, `outlier_bounds_`.
Chỉ chuyển state sang dictionary tại ranh giới lưu artifact bằng `to_metadata()`;
giữ key cũ như `feature_cols`, `districts`, `numeric_medians`, `price_unit` cho app.

## Quy tắc được giữ

- Bắt buộc có giá, diện tích, quận và loại BĐS; giá và diện tích phải dương.
- Bốn cột số tùy chọn được giữ NaN: phòng ngủ, phòng tắm, số tầng, mặt tiền.
  Median học từ train trước lọc ngoại lệ; có bốn feature `*_missing`.
- Numeric âm bị loại nếu có giá trị. Inf/-inf chuyển thành missing ở cleaning.
- Giữ heuristic `frontage_width <= area` nếu có mặt tiền; đây là quy tắc đồ án,
  không phải định luật so sánh mét với mét vuông.
- Giá trên 10.000 tỷ hoặc số tầng trên 70 bị loại.
- Từ 20 phòng ngủ trở lên: loại nếu phòng tắm dưới 10% phòng ngủ.
  Nếu phòng ngủ > 0: loại nếu phòng tắm vượt 3 lần phòng ngủ.
- Giá trên **4 tỷ/m²** bị loại. Không có quy tắc diện tích trên số phòng ngủ.
- Không áp dụng MIN_AREA/MAX_AREA: hai constant cũ không được dùng nên đã bỏ.
- Group key giữ nguyên đặc trưng số, quận, phường, loại BĐS và **giá**; chuỗi
  chuẩn hóa Unicode, khoảng trắng, casefold. Khác giá là khác group theo quy ước.
  Đây là fingerprint, không phải ID bất động sản đã xác thực.
- GroupShuffleSplit test 20% số group, seed 42; không nhất thiết 20% số tin thô.
- GroupKFold 5 outer-fold; inner OOF dùng tối đa 5 fold, tối thiểu 2 group.
- Outlier bounds theo phân vị 1%–99% của giá và diện tích, chỉ lọc train.
- Fallback giá phường → quận → global train. Quận/phường chưa gặp không gây lỗi.
- One-hot quận và loại BĐS học danh mục từ train; unknown category cho dummy = 0.
- Giữ Linear Regression + StandardScaler, Random Forest, Gradient Boosting và
  toàn bộ hyperparameter. Chỉ Random Forest dùng `n_jobs=-1` như bản trước.

Feature gồm 5 numeric, 4 missing indicators, 2 target encoding và one-hot.
Không thêm `house_depth`, thời gian, `price`, `price_per_m2` hay log target.

## Kết quả và phiên bản

R² có thể âm, không phải phần trăm chính xác. MAE/RMSE có đơn vị tỷ VNĐ.
Không thêm MAPE vì chưa có quyết định xử lý giá rất nhỏ. Chọn model bằng
CV R² trung bình trước khi đánh giá test; kết quả test không chọn model.

Model triển khai được refit sau đánh giá trên toàn bộ dữ liệu khử trùng.
R²/MAE/RMSE báo cáo thuộc model chỉ fit trên tập train của thí nghiệm; không
đánh giá lại test bằng model triển khai. Biên ± MAE trên app chỉ tham khảo,
không bảo đảm mức bao phủ như khoảng tin cậy.

Mỗi run lưu source, metadata, split assignment, dự đoán test, model đánh giá và
model triển khai. Hash bao phủ toàn bộ `src/*.py`, tests, app, README và
requirements; hash dữ liệu bao phủ đúng 10 shard theo thứ tự. `source_row_index`
là index sau đọc Hà Nội, không phải số dòng gốc trong từng shard.

Sau khi xuất thành công, chỉ giữ run mới nhất. Nếu dữ liệu hoặc mã nguồn thay
đổi giữa lúc chạy, quá trình lưu sẽ báo lỗi. Mỗi file model/latest được thay
qua file tạm; hai file không phải một giao dịch filesystem duy nhất.

Refactor không ghi đè model/kết quả huấn luyện cũ bằng điểm số kiểm thử.
Xem `version_id` và source snapshot trong artifact để biết model được tạo bởi
phiên bản nào; chạy lệnh train ở trên để tạo artifact từ code mới.

## Cấu trúc lưu kết quả hiện tại

- `artifacts/model.pkl`: estimator và thông tin đánh giá.
- `artifacts/preprocessor.pkl`: preprocessor đã fit và metadata.
- Hai file phải cùng version; app kiểm tra khi nạp.
- `results/metrics.json`: chỉ số và thông tin phiên bản.
- `results/evaluation.csv`: giá thật và dự đoán từng model trên test.
- `results/runs/<version>/`: snapshot source, split và bundle để tái lập; chỉ giữ run mới nhất.
- Dữ liệu, artifacts và results được giữ local, không track Git.
- MAPE chưa được bật; các metric vẫn là MAE, RMSE, R².

Chạy train từ thư mục project: `venv\Scripts\python.exe src\train_model.py`.
