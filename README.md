# Dự đoán giá nhà Hà Nội — dữ liệu 2025–2026

Dự đoán **giá chào bán** (tỷ VNĐ) của nhà ở Hà Nội từ diện tích, số phòng, số tầng,
mặt tiền, loại bất động sản, quận/huyện và phường/xã. Project gồm pipeline huấn
luyện có kiểm soát rò rỉ dữ liệu, ba mô hình hồi quy được so sánh bằng
cross-validation theo nhóm, và ứng dụng Streamlit để định giá và xem kết quả đánh giá.
Ngày đăng chỉ mô tả dữ liệu; không dùng làm feature và không đánh giá theo thời gian.

## Kết quả

Dữ liệu: 1.140.542 tin thô của Hà Nội → 875.967 tin hợp lệ → 657.691 mẫu sau khử
trùng (train 526.152, test 131.539, chia theo nhóm tin trùng). Giá trung vị toàn
tập là 11,26 tỷ. Mô hình huấn luyện trên `log1p(giá)`; metric tính trên giá gốc sau
`expm1()`, riêng R² log tính trên `log1p`.

| Mô hình | CV R² | Test R² | R² log | MAE (tỷ) | Median APE | Lệch ≤ 20% |
|---|---|---|---|---|---|---|
| **Random Forest** (được chọn) | 0,660 ± 0,040 | **0,711** | **0,851** | **5,12** | **17,4%** | **55,9%** |
| Gradient Boosting | 0,634 ± 0,065 | 0,701 | 0,850 | 5,22 | 18,0% | 54,6% |
| Ridge (log1p) | 0,573 ± 0,059 | 0,614 | 0,816 | 5,91 | 21,0% | 48,0% |
| Baseline (đoán trung vị train) | — | — | — | 12,07 | 49,3% | — |

Random Forest giảm MAE khoảng 58% so với baseline; CV và test sát nhau nên không có
dấu hiệu rò rỉ hay overfit. Ridge trên đặc trưng `log1p` đạt R² log 0,816, gần các mô
hình cây (0,85), nên phần lớn cấu trúc của dữ liệu đã nằm trên thang log-log; cây
còn hơn ở tương tác giữa vị trí, loại nhà và diện tích, rõ nhất ở R² trên giá gốc
(0,711 so với 0,614) và tỷ lệ lệch ≤ 20% (55,9% so với 48,0%).

Lần train trước dùng Linear Regression không regularize và cho R² âm rất lớn (test R²
−2,68 triệu, MAE 449,6 tỷ): 45 dự đoán vượt 10.000 tỷ do diện tích thô bị ngoại suy
rồi `expm1` khuếch đại. Việc đổi sang Ridge với `log1p` cho các cột số đã khắc phục.

### Sai số của Random Forest theo nhóm (tập test)

| Khoảng giá thật | Số mẫu | Median APE | Lệch ≤ 20% |
|---|---|---|---|
| < 3 tỷ | 2.295 | 28,6% | 39,6% |
| 3–6 tỷ | 19.121 | 14,4% | 62,2% |
| 6–10 tỷ | 37.437 | 14,4% | 63,8% |
| 10–20 tỷ | 39.563 | 17,9% | 54,9% |
| > 20 tỷ | 33.123 | 22,2% | 45,4% |

Model chính xác nhất ở phân khúc 3–10 tỷ. Theo loại nhà, căn hộ chung cư tốt nhất
(Median APE 12,9%), shophouse kém nhất (26,7%). Theo quận, Hoàn Kiếm (giá trung vị
32 tỷ) có sai số tuyệt đối lớn nhất (MAE khoảng 31 tỷ). Chi tiết từng quận nằm trong
tab *Đánh giá mô hình* của app.

### Cách đọc các chỉ số

| Chỉ số | Ý nghĩa |
|---|---|
| R² | Tỷ lệ phương sai giải thích được; có thể âm nếu kém hơn dự đoán bằng trung bình. Trên giá gốc, R² bị vài căn rất đắt chi phối |
| MAE / RMSE (tỷ) | Sai số tuyệt đối trung bình / căn bậc hai sai số bình phương; RMSE nhạy với sai số lớn |
| MAPE (%) | Sai số phần trăm tuyệt đối trung bình; bị kéo lên bởi vài dự đoán lệch rất lớn |
| Median APE (%) | Trung vị sai số phần trăm; phản ánh căn nhà điển hình tốt hơn MAPE |
| Lệch ≤ 10% / 20% | Tỷ lệ căn có giá dự đoán lệch không quá 10% / 20% so với giá thật |
| R² log, RMSLE | R² và RMSE trên `log1p(giá)`, cùng thang với target khi huấn luyện |
| Bias | Trung vị `dự đoán / thực tế`; nhỏ hơn 1 nghĩa là model thường đoán thấp hơn |

## Hạn chế

- **Giá chào bán, không phải giá giao dịch.** Giá thật khi chốt thường thấp hơn.
- **Chỉ Hà Nội, không có đất.** Dòng `property_type_name = "Đất"` bị loại; app chỉ
  định giá nhà, biệt thự/nhà liền kề, shophouse và căn hộ chung cư.
- **Ít đặc trưng.** Chỉ có 5 biến số, quận, phường và loại bất động sản. Không có tọa
  độ, hướng nhà, pháp lý, khoảng cách tới trung tâm hay độ rộng ngõ — những yếu tố
  ảnh hưởng mạnh đến giá nhà phố.
- **Test được lọc bằng quy tắc có dùng giá.** Các tin có đơn giá dưới 3 triệu/m² hoặc
  trên 4 tỷ/m² bị loại trước khi chia tập, nên điểm số mô tả chất lượng trên dữ liệu
  đã làm sạch chứ không phải mọi tin đăng thực tế.
- **Khử trùng chỉ là fingerprint.** Group key gồm đặc trưng, vị trí và giá; cùng một
  căn đăng lại với giá khác vẫn được coi là hai nhóm.
- **Khoảng tham khảo trên app** là `giá dự đoán × (1 ± MAPE)`, không phải khoảng tin cậy.
- **Sai số theo khoảng giá thật** thường cho thấy dự đoán cao ở nhóm rẻ và thấp ở nhóm
  đắt (hồi quy về trung bình); dùng để biết model yếu ở phân khúc nào, không phải để
  kết luận model lệch trên từng căn.
- Target encoding quận/phường chưa làm mượt, nên phường ít tin có thể cho giá nhiễu.

## Chạy project

Yêu cầu Python 3.12 trở lên (artifact được train bằng Python 3.13). `scikit-learn`
cần đúng dải 1.9.x để load được `model.pkl`; chạy app không cần train lại vì
`artifacts/` đã có sẵn, còn tab *Tổng quan dữ liệu* cần thêm dữ liệu gốc.

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

Repository không bao gồm các file dữ liệu vì dung lượng lớn. Có thể tải dữ liệu
từ [Hugging Face](https://huggingface.co/datasets/tinixai/vietnam-real-estates)
hoặc [Google Drive](https://drive.google.com/drive/folders/1wL0y6u8cnEI2HwrCupKvUy-EIJFlCFMS),
sau đó đặt các shard vào thư mục nêu trên nếu muốn huấn luyện lại.

## Cấu trúc

```text
app.py                    Ứng dụng Streamlit; đọc model bundle
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
  evaluation.py           Metric tổng thể, thang log và sai số theo phân khúc
  training.py             CV, chọn model, test evaluation, final refit
  artifacts.py            Hash, snapshot source, lưu model/state/kết quả
  validation.py           Kiểm tra schema và align index
  train_model.py          main() điều phối pipeline
tests/                    Kiểm thử leakage, edge case, artifact và giao diện
results/metrics.json      Kết quả của lần train hoàn tất gần nhất
results/evaluation.csv    Giá thật và dự đoán trên test
```

## Flow

```text
load raw parquet → clean theo dòng, loại loại BĐS Đất → build group key → GroupShuffleSplit
→ deduplicate riêng train/test → align groups_train theo index còn lại
→ tạo X/y → CV chỉ trên train → chọn model theo CV R² trung bình
→ fit preprocessing/model trên train → transform và evaluate toàn bộ test
→ refit model được chọn trên dữ liệu đã khử trùng → save model + state
```

Mỗi lượt fit preprocessing thực hiện:

```text
fit numeric median trên train → tạo missing indicators → fill train
→ học categories/statistics train, giữ mọi dòng
→ inner GroupKFold → tạo target encoding OOF cho train
```

CV tạo **preprocessor mới trong từng outer-fold**, chạy các model trên cùng
feature của fold rồi giải phóng fold trước. Không dùng preprocessor fit trên
toàn bộ train để biến đổi validation trong CV.

## API preprocessing

```python
import numpy as np
from preprocessing import HousePricePreprocessor

preprocessor = HousePricePreprocessor()
X_fit, y_fit = preprocessor.fit_transform(X_train, y_train, groups_train)
X_test_encoded = preprocessor.transform(X_test)
model.fit(X_fit, np.log1p(y_fit))
predictions = np.expm1(model.predict(X_test_encoded))
```

`fit()` trả `self`. `fit_transform()` trả **(X OOF, y đã align)** và giữ mọi dòng
train đã qua cleaning/dedup; đây không phải sklearn TransformerMixin
để ghép tùy ý vào Pipeline. Dữ liệu train phải dùng `fit_transform()` để nhận
OOF. `transform()` dùng thống kê toàn train, dành cho validation/test và suy luận;
không nhận y, không fit và không bỏ dòng.

State gồm `numeric_medians_`, `district_medians_`, `ward_medians_`,
`global_median_price_`, `categories_`, `feature_cols_`, `outlier_bounds_`.
Chỉ chuyển state sang dictionary tại ranh giới lưu artifact bằng `to_metadata()`;
giữ key cũ như `feature_cols`, `districts`, `numeric_medians`, `price_unit` cho app.

## Quy tắc được giữ

- Bắt buộc có giá, diện tích, quận và loại BĐS; giá và diện tích phải dương.
- Loại dòng có `property_type_name = "Đất"` ở cleaning trước split/dedup,
  sau chuẩn hóa Unicode, khoảng trắng, không phân biệt hoa/thường.
  Giữ cột loại BĐS cho các loại còn lại; không xóa dữ liệu nguồn trên đĩa.
- Bốn cột số tùy chọn được giữ NaN: phòng ngủ, phòng tắm, số tầng, mặt tiền.
  Median học từ train; có bốn feature `*_missing`.
- Numeric âm bị loại nếu có giá trị. Inf/-inf chuyển thành missing ở cleaning.
- Diện tích phải từ 10 đến 2.500 m². Các cột tùy chọn, nếu có, phải thỏa:
  1–50 phòng ngủ, 1–50 phòng tắm, 1–30 tầng và mặt tiền 1–100 m.
- Mặt tiền không được lớn hơn diện tích.
- Giá trên 10.000 tỷ bị loại.
- Giá dưới **3 triệu/m²** hoặc trên **4 tỷ/m²** bị loại. Không có quy tắc diện tích trên số phòng ngủ.
- Group key giữ nguyên đặc trưng số, quận, phường, loại BĐS và **giá**; chuỗi
  chuẩn hóa Unicode, khoảng trắng, casefold. Khác giá là khác group theo quy ước.
  Đây là fingerprint, không phải ID bất động sản đã xác thực.
- GroupShuffleSplit test 20% số group, seed 42; không nhất thiết 20% số tin thô.
- GroupKFold 5 outer-fold; inner OOF dùng tối đa 5 fold, tối thiểu 2 group.
- Không lọc phân vị giá/diện tích trong train, CV hoặc refit. Metadata giữ
  `outlier_bounds = null`, `train_outlier_filter = "none"`; số dòng trước/sau bằng nhau.
- Fallback giá phường → quận → global train. Quận/phường chưa gặp không gây lỗi.
- One-hot quận và loại BĐS học danh mục từ train; unknown category cho dummy = 0.
- Dùng Ridge Regression với `log1p` cho 5 feature số, StandardScaler và RidgeCV;
  Random Forest và Gradient Boosting giữ nguyên feature. RidgeCV chọn alpha nội
  bộ trên từng phần train, không nhìn tập test. Chỉ Random Forest dùng `n_jobs=-1`.

Feature gồm 5 numeric, 4 missing indicators, 2 target encoding và one-hot.
Không thêm `house_depth`, thời gian, `price` hay `price_per_m2` làm feature.
Ridge Regression, Random Forest và Gradient Boosting fit target `np.log1p(price_ty)`
ở CV, train/test và refit. Dự đoán được đổi ngược bằng `np.expm1()` trước khi
tính R²/MAE/RMSE/MAPE, lưu dự đoán hoặc hiển thị trên app. OOF target encoding
vẫn học từ giá gốc. Results và artifacts ghi
`target_transform = "log1p"`.

## Kết quả và phiên bản

R² có thể âm, không phải phần trăm chính xác. MAE/RMSE có đơn vị tỷ VNĐ.
MAPE = `100 * mean(abs((y_true - y_pred) / y_true))`, lưu theo phần trăm
trong `fold_scores.mape`, `cv_mape` (trung bình các fold) và `test_mape`.
Giá thật phải dương; giá rất nhỏ có thể khiến MAPE cao, không clip hoặc bỏ mẫu.
Ngoài ra pipeline báo Median APE, tỷ lệ lệch không quá 10%/20%, bias,
R²/RMSLE trên thang log và sai số theo khoảng giá, loại BĐS, quận/huyện.
Chọn model bằng
CV R² trung bình trước khi đánh giá test; kết quả test không chọn model.

Model triển khai được refit sau đánh giá trên toàn bộ dữ liệu khử trùng.
Các metric báo cáo thuộc model chỉ fit trên tập train của thí nghiệm; không
đánh giá lại test bằng model triển khai. Khoảng tham khảo trên app là
`giá dự đoán × (1 ± MAPE test)`, không phải khoảng tin cậy và không bảo đảm
một mức bao phủ xác suất cụ thể.

Mỗi run lưu source, metadata, split assignment, dự đoán test, model đánh giá và
model triển khai. Hash bao phủ toàn bộ `src/*.py`, tests, app, README và
requirements; hash dữ liệu bao phủ đúng 10 shard theo thứ tự. `source_row_index`
là index sau đọc Hà Nội, không phải số dòng gốc trong từng shard.

Sau khi xuất thành công, chỉ giữ run mới nhất. Nếu dữ liệu hoặc mã nguồn thay
đổi giữa lúc chạy, quá trình lưu sẽ báo lỗi. Các file model, preprocessor,
metrics và evaluation được thay qua file tạm, không phải một giao dịch duy nhất.

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
- `data/parquet_dataset/` và `results/` không track Git (xem `.gitignore`);
  `artifacts/` được kèm trong repo để chạy app ngay mà không cần train.
- Metric chính tính trên giá gốc sau `expm1()`; R² log và RMSLE tính trên
  `log1p(giá)`. Tab đánh giá còn hiển thị sai số theo phân khúc khi artifact mới có dữ liệu.

Chạy train từ thư mục project: `venv\Scripts\python.exe src\train_model.py`.
