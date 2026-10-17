import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))
# EDA và training dùng chung cleaning, bao gồm loại các dòng Đất.
from cleaning import clean_listings
from data_loader import load_raw_data, parquet_files
from artifacts import load_artifacts
from config import ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH
from cleaning import deduplicate

# Config
st.set_page_config(
    page_title="Dự đoán giá nhà Hà Nội",
    page_icon=":material/home_work:",
    layout="wide",
)


# Load model & data
@st.cache_resource(max_entries=1)
def load_bundle(model_modified_ns, preprocessor_modified_ns):
    """Cache model và tự làm mới khi file model thay đổi."""
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(show_spinner="Đang tải dữ liệu...", max_entries=2)
def load_data(sample_size=None, data_modified_ns=()) -> pd.DataFrame:
    # Khử trùng trước khi lấy mẫu để mỗi tin duy nhất có cơ hội được chọn như nhau.
    df = load_raw_data()
    df = clean_listings(df)
    df = deduplicate(df)

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df


# Main
if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    st.info("Chưa có model và preprocessor. Chạy python src/train_model.py trước.")
    st.stop()
try:
    bundle = load_bundle(
        MODEL_PATH.stat().st_mtime_ns, PREPROCESSOR_PATH.stat().st_mtime_ns
    )
except Exception as error:
    st.error(
        "Không thể tải model và preprocessor. Các artifact có thể bị hỏng, "
        "khác phiên bản hoặc không tương thích với môi trường hiện tại."
    )
    st.caption(f"Chi tiết: {error}")
    st.stop()
model = bundle["model"]
preprocessor = bundle["preprocessor"]
districts = bundle["districts"]  # {district_name: median_price_ty}
wards_by_district = bundle.get("wards_by_district", {})
best_name = bundle["best_model_name"]
cv_summary = bundle.get("cv_summary", {})
importance = bundle.get("feature_importance")
best_metrics = cv_summary.get(best_name, {})
best_mape = best_metrics.get("test_mape", best_metrics.get("cv_mape"))

# Danh sách quận từ bundle
district_list = sorted(districts.keys())

# Header
st.title("Dự đoán Giá Nhà Hà Nội")
st.caption(
    f"Model tốt nhất: {best_name} | Dữ liệu năm 2025–2026 | Source: huggingface/tinixai/vietnam-real-estates"
)

tab1, tab2, tab3 = st.tabs(["Tổng quan dữ liệu", "Dự đoán giá", "Đánh giá mô hình"])


# TAB 1: TỔNG QUAN
with tab1:
    data_modified_ns = tuple(path.stat().st_mtime_ns for path in parquet_files())
    df = load_data(50000, data_modified_ns)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mẫu dữ liệu minh họa", f"{len(df):,}")
    col2.metric("Giá trung vị", f"{df['price_ty'].median():.2f} tỷ")
    col3.metric("Diện tích trung vị", f"{df['area'].median():.0f} m²")
    col4.metric("Số quận/huyện", df["district_name"].nunique())

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Phân phối giá (tỷ VNĐ)")
        fig = px.histogram(
            df,
            x="price_ty",
            nbins=80,
            labels={"price_ty": "Giá (tỷ VNĐ)"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig)

    with col_b:
        st.subheader("Giá trung vị theo quận")
        med_by_district = (
            df.groupby("district_name")["price_ty"]
            .median()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )
        fig2 = px.bar(
            med_by_district,
            x="price_ty",
            y="district_name",
            orientation="h",
            labels={"price_ty": "Giá trung vị (tỷ)", "district_name": "Quận/Huyện"},
            color="price_ty",
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Giá theo loại BĐS")
        fig3 = px.box(
            df.dropna(subset=["property_type_name"]),
            x="property_type_name",
            y="price_ty",
            labels={"property_type_name": "Loại BĐS", "price_ty": "Giá (tỷ VNĐ)"},
            color="property_type_name",
        )
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3)

    with col_d:
        st.subheader("Scatter: Diện tích vs Giá")
        sample_sc = df.sample(min(5000, len(df)), random_state=1)
        fig4 = px.scatter(
            sample_sc,
            x="area",
            y="price_ty",
            color="property_type_name",
            labels={
                "property_type_name": "Loại hình BĐS",
                "area": "Diện tích (m²)",
                "price_ty": "Giá (tỷ VNĐ)",
            },
            opacity=0.5,
        )
        fig4.update_layout(height=350)
        st.plotly_chart(fig4)


# TAB 2: DỰ ĐOÁN
with tab2:
    st.header("Định giá căn nhà", icon=":material/calculate:")
    st.caption(
        "Nhập đặc điểm bất động sản để nhận giá ước tính và khoảng tham khảo "
        "theo sai số phần trăm của mô hình."
    )

    with st.container(border=True):
        st.subheader("Thông tin bất động sản")
        col1, col2 = st.columns(2)

        with col1:
            district = st.selectbox("Quận/huyện", district_list)
            ward = st.selectbox(
                "Phường/xã",
                ["Không rõ", *wards_by_district.get(district, [])],
                key=f"ward_{district}",
                help="Chọn “Không rõ” để dùng mức giá trung vị của quận/huyện.",
            )
            prop_type = st.selectbox(
                "Loại bất động sản",
                ["Nhà", "Biệt thự/Nhà liền kề", "Shophouse", "Căn hộ chung cư"],
            )
            area = st.number_input(
                "Diện tích (m²)",
                min_value=10.0,
                max_value=2500.0,
                value=60.0,
                step=5.0,
            )

        with col2:
            bedroom_count = st.number_input(
                "Số phòng ngủ", min_value=1, max_value=50, value=3, step=1
            )
            bathroom_count = st.number_input(
                "Số phòng tắm", min_value=1, max_value=50, value=2, step=1
            )
            floor_count = st.number_input(
                "Số tầng", min_value=1, max_value=30, value=3, step=1
            )
            frontage_width = st.number_input(
                "Chiều rộng mặt tiền (m)",
                min_value=1.0,
                max_value=100.0,
                value=4.0,
                step=0.5,
            )

        predict_clicked = st.button(
            "Dự đoán giá",
            type="primary",
            icon=":material/analytics:",
            width="stretch",
        )

    if predict_clicked and frontage_width > area:
        st.warning(
            "Chiều rộng mặt tiền không được lớn hơn diện tích.",
            icon=":material/warning:",
        )

    if predict_clicked and frontage_width <= area:
        input_df = pd.DataFrame(
            [
                {
                    "area": float(area),
                    "bedroom_count": float(bedroom_count),
                    "bathroom_count": float(bathroom_count),
                    "floor_count": float(floor_count),
                    "frontage_width": float(frontage_width),
                    "property_type_name": prop_type,
                    "district_name": district,
                    "ward_name": None if ward == "Không rõ" else ward,
                }
            ]
        )

        X_input = preprocessor.transform(input_df)
        pred = model.predict(X_input)[0]
        if bundle.get("target_transform") == "log1p":
            pred = np.expm1(pred)

        if not np.isfinite(pred) or pred <= 0:
            st.warning(
                "Chưa ước lượng được mức giá hợp lệ cho tổ hợp thông tin này. "
                "Bạn có thể kiểm tra lại thông tin; nếu thông tin đã đúng, "
                "mô hình hiện chưa đáp ứng trường hợp này.",
                icon=":material/warning:",
            )
        else:
            with st.container(border=True):
                st.subheader("Kết quả định giá", icon=":material/home_work:")
                st.success(
                    f"Giá ước tính: **{pred:.2f} tỷ VNĐ**",
                    icon=":material/check_circle:",
                )

                if best_mape is not None and np.isfinite(best_mape):
                    error_rate = best_mape / 100
                    lo = max(0, pred * (1 - error_rate))
                    hi = pred * (1 + error_rate)
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("Giá dự đoán", f"{pred:.2f} tỷ", border=True)
                    col_r2.metric("Cận tham khảo dưới", f"{lo:.2f} tỷ", border=True)
                    col_r3.metric("Cận tham khảo trên", f"{hi:.2f} tỷ", border=True)
                    st.caption(
                        f"Khoảng tham khảo được quy đổi theo MAPE {best_mape:.2f}% "
                        f"của {best_name}. Đây không phải khoảng tin cậy và không "
                        "bảo đảm một mức bao phủ xác suất cụ thể."
                    )
                else:
                    st.caption(
                        f"Mô hình sử dụng: {best_name}. Artifact hiện chưa có MAPE "
                        "để tính khoảng tham khảo."
                    )


# TAB 3: ĐÁNH GIÁ MÔ HÌNH
with tab3:
    artifact_version = bundle.get("artifact_version", {})
    data_report = artifact_version.get("data", {})
    evaluation_report = artifact_version.get("evaluation", {})

    st.header("Dữ liệu và mô hình", icon=":material/model_training:")
    st.caption(
        "Toàn bộ các bước lọc, chuẩn hóa, tạo đặc trưng và đánh giá trước khi "
        "mô hình được dùng để dự đoán."
    )

    if artifact_version:
        st.caption(
            f"Phiên bản kết quả: {artifact_version['version_id']} | "
            f"Ngày lưu (UTC): {artifact_version['created_at_utc']}"
        )

    if data_report:
        raw_rows = data_report.get("raw_rows", 0)
        eligible_rows = data_report.get("eligible_rows", 0)
        deduplicated_rows = data_report.get("deduplicated_rows", eligible_rows)
        removed_rows = data_report.get(
            "rows_removed_by_fixed_checks", max(0, raw_rows - eligible_rows)
        )
        duplicate_rows = data_report.get(
            "duplicate_rows_removed", max(0, eligible_rows - deduplicated_rows)
        )
        test_rows = data_report.get("test_rows", evaluation_report.get("test_rows", 0))

        st.subheader(
            f"Từ {raw_rows:,} tin thô đến {deduplicated_rows:,} mẫu sạch",
            icon=":material/filter_alt:",
        )
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Tin thô", f"{raw_rows:,}", border=True)
        kpi_cols[1].metric(
            "Sau kiểm tra hợp lệ",
            f"{eligible_rows:,}",
            f"-{removed_rows:,} dòng",
            delta_color="inverse",
            border=True,
        )
        kpi_cols[2].metric(
            "Sau khử trùng",
            f"{deduplicated_rows:,}",
            f"-{duplicate_rows:,} dòng trùng",
            delta_color="inverse",
            border=True,
        )
        kpi_cols[3].metric("Tập kiểm tra", f"{test_rows:,}", border=True)

        with st.container(border=True):
            st.subheader("Những dữ liệu đã lọc bỏ")
            st.markdown(
                "- Thiếu **giá, diện tích, quận/huyện hoặc loại bất động sản**.\n"
                "- Loại bất động sản là **Đất** sau khi chuẩn hóa chữ.\n"
                "- Giá hoặc diện tích không dương; giá trị số tùy chọn âm.\n"
                "- Vi phạm các ngưỡng cố định trong bảng dưới đây.\n"
                "- Bản ghi trùng theo đặc trưng, vị trí, loại bất động sản và giá."
            )

            rule_names = {
                "price_non_positive": "Giá không dương",
                "price_above_10000_ty": "Giá trên 10.000 tỷ",
                "area_below_min": "Diện tích dưới 10 m²",
                "area_above_max": "Diện tích trên 2.500 m²",
                "bedroom_count_outside_bounds": "Phòng ngủ ngoài khoảng 1–50",
                "bathroom_count_outside_bounds": "Phòng tắm ngoài khoảng 1–50",
                "floor_count_outside_bounds": "Số tầng ngoài khoảng 1–30",
                "frontage_width_outside_bounds": "Mặt tiền ngoài khoảng 1–100 m",
                "frontage_width_above_area": "Mặt tiền lớn hơn diện tích",
                "price_per_m2_below_min": "Đơn giá dưới 3 triệu/m²",
                "price_per_m2_above_max": "Đơn giá trên 4 tỷ/m²",
            }
            error_counts = data_report.get("data_entry_error_counts", {})
            rule_rows = [
                {"Quy tắc gắn cờ": rule_names.get(key, key), "Số dòng": count}
                for key, count in error_counts.items()
                if count
            ]
            if rule_rows:
                st.dataframe(
                    pd.DataFrame(rule_rows),
                    hide_index=True,
                    column_config={
                        "Số dòng": st.column_config.NumberColumn(format="%d")
                    },
                    alt="Số dòng bị gắn cờ theo từng quy tắc làm sạch",
                )
                st.caption(
                    "Một dòng có thể vi phạm nhiều quy tắc, vì vậy không cộng "
                    "trực tiếp các số trong bảng để suy ra tổng số dòng bị loại."
                )

        process_col, feature_col = st.columns(2)
        with process_col.container(border=True, height="stretch"):
            st.subheader("Đã sửa và chuẩn hóa")
            st.markdown(
                "- Chuyển các cột số về kiểu số; `Inf/-Inf` thành giá trị thiếu.\n"
                "- Chuẩn hóa Unicode NFKC, khoảng trắng và chuỗi rỗng.\n"
                "- Giữ giá trị thiếu ở phòng ngủ, phòng tắm, số tầng và mặt tiền.\n"
                "- Điền thiếu bằng trung vị học riêng từ tập train.\n"
                "- Tạo thêm cờ thiếu dữ liệu cho bốn trường số tùy chọn."
            )

        with feature_col.container(border=True, height="stretch"):
            st.subheader("Đã tính toán và tạo đặc trưng")
            st.markdown(
                "- Quy đổi giá sang **tỷ VNĐ** và tính đơn giá/m² để kiểm tra dữ liệu.\n"
                "- Target encoding ngoài-fold cho quận và phường.\n"
                "- One-hot encoding cho quận và loại bất động sản.\n"
                "- Huấn luyện trên `log1p(giá)` và đổi ngược bằng `expm1`.\n"
                "- Không dùng ngày đăng, giá hoặc đơn giá/m² làm feature đầu vào."
            )

        with st.container(border=True):
            st.subheader("Chia dữ liệu và chống rò rỉ")
            train_rows = evaluation_report.get("train_rows_before_filter", 0)
            unique_groups = evaluation_report.get("unique_groups", deduplicated_rows)
            group_overlap = evaluation_report.get("train_test_group_overlap", 0)
            split_cols = st.columns(4)
            split_cols[0].metric("Mẫu train", f"{train_rows:,}")
            split_cols[1].metric("Mẫu test", f"{test_rows:,}")
            split_cols[2].metric("Số group", f"{unique_groups:,}")
            split_cols[3].metric("Group trùng train/test", f"{group_overlap:,}")
            st.markdown(
                "Dữ liệu được chia bằng **GroupShuffleSplit 80/20**; đánh giá chéo "
                "dùng **5-fold GroupKFold**. Mỗi fold fit lại imputation, danh mục "
                "và target encoding chỉ từ phần train của fold."
            )

    st.subheader("Hiệu quả mô hình", icon=":material/query_stats:")

    if cv_summary:
        selected = cv_summary[best_name]
        baseline_mae = evaluation_report.get("baseline_train_median_test_mae")
        improvement = None
        if baseline_mae and baseline_mae > 0:
            improvement = (1 - selected["test_mae"] / baseline_mae) * 100

        has_extra = "test_median_ape" in selected
        mae_delta = (
            f"{-improvement:.0f}% so với baseline" if improvement is not None else None
        )
        score_cols = st.columns(4)
        score_cols[0].metric(
            "Test R²",
            f"{selected['test_r2']:.3f}",
            border=True,
            help="Tỷ lệ phương sai giải thích được trên giá gốc.",
        )
        score_cols[1].metric(
            "Test MAE",
            f"{selected['test_mae']:.2f} tỷ",
            delta=mae_delta,
            delta_color="inverse",
            border=True,
            help="Sai số tuyệt đối trung bình; mũi tên so với baseline đoán trung vị.",
        )
        if has_extra:
            score_cols[2].metric(
                "Sai lệch trung vị",
                f"{selected['test_median_ape']:.2f}%",
                border=True,
                help="Median APE: một nửa số dự đoán lệch ít hơn mức này.",
            )
            score_cols[3].metric(
                "Dự đoán lệch ≤ 20%",
                f"{selected['test_hit_rate_20'] * 100:.2f}%",
                border=True,
                help="Tỷ lệ căn có giá dự đoán lệch không quá 20%.",
            )
        else:
            score_cols[2].metric(
                "Test MAPE",
                f"{selected['test_mape']:.2f}%",
                border=True,
            )
            st.info(
                "Artifact hiện tại chưa có metric mở rộng và sai số theo nhóm. "
                "Chạy lại `python src/train_model.py` để tạo."
            )

        # Bảng chính: mỗi câu hỏi một chỉ số; phần còn lại nằm trong expander.
        rows = []
        for name, metrics in sorted(
            cv_summary.items(),
            key=lambda item: -item[1]["cv_r2_mean"],
        ):
            row = {
                "Mô hình": f"✅ {name}" if name == best_name else name,
                "CV R² (± std)": f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}",
                "Test R²": f"{metrics['test_r2']:.3f}",
                "Test MAE (tỷ)": f"{metrics['test_mae']:.2f}",
            }
            if "test_median_ape" in metrics:
                row["Median APE"] = f"{metrics['test_median_ape']:.2f}%"
                row["Lệch ≤ 20%"] = f"{metrics['test_hit_rate_20'] * 100:.2f}%"
            else:
                row["Test MAPE"] = f"{metrics['test_mape']:.2f}%"
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            alt="So sánh các mô hình theo CV R², Test R², MAE, Median APE và tỷ lệ lệch dưới 20%",
        )
        st.caption(
            "✅ là mô hình đang dùng, được chọn bằng CV trên tập train; tập test chỉ "
            "dùng để báo cáo kết quả cuối cùng."
        )

        with st.expander("Xem toàn bộ chỉ số", icon=":material/table_view:"):
            full_columns = [
                ("cv_r2_mean", "CV R² (mean)", "{:.3f}"),
                ("cv_r2_std", "CV R² (std)", "{:.3f}"),
                ("cv_mae", "CV MAE (tỷ)", "{:.2f}"),
                ("cv_mape", "CV MAPE (%)", "{:.2f}"),
                ("cv_median_ape", "CV Median APE (%)", "{:.2f}"),
                ("test_r2", "Test R²", "{:.3f}"),
                ("test_r2_log", "Test R² log", "{:.3f}"),
                ("test_mae", "Test MAE (tỷ)", "{:.2f}"),
                ("test_rmse", "Test RMSE (tỷ)", "{:.2f}"),
                ("test_rmsle", "Test RMSLE", "{:.3f}"),
                ("test_mape", "Test MAPE (%)", "{:.2f}"),
                ("test_median_ape", "Test Median APE (%)", "{:.2f}"),
                ("test_hit_rate_10", "Test lệch ≤ 10%", "pct"),
                ("test_hit_rate_20", "Test lệch ≤ 20%", "pct"),
                ("test_bias", "Test dự đoán / thực tế", "{:.2f}"),
            ]
            full_rows = []
            for name, metrics in sorted(
                cv_summary.items(),
                key=lambda item: -item[1]["cv_r2_mean"],
            ):
                row = {"Mô hình": f"✅ {name}" if name == best_name else name}
                for key, label, fmt in full_columns:
                    if key in metrics:
                        value = metrics[key]
                        row[label] = (
                            f"{value * 100:.2f}%" if fmt == "pct" else fmt.format(value)
                        )
                full_rows.append(row)
            st.dataframe(
                pd.DataFrame(full_rows),
                hide_index=True,
                alt="Toàn bộ chỉ số CV và test của từng mô hình",
            )
    else:
        st.info("Chưa có dữ liệu CV (model cũ). Train lại để xem.")

    segment_metrics = evaluation_report.get("segment_metrics")
    if segment_metrics:
        st.subheader(
            f"Sai số theo nhóm — {evaluation_report.get('segment_metrics_model', best_name)}",
            icon=":material/stacked_bar_chart:",
        )
        segment_tabs = st.tabs(["Khoảng giá thật", "Loại bất động sản", "Quận/huyện"])
        segment_sources = [
            ("price_band", "Khoảng giá"),
            ("property_type", "Loại BĐS"),
            ("district", "Quận/huyện"),
        ]
        for tab, (key, title) in zip(segment_tabs, segment_sources):
            with tab:
                records = segment_metrics.get(key, [])
                if not records:
                    st.caption("Không có nhóm nào đủ số mẫu để báo cáo.")
                    continue
                table = pd.DataFrame(records).rename(
                    columns={
                        "segment": title,
                        "n": "Số mẫu test",
                        "median_price": "Giá trung vị (tỷ)",
                        "mae": "MAE (tỷ)",
                        "median_ape": "Median APE (%)",
                        "hit_rate_20": "Lệch ≤ 20%",
                        "bias": "Dự đoán / thực tế",
                    }
                )
                st.dataframe(
                    table,
                    hide_index=True,
                    column_config={
                        "Số mẫu test": st.column_config.NumberColumn(format="%d"),
                        "Giá trung vị (tỷ)": st.column_config.NumberColumn(
                            format="%.2f"
                        ),
                        "MAE (tỷ)": st.column_config.NumberColumn(format="%.2f"),
                        "Median APE (%)": st.column_config.NumberColumn(format="%.2f"),
                        "Lệch ≤ 20%": st.column_config.NumberColumn(format="percent"),
                        "Dự đoán / thực tế": st.column_config.NumberColumn(
                            format="%.2f"
                        ),
                    },
                    alt=f"Sai số của mô hình theo {title.lower()}",
                )
        st.caption(
            "Chỉ hiển thị nhóm có từ 200 mẫu test trở lên; dùng để nhận biết "
            "mô hình còn yếu ở phân khúc nào."
        )

    if importance is not None:
        st.subheader(
            f"20 đặc trưng ảnh hưởng nhiều nhất — {best_name}",
            icon=":material/bar_chart:",
        )
        top20 = importance.head(20).reset_index()
        top20.columns = ["Đặc trưng", "Mức ảnh hưởng"]
        fig_imp = px.bar(
            top20,
            x="Mức ảnh hưởng",
            y="Đặc trưng",
            orientation="h",
            color="Mức ảnh hưởng",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(
            height=540,
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(
            fig_imp,
            alt="Mức ảnh hưởng của 20 đặc trưng quan trọng nhất",
        )

    with st.expander("Giải thích các chỉ số", icon=":material/info:"):
        st.markdown("""
| Chỉ số | Ý nghĩa |
|---|---|
| **Test R²** | Hệ số xác định trên tập test; càng cao càng tốt và có thể âm nếu mô hình kém hơn dự đoán bằng giá trị trung bình |
| **CV R²** | R² trung bình qua 5 fold — đánh giá độ ổn định của model |
| **CV R² std** | Độ lệch chuẩn R² qua các fold — thấp = model ổn định |
| **MAE (tỷ VNĐ)** | Sai số tuyệt đối trung bình (Mean Absolute Error) |
| **MAPE (%)** | Sai số phần trăm tuyệt đối trung bình trên giá gốc; càng thấp càng tốt, có thể vượt 100% và nhạy với giá thật rất nhỏ |
| **Median APE (%)** | Trung vị sai số phần trăm; ít bị vài dự đoán cực đoan kéo lệch hơn MAPE |
| **Lệch ≤ 10% / ≤ 20%** | Tỷ lệ căn có giá dự đoán lệch không quá 10% hoặc 20% |
| **R² log** | R² trên log1p(giá), cùng thang với target lúc huấn luyện |
| **Dự đoán / thực tế** | Trung vị tỷ lệ dự đoán trên thực tế; nhỏ hơn 1 nghĩa là thường dự đoán thấp |
""")
