import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))
# EDA dùng toàn bộ dòng hợp lệ; chỉ các tập train mới được lọc ngoại lệ.
from cleaning import clean_listings
from data_loader import load_raw_data
from artifacts import load_artifacts
from config import ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH
from cleaning import deduplicate

# Config
st.set_page_config(
    page_title="Dự đoán Giá Nhà Hà Nội",
    page_icon=r"C:\Users\DELL\Downloads\page-icon.png",
    layout="wide",
)


# Load model & data
@st.cache_resource(max_entries=1)
def load_bundle(model_modified_ns, preprocessor_modified_ns):
    """Cache model và tự làm mới khi file model thay đổi."""
    return load_artifacts(ARTIFACTS_DIR)


@st.cache_data(show_spinner="Đang tải dữ liệu...", max_entries=2)
def load_data(sample_size=None) -> pd.DataFrame:
    # load_raw_data() tự đọc 10 parquet và lọc Hà Nội
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
bundle = load_bundle(
    MODEL_PATH.stat().st_mtime_ns, PREPROCESSOR_PATH.stat().st_mtime_ns
)
model = bundle["model"]
feature_cols = bundle["feature_cols"]
preprocessor = bundle["preprocessor"]
districts = bundle["districts"]  # {district_name: median_price_ty}
ward_medians = bundle.get("ward_medians", {})
wards_by_district = bundle.get("wards_by_district", {})
global_median = bundle["global_median_price"]
best_name = bundle["best_model_name"]
best_mae = bundle["best_model_mae"]
cv_summary = bundle.get("cv_summary", {})
importance = bundle.get("feature_importance")

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
    df = load_data(50000)
    df = df[df["property_type_name"] != "Đất"].copy()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mẫu dữ liệu minh họa", f"{len(df):,}")
    col2.metric("Giá trung vị", f"{df['price_ty'].median():.1f} tỷ")
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
    st.subheader("Nhập thông tin căn nhà")

    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox("Quận/Huyện", district_list)
        ward = st.selectbox(
            "Phường/Xã",
            ["Không rõ", *wards_by_district.get(district, [])],
            key=f"ward_{district}",
            help="Chọn “Không rõ” để dùng mức giá trung vị của quận/huyện.",
        )
        prop_type = st.selectbox(
            "Loại BĐS",
            ["Nhà", "Biệt thự/Nhà liền kề", "Shophouse", "Căn hộ chung cư"],
        )
        area = st.number_input(
            "Diện tích (m²)", min_value=10.0, max_value=2000.0, value=60.0, step=5.0
        )

    with col2:
        bedroom_count = st.number_input(
            "Số phòng ngủ", min_value=0, max_value=20, value=3, step=1
        )
        bathroom_count = st.number_input(
            "Số phòng tắm", min_value=0, max_value=20, value=2, step=1
        )
        floor_count = st.number_input(
            "Số tầng", min_value=1, max_value=50, value=3, step=1
        )
        frontage_width = st.number_input(
            "Chiều rộng mặt tiền (m)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.5,
        )

    if st.button("Dự đoán giá", type="primary"):
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

        # Cảnh báo quận lạ
        if district not in districts:
            st.warning(
                f"Quận {district} không có trong training data — "
                "kết quả có thể kém chính xác."
            )

        if not np.isfinite(pred) or pred <= 0:
            st.warning(
                "Chưa ước lượng được mức giá hợp lệ cho tổ hợp thông tin này. "
                "Bạn có thể kiểm tra lại thông tin; nếu thông tin đã đúng, "
                "mô hình hiện chưa đáp ứng trường hợp này."
            )
        else:
            lo = max(0, pred - best_mae)
            hi = pred + best_mae
            st.success(
                f"Giá dự đoán: {pred:.2f} tỷ VNĐ\n\n"
                f"Biên sai số tham khảo ± MAE: {lo:.2f} – {hi:.2f} tỷ VNĐ"
            )
            st.caption(
                "Biên này không phải khoảng tin cậy và không bảo đảm mức bao phủ 90% hay 95%."
            )
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Dự đoán", f"{pred:.2f} tỷ")
            col_r2.metric("Cận tham khảo dưới", f"{lo:.2f} tỷ")
            col_r3.metric("Cận tham khảo trên", f"{hi:.2f} tỷ")
            st.info(
                f"Model sử dụng: {best_name} | MAE = {best_mae:.2f} tỷ VNĐ"
                f"(≈ {best_mae*1000:.0f} triệu VNĐ)"
            )


# TAB 3: ĐÁNH GIÁ MÔ HÌNH
with tab3:
    st.subheader("So sánh hiệu quả các mô hình (5-Fold CV)")
    artifact_version = bundle.get("artifact_version", {})
    if artifact_version:
        st.caption(
            f"Phiên bản kết quả: {artifact_version['version_id']} | "
            f"Ngày lưu (UTC): {artifact_version['created_at_utc']}"
        )

    if cv_summary:
        rows = []
        for name, metrics in cv_summary.items():
            rows.append(
                {
                    "Mô hình": name,
                    "Test R²": f"{metrics['test_r2']:.3f}",
                    "CV R² (mean)": f"{metrics['cv_r2_mean']:.3f}",
                    "CV R² (std)": f"±{metrics['cv_r2_std']:.3f}",
                    "CV MAE (tỷ)": f"{metrics['cv_mae']:.3f}",
                    "Test MAE (tỷ)": f"{metrics['test_mae']:.3f}",
                    "Best": "✅" if name == best_name else "",
                }
            )
        st.dataframe(pd.DataFrame(rows))
    else:
        st.info("Chưa có dữ liệu CV (model cũ). Train lại để xem.")

    st.divider()

    if importance is not None:
        st.subheader(f"Top 20 Feature Importance — {best_name}")
        top20 = importance.head(20).reset_index()
        top20.columns = ["Feature", "Importance"]
        fig_imp = px.bar(
            top20,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_imp.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_imp)

    st.divider()
    st.subheader("Giải thích chỉ số")
    st.markdown("""
| Chỉ số | Ý nghĩa |
|---|---|
| **Test R²** | Hệ số xác định trên tập test; càng cao càng tốt và có thể âm nếu mô hình kém hơn dự đoán bằng giá trị trung bình |
| **CV R²** | R² trung bình qua 5 fold — đánh giá độ ổn định của model |
| **CV R² std** | Độ lệch chuẩn R² qua các fold — thấp = model ổn định |
| **MAE (tỷ VNĐ)** | Sai số tuyệt đối trung bình (Mean Absolute Error) |
""")
