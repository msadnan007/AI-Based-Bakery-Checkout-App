import os
import tempfile
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# -----------------------------
# Price table
# -----------------------------
PRICES = {
    "bic-ma-chee":            2300,
    "butter bread":           1500,
    "cheese bagel":           2300,
    "choco pain au raisin":   3500,
    "fried soboro":           2200,
    "kouign amann":           3500,
    "milk croquette":         3000,
    "tuna croquette":         3500,
}
DEFAULT_PRICE = 2500  # fallback for any class not in the table


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Sungsimdang AI Checkout System",
    page_icon="🥐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Modern styling
# -----------------------------
st.markdown(
    """
    <style>
        .main {background-color: #faf7f2;}
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}

        .hero-card {
            background: linear-gradient(135deg, #2b2118 0%, #7a4d2d 55%, #d99a5b 100%);
            padding: 32px;
            border-radius: 26px;
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 14px 35px rgba(43, 33, 24, 0.22);
        }
        .hero-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 6px;
            letter-spacing: -0.8px;
        }
        .hero-subtitle {
            font-size: 18px;
            opacity: 0.92;
            max-width: 780px;
        }

        .step-card {
            background: white;
            border: 1px solid #eee2d4;
            border-radius: 20px;
            padding: 18px;
            min-height: 128px;
            box-shadow: 0 8px 22px rgba(65, 45, 25, 0.07);
        }
        .step-number {
            display: inline-block;
            background: #f3d6b5;
            color: #4b2d18;
            font-weight: 800;
            padding: 4px 11px;
            border-radius: 999px;
            margin-bottom: 10px;
        }
        .step-title {
            font-size: 18px;
            font-weight: 800;
            color: #2b2118;
            margin-bottom: 4px;
        }
        .step-text {color: #6c5a4a; font-size: 14px;}

        .section-card {
            background: white;
            border: 1px solid #eee2d4;
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 8px 22px rgba(65, 45, 25, 0.07);
        }

        .status-success {
            background: #eaf7ee;
            color: #166534;
            border: 1px solid #bde7c8;
            padding: 12px 14px;
            border-radius: 16px;
            font-weight: 700;
        }
        .status-warn {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            padding: 12px 14px;
            border-radius: 16px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #eee2d4;
            padding: 15px;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(65, 45, 25, 0.06);
        }

        .menu-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid #f0e8df;
            font-size: 13px;
        }
        .menu-row:last-child { border-bottom: none; }
        .menu-name { color: #4b2d18; font-weight: 500; }
        .menu-price { color: #7a4d2d; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# App text/header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🥐 Sungsimdang AI Checkout System</div>
        <div class="hero-subtitle">Upload-Detect-Checkout </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar controls
# -----------------------------
DEFAULT_MODEL_PATH = r"my_model.pt"

with st.sidebar:
    st.header("⚙️ Detection Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    conf = st.slider("Confidence threshold", 0.10, 1.00, 0.25, 0.05)
    imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 768, 1024], value=640)

    st.divider()

    # Real price menu in sidebar
    st.header("💸 Menu & Prices")
    menu_html = ""
    for item, price in PRICES.items():
        menu_html += (
            f'<div class="menu-row">'
            f'<span class="menu-name">{item.title()}</span>'
            f'<span class="menu-price">₩{price:,}</span>'
            f'</div>'
        )
    st.markdown(menu_html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return YOLO(path)


# -----------------------------
# Process cards
# -----------------------------
st.subheader("Checkout Process")
step_cols = st.columns(4)
steps = [
    ("1", "Upload Tray", "Cashier/customer takes one tray photo."),
    ("2", "AI Detection", "YOLO finds each bakery item on the tray."),
    ("3", "Review Basket", "Detected items become an editable checkout list."),
    ("4", "Fast Payment", "POS receives item names, quantities, and total."),
]

for col, (num, title, text) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-number">Step {num}</span>
                <div class="step-title">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# -----------------------------
# Model loading
# -----------------------------
try:
    model = load_model(model_path)
    model_ready = True
except Exception as e:
    model_ready = False
    st.error("Model could not be loaded. Check that your path points directly to a .pt file.")
    st.exception(e)


# -----------------------------
# Upload and prediction
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload bakery tray image",
    type=["jpg", "jpeg", "png"],
    disabled=not model_ready,
)

if uploaded_file and model_ready:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Original Tray Image")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image.save(temp.name)
            temp_path = temp.name

        with st.spinner("Detecting bakery items..."):
            results = model.predict(
                source=temp_path,
                conf=conf,
                imgsz=imgsz,
                save=False,
                verbose=False,
            )

        result = results[0]
        result_img = result.plot()
        boxes = result.boxes

        detected_rows = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                item_name = model.names[cls_id]
                confidence = float(box.conf[0])
                unit_price = PRICES.get(item_name, DEFAULT_PRICE)
                detected_rows.append({
                    "Item": item_name,
                    "Confidence": confidence,
                    "Unit Price": unit_price,
                })

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("AI Detection Result")
            st.image(result_img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        if detected_rows:
            total_items = len(detected_rows)
            unique_items = len(set(row["Item"] for row in detected_rows))
            avg_conf = sum(row["Confidence"] for row in detected_rows) / total_items
            real_total = sum(row["Unit Price"] for row in detected_rows)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detected Items", total_items)
            m2.metric("Unique Products", unique_items)
            m3.metric("Avg. Confidence", f"{avg_conf:.0%}")
            m4.metric("Total", f"₩{real_total:,.0f}")

            st.markdown(
                '<div class="status-success">✅ Checkout basket created successfully. Cashier can now review and confirm.</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            # Build basket with real per-item prices
            counts = Counter(row["Item"] for row in detected_rows)
            basket_df = pd.DataFrame(
                [
                    {
                        "Bakery Item": item,
                        "Quantity": qty,
                        "Unit Price (₩)": PRICES.get(item, DEFAULT_PRICE),
                        "Subtotal (₩)": qty * PRICES.get(item, DEFAULT_PRICE),
                    }
                    for item, qty in counts.items()
                ]
            )

            detail_df = pd.DataFrame([
                {"Item": r["Item"], "Confidence": f"{r['Confidence']:.2%}"}
                for r in detected_rows
            ])

            checkout_col, detail_col = st.columns([1.2, 1], gap="large")

            with checkout_col:
                st.subheader("🧾 AI Checkout Basket")
                edited_basket = st.data_editor(
                    basket_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={
                        "Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                        "Unit Price (₩)": st.column_config.NumberColumn(min_value=0, step=100),
                        "Subtotal (₩)": st.column_config.NumberColumn(disabled=True),
                    },
                )

                recalculated_total = int(
                    (edited_basket["Quantity"] * edited_basket["Unit Price (₩)"]).sum()
                )
                st.success(f"Final payment total: ₩{recalculated_total:,.0f}")

            with detail_col:
                st.subheader("🔍 Detection Details")
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

        else:
            st.markdown(
                '<div class="status-warn">⚠️ No bakery items were detected. Try lowering the confidence threshold or using a clearer tray image.</div>',
                unsafe_allow_html=True,
            )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("Upload a tray image to start the AI checkout demo.")
