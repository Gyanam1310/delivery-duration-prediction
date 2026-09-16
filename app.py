"""
app.py — Delivery Duration Prediction
======================================
Streamlit web application that predicts food delivery duration using a
Linear Regression model trained on the DoorDash historical delivery dataset.

Run locally:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Delivery Duration Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }

    /* Main container card */
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        border-left: 4px solid #7c3aed;
        padding-left: 0.75rem;
    }

    /* Hero title */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f9a8d4, #c084fc, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.55);
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3), rgba(236, 72, 153, 0.2));
        border: 1px solid rgba(167, 139, 250, 0.4);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(124, 58, 237, 0.2);
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(124, 58, 237, 0.2); }
        to   { box-shadow: 0 0 50px rgba(236, 72, 153, 0.35); }
    }

    .result-number {
        font-size: 4.5rem;
        font-weight: 800;
        color: #fff;
        line-height: 1;
    }

    .result-unit {
        font-size: 1.4rem;
        font-weight: 500;
        color: #c4b5fd;
        margin-top: 0.25rem;
    }

    .result-seconds {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.45);
        margin-top: 0.5rem;
    }

    /* Info badge */
    .info-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        color: #a5b4fc;
        font-weight: 500;
    }

    /* Override Streamlit widget labels */
    .stSelectbox label, .stSlider label, .stNumberInput label, .stToggle label {
        color: rgba(255,255,255,0.8) !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 2.5rem !important;
        width: 100% !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
        letter-spacing: 0.03em !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6) !important;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
    }

    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #c4b5fd;
    }

    .metric-card .metric-label {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.5);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }

    /* Warning box */
    .stAlert {
        border-radius: 12px !important;
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# CONSTANTS – must match notebook exactly
# ──────────────────────────────────────────────────────────────
LOG1P_COLS = [
    "subtotal",
    "total_items",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "estimated_store_to_consumer_driving_duration",
]

ARTIFACTS_DIR = "model_artifacts"

# ──────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model  = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    meta   = joblib.load(os.path.join(ARTIFACTS_DIR, "ohe_meta.pkl"))
    return model, scaler, meta


def artifacts_exist() -> bool:
    return all(
        os.path.exists(os.path.join(ARTIFACTS_DIR, f))
        for f in ["model.pkl", "scaler.pkl", "ohe_meta.pkl"]
    )


# ──────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ──────────────────────────────────────────────────────────────
def preprocess_input(raw: dict, feature_columns: list) -> pd.DataFrame:
    """
    Apply the EXACT notebook transformations to a single-row input dict
    and return a DataFrame aligned with the training feature columns.
    """
    df = pd.DataFrame([raw])

    # 1. log1p transform (same cols as notebook cell 34)
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].astype(float))

    # 2. OHE store_primary_category (drop_first=True → drop alphabetically first)
    df = pd.get_dummies(df, columns=["store_primary_category"], drop_first=True)

    # 3. OHE order_protocol (drop_first=True)
    df = pd.get_dummies(df, columns=["order_protocol"], drop_first=True)

    # 4. market_id → category → OHE (drop_first=True)
    df["market_id"] = df["market_id"].astype("category")
    df = pd.get_dummies(df, columns=["market_id"], drop_first=True)

    # 5. Align columns with training feature set
    #    (add any missing OHE columns as 0, remove any extras)
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


def predict_duration(raw: dict) -> float:
    model, scaler, meta = load_artifacts()
    feature_columns = meta["feature_columns"]

    X = preprocess_input(raw, feature_columns)
    X_scaled = scaler.transform(X)
    duration_seconds = model.predict(X_scaled)[0]
    return max(0.0, duration_seconds)   # clamp at 0


# ──────────────────────────────────────────────────────────────
# SIDEBAR – info panel
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:1.5rem;'>
            <div style='font-size:3rem;'>🍔</div>
            <div style='font-size:1.2rem; font-weight:700; color:#c4b5fd;'>Delivery Predictor</div>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.45);'>ML-Powered Estimation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("#### 📊 Model Info")
    st.markdown(
        """
        <div class='metric-card' style='margin-bottom:0.6rem;'>
            <div class='metric-value'>Linear</div>
            <div class='metric-label'>Model Type</div>
        </div>
        <div class='metric-card' style='margin-bottom:0.6rem;'>
            <div class='metric-value'>~11.7 min</div>
            <div class='metric-label'>Test MAE</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>R² ≈ 0.23</div>
            <div class='metric-label'>Test Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown(
            """
            This app uses a **Linear Regression** model trained on DoorDash's
            historical delivery dataset (~180k orders).

            **Preprocessing applied:**
            - Missing dasher values imputed by market median
            - Log₁⁺ transform on skewed price/item features
            - One-hot encoding for category, protocol & market
            - StandardScaler normalization

            **Target:** Total delivery duration in seconds.
            """
        )

    st.markdown("---")
    st.markdown(
        "<div style='color:rgba(255,255,255,0.3); font-size:0.73rem; text-align:center;'>"
        "Delivery Duration Prediction · ML Project"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────
st.markdown(
    "<div class='hero-title'>🚀 Delivery Duration Predictor</div>"
    "<div class='hero-subtitle'>Enter order details below to estimate how long your delivery will take.</div>",
    unsafe_allow_html=True,
)

# ── Check model artifacts ────────────────────────────────────
if not artifacts_exist():
    st.error(
        "⚠️ **Model artifacts not found.**\n\n"
        "Please run the training script first:\n"
        "```bash\npython train_model.py\n```\n"
        "This will generate the required files in `model_artifacts/`."
    )
    st.stop()

# ── Load artifacts once ───────────────────────────────────────
model, scaler, meta = load_artifacts()
feature_columns = meta["feature_columns"]

# ── Extract known categories from feature_columns ────────────
# e.g. 'store_primary_category_mexican' → 'mexican'
known_categories = sorted(set(
    col.replace("store_primary_category_", "")
    for col in feature_columns
    if col.startswith("store_primary_category_")
))

known_protocols = sorted(set(
    float(col.replace("order_protocol_", ""))
    for col in feature_columns
    if col.startswith("order_protocol_")
))

known_markets = sorted(set(
    float(col.replace("market_id_", ""))
    for col in feature_columns
    if col.startswith("market_id_")
))

# Add the dropped-first category back for display (user can still select it)
ALL_CATEGORIES = ["(drop_first baseline)"] + known_categories  # first cat was dropped in OHE
ALL_PROTOCOLS  = [1.0] + known_protocols    # 1.0 was dropped first
ALL_MARKETS    = [1.0] + known_markets      # 1.0 was dropped first

# Friendly label for categories: find the dropped-first one from all categories
# We show ALL options including the reference category
ALL_STORE_CATS = (
    ["american", "Unknown"] +
    [c for c in known_categories if c not in ("american", "Unknown")]
) if known_categories else ["american", "Unknown", "mexican", "asian", "italian", "fast-food"]

# ──────────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    # ── Section 1: Order Information ─────────────────────────
    st.markdown("<div class='section-header'>📦 Order Details</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        total_items = st.number_input(
            "Total Items",
            min_value=1, max_value=50, value=3, step=1,
            help="Total number of food items in the order",
        )
        subtotal = st.number_input(
            "Subtotal (in cents)",
            min_value=100, max_value=100000, value=2500, step=100,
            help="Order subtotal in cents (e.g. $25.00 = 2500)",
        )

    with col2:
        num_distinct_items = st.number_input(
            "Distinct Items",
            min_value=1, max_value=30, value=2, step=1,
            help="Number of unique menu items in the order",
        )
        min_item_price = st.number_input(
            "Min Item Price (cents)",
            min_value=0, max_value=50000, value=800, step=100,
            help="Price of the cheapest item in the order (cents)",
        )

    with col3:
        max_item_price = st.number_input(
            "Max Item Price (cents)",
            min_value=0, max_value=100000, value=1500, step=100,
            help="Price of the most expensive item in the order (cents)",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Store & Market Info ───────────────────────
    st.markdown("<div class='section-header'>🏪 Store & Market</div>", unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        store_primary_category = st.selectbox(
            "Store Category",
            options=ALL_STORE_CATS,
            index=0,
            help="Type of restaurant/cuisine",
        )

    with col5:
        order_protocol = st.selectbox(
            "Order Protocol",
            options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            index=0,
            help="How the restaurant receives orders (1=manual, higher=automated)",
        )

    with col6:
        market_id = st.selectbox(
            "Market ID",
            options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            index=0,
            help="Geographic market / city region",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Dasher Availability ───────────────────────
    st.markdown("<div class='section-header'>🛵 Dasher Availability</div>", unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)

    with col7:
        total_onshift_dashers = st.number_input(
            "Dashers On-Shift",
            min_value=0, max_value=300, value=20, step=1,
            help="Total delivery drivers available in the region right now",
        )

    with col8:
        total_busy_dashers = st.number_input(
            "Dashers Currently Busy",
            min_value=0, max_value=300, value=10, step=1,
            help="Drivers currently on active deliveries",
        )

    with col9:
        total_outstanding_orders = st.number_input(
            "Outstanding Orders",
            min_value=0, max_value=500, value=15, step=1,
            help="Total pending delivery orders in the system right now",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: Platform Estimates & Time ─────────────────
    st.markdown("<div class='section-header'>⏱️ Platform Estimates & Timing</div>", unsafe_allow_html=True)
    col10, col11, col12, col13 = st.columns(4)

    with col10:
        estimated_order_place_duration = st.number_input(
            "Est. Order Place Duration (s)",
            min_value=0, max_value=3600, value=446, step=10,
            help="Estimated seconds between order placement and restaurant confirmation",
        )

    with col11:
        estimated_store_to_consumer_driving_duration = st.number_input(
            "Est. Driving Duration (s)",
            min_value=0, max_value=7200, value=700, step=50,
            help="Estimated driving time from restaurant to customer (seconds)",
        )

    with col12:
        hour = st.slider(
            "Order Hour (0–23)",
            min_value=0, max_value=23, value=18,
            help="Hour of the day when the order is placed",
        )

    with col13:
        is_weekend = st.toggle(
            "Weekend Order?",
            value=False,
            help="Is the order placed on a Saturday or Sunday?",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Submit Button ─────────────────────────────────────────
    submitted = st.form_submit_button("🔮 Predict Delivery Duration", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# PREDICTION OUTPUT
# ──────────────────────────────────────────────────────────────
if submitted:
    raw_input = {
        "market_id": market_id,
        "store_primary_category": store_primary_category,
        "order_protocol": order_protocol,
        "total_items": float(total_items),
        "subtotal": float(subtotal),
        "num_distinct_items": float(num_distinct_items),
        "min_item_price": float(min_item_price),
        "max_item_price": float(max_item_price),
        "total_onshift_dashers": float(total_onshift_dashers),
        "total_busy_dashers": float(total_busy_dashers),
        "total_outstanding_orders": float(total_outstanding_orders),
        "estimated_order_place_duration": float(estimated_order_place_duration),
        "estimated_store_to_consumer_driving_duration": float(
            estimated_store_to_consumer_driving_duration
        ),
        "hour": float(hour),
        "is_weekend": int(is_weekend),
    }

    with st.spinner("Calculating your delivery time..."):
        try:
            duration_seconds = predict_duration(raw_input)
            duration_minutes = duration_seconds / 60
            hours_part   = int(duration_minutes // 60)
            minutes_part = int(duration_minutes % 60)
            seconds_part = int(duration_seconds % 60)

            # Format display
            if hours_part > 0:
                display_time = f"{hours_part}h {minutes_part}m"
            else:
                display_time = f"{minutes_part}m {seconds_part}s"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='result-box'>
                    <div style='font-size:1rem; color:rgba(255,255,255,0.6); margin-bottom:0.5rem;
                                text-transform:uppercase; letter-spacing:0.1em; font-weight:600;'>
                        🎯 Estimated Delivery Duration
                    </div>
                    <div class='result-number'>{display_time}</div>
                    <div class='result-unit'>from order placement to doorstep</div>
                    <div class='result-seconds'>({duration_seconds:,.0f} seconds · {duration_minutes:.1f} minutes)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Breakdown metrics ─────────────────────────────
            st.markdown("<div class='section-header'>📈 Prediction Breakdown</div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("⏱ Total Duration", f"{duration_minutes:.1f} min")
            with m2:
                busy_ratio = (total_busy_dashers / max(total_onshift_dashers, 1)) * 100
                st.metric("🛵 Dasher Utilization", f"{busy_ratio:.0f}%")
            with m3:
                drive_min = estimated_store_to_consumer_driving_duration / 60
                st.metric("🚗 Drive Estimate", f"{drive_min:.1f} min")
            with m4:
                wait_min = max(0, duration_minutes - drive_min)
                st.metric("⏳ Prep + Wait Est.", f"{wait_min:.1f} min")

            if duration_minutes > 60:
                st.warning(
                    "⚠️ The predicted duration is over 60 minutes. "
                    "High dasher utilization or long driving distances may be causing delays."
                )
            elif duration_minutes < 15:
                st.info("✅ Great! This looks like a fast delivery order.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
