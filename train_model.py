"""
train_model.py
==============
Reproduces the EXACT preprocessing pipeline and model training from
Delivery_Duration.ipynb and saves all artifacts to model_artifacts/.

Run this script ONCE before launching the Streamlit app:
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_URL = "https://new-assets.ccbp.in/frontend/content/aiml/classical-ml/historical_data.csv"
LOCAL_CSV = "delivery_dataset.csv"          # preferred (faster, no network needed)
ARTIFACTS_DIR = "model_artifacts"

# Columns that receive log1p transformation (from notebook cell 34)
LOG1P_COLS = [
    "subtotal",
    "total_items",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "estimated_store_to_consumer_driving_duration",
]


def load_data() -> pd.DataFrame:
    """Load the dataset from local file or URL (notebook cell 1)."""
    if os.path.exists(LOCAL_CSV):
        print(f"[INFO] Loading data from local file: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV)
    else:
        print(f"[INFO] Local file not found. Downloading from URL...")
        df = pd.read_csv(DATA_URL)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact preprocessing pipeline from the notebook.

    Notebook cells (in order):
      5  → parse datetimes
      10 → drop NaN rows, engineer target + time features
      18 → impute dasher columns with per-market_id median
      19 → fill store_primary_category NaN with 'Unknown'
      26 → filter delivery_duration <= 21600
      32 → filter min_item_price >= 0
      34 → log1p transform on skewed numeric columns
      42 → OHE store_primary_category (drop_first=True)
      43 → OHE order_protocol (drop_first=True)
      45 → market_id → category → OHE (drop_first=True)
      47 → drop created_at, actual_delivery_time, order_date
      Note: store_id is NOT dropped inplace in cell 46 (no assignment),
            so it remains. We drop it here explicitly to match training X.
    """

    # Cell 5: parse timestamps
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])

    # Cell 10: drop NaN rows
    df = df.dropna(
        subset=[
            "actual_delivery_time",
            "market_id",
            "estimated_store_to_consumer_driving_duration",
            "order_protocol",
        ]
    )

    # Cell 10: engineer features
    df["delivery_duration"] = (
        df["actual_delivery_time"] - df["created_at"]
    ).dt.total_seconds()
    df["hour"] = df["created_at"].dt.hour
    df["order_date"] = df["created_at"].dt.date
    df["is_weekend"] = (df["created_at"].dt.dayofweek >= 5).astype(int)
    df.drop(["actual_delivery_time", "created_at"], axis=1, inplace=True)

    # Cell 18: impute dasher cols with per-market_id median
    for col in ["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]:
        df[col] = df[col].fillna(
            df.groupby("market_id")[col].transform("median")
        )

    # Cell 19: fill store_primary_category NaN
    df["store_primary_category"] = df["store_primary_category"].fillna("Unknown")

    # Cell 26: outlier filter on target
    df = df[df["delivery_duration"] <= 21600].copy()

    # Cell 32: filter negative min_item_price
    df = df[df["min_item_price"] >= 0]

    # Cell 34: log1p transform
    for col in LOG1P_COLS:
        df[col] = np.log1p(df[col])

    # Cell 42: OHE store_primary_category
    df = pd.get_dummies(df, columns=["store_primary_category"], drop_first=True)

    # Cell 43: OHE order_protocol
    df = pd.get_dummies(df, columns=["order_protocol"], drop_first=True)

    # Cell 45: market_id → category → OHE
    df["market_id"] = df["market_id"].astype("category")
    df = pd.get_dummies(df, columns=["market_id"], drop_first=True)

    # Cell 47: drop date columns
    df = df.drop(
        columns=["created_at", "actual_delivery_time", "order_date"],
        errors="ignore",
    )

    # Drop store_id (not used in training; cell 46 had no inplace assignment,
    # but store_id has no predictive meaning and was clearly intended to be excluded)
    df = df.drop(columns=["store_id"], errors="ignore")

    return df


def train_and_save():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. Load & preprocess ──────────────────────────────────────
    print("[INFO] Loading data...")
    df = load_data()

    print("[INFO] Preprocessing...")
    df = preprocess(df)

    # ── 2. Split features / target ────────────────────────────────
    X = df.drop("delivery_duration", axis=1)
    y = df["delivery_duration"]

    print(f"[INFO] Dataset shape after preprocessing: {df.shape}")
    print(f"[INFO] Feature columns ({len(X.columns)}): {X.columns.tolist()}")

    # ── 3. Train / test split (same as notebook: shuffle=False) ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ── 4. Scale ──────────────────────────────────────────────────
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # ── 5. Train Linear Regression ────────────────────────────────
    print("[INFO] Training Linear Regression...")
    lr = LinearRegression(n_jobs=-1)
    lr.fit(X_train_scaled, y_train)

    # ── 6. Quick evaluation ───────────────────────────────────────
    from sklearn.metrics import mean_absolute_error, r2_score
    y_pred = lr.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[INFO] Test MAE : {mae:.1f} seconds  ({mae/60:.1f} min)")
    print(f"[INFO] Test R²  : {r2:.4f}")

    # ── 7. Save artifacts ─────────────────────────────────────────
    feature_columns = X.columns.tolist()

    # Collect the OHE category values so the app can reconstruct columns
    # We need: which store categories, order protocols, and market_ids exist
    ohe_meta = {
        "log1p_cols": LOG1P_COLS,
        "feature_columns": feature_columns,
    }

    joblib.dump(lr, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(sc, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(ohe_meta, os.path.join(ARTIFACTS_DIR, "ohe_meta.pkl"))

    print(f"[INFO] Artifacts saved to '{ARTIFACTS_DIR}/'")
    print("       model.pkl, scaler.pkl, ohe_meta.pkl")


if __name__ == "__main__":
    train_and_save()
