"""
seller_risk_model.py  (point-in-time / leakage-free)
====================================================
Seller-churn EARLY-WARNING model for the marketplace.

Why point-in-time?
------------------
The SQL output CSVs are all anchored to the dataset's last date (2024-06-30).
By then every churned seller already reads as zero orders, so any model trained
on them just learns "orders == 0 => churned" (label leakage, AUC ~1.0, but
useless for early warning). To fix this we IGNORE the pre-aggregated outputs and
compute every feature directly from the raw event tables, time-bounded to a
CUTOFF date that acts as "now":

    CUTOFF = 2024-03-31
    - features  : built ONLY from orders/returns on or before the cutoff
    - label     : ground-truth is_active, restricted to sellers still
                  transacting at the cutoff (>=1 order in trailing 90d)

That restriction is the key: it drops sellers who already fully churned, so the
positive class becomes *declining-but-still-active* sellers — exactly the
population a marketplace team wants flagged before they leave.

STEP 1  Feature engineering (point-in-time, from raw events)
STEP 2  Model training (XGBoost, stratified split, SMOTE, 5-fold CV)
STEP 3  SHAP explanation (top-10 drivers, #1 churn driver)
STEP 4  Score active sellers (out-of-fold P(churn), risk tier, action)

Run:  python seller_risk_model.py
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "data"
OUT_DIR = "outputs"
SEED = 42
np.random.seed(SEED)

# CUTOFF = the "as-of / prediction" date. Features look back from here; the
# label looks at whether the seller ultimately churned. 2024-03-31 leaves a
# 3-month forward window (Apr-Jun) in which the post-cutoff decline plays out.
CUTOFF = pd.Timestamp("2024-03-31")


# ============================================================================
# STEP 1 — FEATURE ENGINEERING (point-in-time, from raw events)
# ============================================================================
def linreg_slope_rows(wide_df):
    """Vectorised OLS slope across the columns of a wide month matrix.
    Negative slope = declining trajectory (pre-churn signal)."""
    y = wide_df.to_numpy(dtype=float)
    n = y.shape[1]
    if n < 2:
        return np.zeros(y.shape[0])
    x = np.arange(n)
    x_mean = x.mean()
    # slope = cov(x,y)/var(x), computed row-wise.
    denom = ((x - x_mean) ** 2).sum()
    num = ((x - x_mean) * (y - y.mean(axis=1, keepdims=True))).sum(axis=1)
    return num / denom


def build_features_asof(anchor):
    """Compute the full feature matrix for every seller using only data on or
    before `anchor`. Returns (df, feature_cols)."""
    sellers = pd.read_csv(os.path.join(DATA_DIR, "sellers.csv"), parse_dates=["join_date"])
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["order_date"])
    returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), parse_dates=["return_date"])
    listings = pd.read_csv(os.path.join(DATA_DIR, "product_listings.csv"))

    # Hard time cut: nothing after the anchor may touch a feature.
    o = orders[orders["order_date"] <= anchor].copy()
    r = returns[returns["return_date"] <= anchor].copy()

    def window(df, col, days):
        lo = anchor - pd.Timedelta(days=days)
        return df[(df[col] > lo) & (df[col] <= anchor)]

    df = sellers[["seller_id", "category", "account_tier", "join_date", "is_active"]].copy()

    # --- Rolling GMV / order-count windows ------------------------------------
    o30, o60, o90 = window(o, "order_date", 30), window(o, "order_date", 60), window(o, "order_date", 90)
    df = df.merge(o30.groupby("seller_id")["gmv"].sum().rename("gmv_30d"), on="seller_id", how="left")
    df = df.merge(o60.groupby("seller_id")["gmv"].sum().rename("gmv_60d"), on="seller_id", how="left")
    df = df.merge(o90.groupby("seller_id")["gmv"].sum().rename("gmv_90d"), on="seller_id", how="left")
    df = df.merge(o30.groupby("seller_id").size().rename("orders_30d"), on="seller_id", how="left")
    # orders in trailing 90d — also used as the "active at anchor" gate.
    orders_90d = o90.groupby("seller_id").size().rename("orders_90d")
    df = df.merge(orders_90d, on="seller_id", how="left")

    # --- Month-over-month GMV change % (last 30d vs the 30d before that) -------
    prev_lo, prev_hi = anchor - pd.Timedelta(days=60), anchor - pd.Timedelta(days=30)
    o_prev = o[(o["order_date"] > prev_lo) & (o["order_date"] <= prev_hi)]
    gmv_prev = o_prev.groupby("seller_id")["gmv"].sum().rename("gmv_prev30")
    df = df.merge(gmv_prev, on="seller_id", how="left")
    df["gmv_30d"] = df["gmv_30d"].fillna(0.0)
    df["gmv_prev30"] = df["gmv_prev30"].fillna(0.0)
    df["gmv_mom_change"] = np.where(
        df["gmv_prev30"] > 0,
        100.0 * (df["gmv_30d"] - df["gmv_prev30"]) / df["gmv_prev30"],
        0.0,
    )

    # --- 6-month GMV & order-count trend slopes -------------------------------
    months = pd.period_range(end=anchor.to_period("M"), periods=6)
    o["ym"] = o["order_date"].dt.to_period("M")
    o6 = o[o["ym"].isin(months)]
    gmv_wide = (o6.groupby(["seller_id", "ym"])["gmv"].sum()
                  .unstack("ym").reindex(columns=months).fillna(0.0))
    cnt_wide = (o6.groupby(["seller_id", "ym"]).size()
                  .unstack("ym").reindex(columns=months).fillna(0.0))
    slope_df = pd.DataFrame({
        "seller_id": gmv_wide.index,
        "gmv_trend_slope": linreg_slope_rows(gmv_wide),
        "order_trend_slope": linreg_slope_rows(cnt_wide),  # real order counts now
    })
    df = df.merge(slope_df, on="seller_id", how="left")

    # --- Return features (trailing 90d, leak-free via return_date <= anchor) ---
    r90 = window(r, "return_date", 90)
    ret_cnt = r90.groupby("seller_id").size().rename("ret_cnt")
    qual_cnt = (r90[r90["return_reason"] == "quality issue"]
                .groupby("seller_id").size().rename("qual_cnt"))
    df = df.merge(ret_cnt, on="seller_id", how="left").merge(qual_cnt, on="seller_id", how="left")
    df["ret_cnt"] = df["ret_cnt"].fillna(0.0)
    df["qual_cnt"] = df["qual_cnt"].fillna(0.0)
    df["orders_90d"] = df["orders_90d"].fillna(0.0)
    df["return_rate"] = np.where(df["orders_90d"] > 0, 100.0 * df["ret_cnt"] / df["orders_90d"], 0.0)
    # quality_return_pct: quality issues as a share of this seller's returns.
    # Stronger churn signal than size issues (product/supplier problem vs a
    # fixable size chart).
    df["quality_return_pct"] = np.where(df["ret_cnt"] > 0, 100.0 * df["qual_cnt"] / df["ret_cnt"], 0.0)

    # --- Catalogue / pricing (current snapshot) -------------------------------
    listings = listings.copy()
    listings["above"] = (listings["listed_price"] > listings["competitor_price"]).astype(int)
    listings["gap_pct"] = (100.0 * (listings["listed_price"] - listings["competitor_price"])
                           / listings["competitor_price"])
    cat = listings.groupby("seller_id").agg(
        catalogue_size=("listing_id", "count"),
        price_above_competitor_pct=("above", lambda s: 100.0 * s.mean()),
        avg_price_gap_pct=("gap_pct", "mean"),
    ).reset_index()
    df = df.merge(cat, on="seller_id", how="left")

    # --- Tenure ---------------------------------------------------------------
    df["account_tenure_days"] = (anchor - df["join_date"]).dt.days

    # --- Tier + category dummies ----------------------------------------------
    df["is_gold"] = (df["account_tier"] == "Gold").astype(int)
    df["is_silver"] = (df["account_tier"] == "Silver").astype(int)
    df["is_bronze"] = (df["account_tier"] == "Bronze").astype(int)
    cat_dummies = pd.get_dummies(df["category"], prefix="cat").astype(int)
    df = pd.concat([df, cat_dummies], axis=1)

    feature_cols = [
        "gmv_30d", "gmv_60d", "gmv_90d", "gmv_mom_change", "gmv_trend_slope",
        "return_rate", "quality_return_pct", "orders_30d", "order_trend_slope",
        "price_above_competitor_pct", "avg_price_gap_pct", "catalogue_size",
        "account_tenure_days", "is_gold", "is_silver", "is_bronze",
    ] + list(cat_dummies.columns)

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols


# ============================================================================
# STEP 2 — MODEL TRAINING
# ============================================================================
def make_model(use_smote):
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=SEED, n_jobs=-1,
    )
    steps = [("smote", SMOTE(random_state=SEED))] if use_smote else []
    steps.append(("xgb", xgb))
    return ImbPipeline(steps)


def train_model(X, y):
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    majority, minority = max(n_pos, n_neg), min(n_pos, n_neg)
    imbalance = majority / max(minority, 1)
    print(f"\nClass balance -> churned={n_pos}, active={n_neg}, imbalance={imbalance:.2f}:1")
    use_smote = imbalance > 3.0
    print(f"SMOTE {'ENABLED' if use_smote else 'disabled'} (rule: imbalance > 3:1)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    # 5-fold CV metrics (out-of-fold predictions over the whole population).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_proba = cross_val_predict(make_model(use_smote), X, y, cv=skf, method="predict_proba")[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)
    print("\n--- 5-fold cross-validated metrics ---")
    print(f"  accuracy  : {accuracy_score(y, cv_pred):.3f}")
    print(f"  precision : {precision_score(y, cv_pred, zero_division=0):.3f}")
    print(f"  recall    : {recall_score(y, cv_pred, zero_division=0):.3f}")
    print(f"  ROC-AUC   : {roc_auc_score(y, cv_proba):.3f}")

    # Held-out test split.
    model = make_model(use_smote)
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    print("\n--- Held-out test set (20%) ---")
    print(f"  accuracy  : {accuracy_score(y_test, test_pred):.3f}")
    print(f"  precision : {precision_score(y_test, test_pred, zero_division=0):.3f}")
    print(f"  recall    : {recall_score(y_test, test_pred, zero_division=0):.3f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, test_proba):.3f}")
    print("  confusion matrix [rows=actual, cols=pred] (0=active,1=churn):",
          confusion_matrix(y_test, test_pred).tolist())
    print("\n", classification_report(y_test, test_pred,
                                       target_names=["active", "churned"], zero_division=0))

    return model, model.named_steps["xgb"], X_test, use_smote, cv_proba


# ============================================================================
# STEP 3 — SHAP EXPLANATION
# ============================================================================
def explain_model(fitted_xgb, X_test, feature_cols):
    explainer = shap.TreeExplainer(fitted_xgb)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    print("\n--- SHAP top 10 churn drivers (mean |SHAP|) ---")
    for i, row in importance.head(10).iterrows():
        print(f"  {i + 1:>2}. {row['feature']:<28} {row['mean_abs_shap']:.4f}")
    print(f"\n  >> #1 driver of churn risk: {importance.iloc[0]['feature']}")
    return importance


# ============================================================================
# STEP 4 — SCORE ACTIVE SELLERS
# ============================================================================
def risk_tier(p):
    if p > 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def recommend(row):
    """Prescriptive action routed by *why* a high-risk seller is risky."""
    tier = row["risk_tier"]
    if tier == "Low":
        return "No action needed"
    if tier == "Medium":
        return "Monitor closely"
    if row["return_rate"] >= 20:
        return "Quality audit + account review"
    if row["gmv_trend_slope"] < 0:
        return "Engagement call + growth incentive"
    if row["price_above_competitor_pct"] > 50:
        return "Pricing strategy review"
    return "Engagement call + growth incentive"


def score_active_sellers(active_df, oof_proba):
    out = active_df.copy()
    out["churn_probability"] = np.round(oof_proba, 4)
    out["risk_tier"] = out["churn_probability"].apply(risk_tier)
    out["recommended_action"] = out.apply(recommend, axis=1)

    result = (out[["seller_id", "category", "gmv_30d", "orders_30d", "return_rate",
                   "churn_probability", "risk_tier", "recommended_action"]]
              .sort_values("churn_probability", ascending=False))
    result["return_rate"] = result["return_rate"].round(1)
    result["gmv_30d"] = result["gmv_30d"].round(2)
    result["orders_30d"] = result["orders_30d"].astype(int)
    result.to_csv(os.path.join(OUT_DIR, "seller_risk_scores.csv"), index=False)

    counts = (result["risk_tier"].value_counts()
              .reindex(["High", "Medium", "Low"]).fillna(0).astype(int))
    gmv_at_risk = result.loc[result["risk_tier"] == "High", "gmv_30d"].sum()
    print("\n--- Active-seller risk summary (scored as of "
          f"{CUTOFF.date()}, out-of-fold) ---")
    print(f"  High   risk : {counts['High']:>3} sellers")
    print(f"  Medium risk : {counts['Medium']:>3} sellers")
    print(f"  Low    risk : {counts['Low']:>3} sellers")
    print(f"  Total active scored : {len(result)}")
    print(f"\n  GMV at risk (30d GMV of High-tier sellers): Rs {gmv_at_risk:,.0f}")
    # How many of the flagged High-risk sellers truly churned (sanity check).
    if "is_active" in out.columns:
        high = out[out["risk_tier"] == "High"]
        if len(high):
            caught = int((high["is_active"] == 0).sum())
            print(f"  (of {len(high)} High-risk, {caught} are ground-truth churned)")
    print(f"\n  seller_risk_scores.csv written to ./{OUT_DIR}")
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print(f"STEP 1 — Point-in-time feature engineering (cutoff = {CUTOFF.date()})")
    df, feature_cols = build_features_asof(CUTOFF)

    # Population = sellers still transacting at the cutoff (>=1 order in the
    # trailing 90 days). This drops already-churned sellers, so the positives
    # are declining-but-still-active sellers — the early-warning target.
    active_at_cutoff = df[df["orders_90d"] > 0].reset_index(drop=True)
    print(f"  total sellers: {len(df)}  |  active at cutoff: {len(active_at_cutoff)}")
    X = active_at_cutoff[feature_cols]
    y = (1 - active_at_cutoff["is_active"]).astype(int)  # 1 = churned

    print("\n" + "=" * 70)
    print("STEP 2 — Model training")
    model, fitted_xgb, X_test, use_smote, cv_proba = train_model(X, y)

    print("\n" + "=" * 70)
    print("STEP 3 — SHAP explanation")
    explain_model(fitted_xgb, X_test, feature_cols)

    print("\n" + "=" * 70)
    print("STEP 4 — Score active sellers")
    score_active_sellers(active_at_cutoff, cv_proba)


if __name__ == "__main__":
    main()
