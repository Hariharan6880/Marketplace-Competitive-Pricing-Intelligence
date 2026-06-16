# Marketplace Seller Intelligence Platform

An end-to-end seller-churn **early-warning** analytics project that mimics a
marketplace seller-operations workflow (Myntra-style). It generates realistic
synthetic data, runs SQL analytics, trains an explainable churn model, and
serves the results in an interactive Streamlit dashboard.

## Pipeline

```
generate_data.py  ─►  sql_analysis.py  ─►  seller_risk_model.py  ─►  dashboard/app.py
   (synthetic         (SQLite + 5            (point-in-time           (Streamlit +
    raw data)          analytical            XGBoost + SHAP            Plotly, 5 tabs)
                       queries → CSV)         churn model)
```

| Stage | Script | Output |
|-------|--------|--------|
| 1. Data generation | [`generate_data.py`](generate_data.py) | `data/*.csv` — 500 sellers, ~265K orders, returns, listings |
| 2. SQL analytics | [`sql_analysis.py`](sql_analysis.py) | `marketplace.db` + `outputs/*.csv` (GMV, returns, pricing, trends) |
| 3. Risk model | [`seller_risk_model.py`](seller_risk_model.py) | `outputs/seller_risk_scores.csv` + metrics + SHAP drivers |
| 4. Dashboard | [`dashboard/app.py`](dashboard/app.py) | Interactive web app |

## Quick start

```bash
pip install -r requirements.txt

# regenerate the full pipeline (optional — outputs are committed)
python generate_data.py
python sql_analysis.py
python seller_risk_model.py

# launch the dashboard
streamlit run dashboard/app.py
```

## Modelling notes

- **Target:** seller churn (`is_active = 0`), framed as `P(churn)`.
- **Point-in-time / leakage-free:** features are time-bounded to a **2024-03-31
  cutoff**; the label uses only sellers still transacting at that cutoff, so the
  model learns the *declining-but-still-active* pattern rather than the trivial
  "already stopped ordering" signal. Held-out ROC-AUC ≈ 0.93 (realistic, not 1.0).
- **Imbalance:** SMOTE is applied inside the CV pipeline when the class ratio
  exceeds 3:1.
- **Explainability:** SHAP `TreeExplainer` reports the top churn drivers
  (return rate, recent order volume, GMV trend).

## Dashboard

Five business-themed tabs (with global category + risk-tier filters):
**Risk Overview**, **Revenue & Growth**, **Returns & Quality**, **Pricing**,
and an interactive at-risk **Watchlist**.

## Repo notes

- `marketplace.db` is **not** committed — it's a binary artifact rebuilt by
  `sql_analysis.py`. The `data/` and `outputs/` CSVs **are** committed so the
  dashboard runs immediately after clone.
- All randomness is seeded (`SEED = 42`) for reproducibility.

---
Built with Python · XGBoost · Streamlit | Hariharan Balaji
