"""
Marketplace Seller Intelligence Platform
========================================
Streamlit dashboard over the seller-churn risk pipeline.

Organised into business-themed tabs so different stakeholders land on what they
need: Risk Ops on the risk overview, Category/Growth managers on revenue, the
Quality team on returns, and Pricing on competitiveness.

Data sources (produced upstream by sql_analysis.py / seller_risk_model.py):
    outputs/seller_risk_scores.csv         - per-seller churn risk + GMV/returns
    outputs/category_pricing_benchmark.csv - category pricing vs competition
    outputs/seller_monthly_trend.csv       - monthly GMV per seller (history)
    outputs/seller_return_rates.csv        - per-seller orders/returns
    data/sellers.csv                       - seller master (city, tier)
    data/returns.csv                       - raw return events (reasons)
    data/product_listings.csv              - raw listings (competitor median)

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Paths — resolve relative to the project root (one level up from /dashboard)
# so the app works regardless of the launch directory.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")

# Consistent colour maps reused across every chart.
TIER_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
TIER_ORDER = ["High", "Medium", "Low"]
ACCOUNT_TIER_ORDER = ["Gold", "Silver", "Bronze"]
ACCOUNT_TIER_COLORS = {"Gold": "#d4af37", "Silver": "#9aa0a6", "Bronze": "#cd7f32"}

st.set_page_config(
    page_title="Marketplace Seller Intelligence Platform",
    layout="wide",
    page_icon="📊",
)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_risk_scores():
    """Risk scores enriched with city + account_tier from the seller master so
    geographic / tier views also respond to the sidebar filters."""
    df = pd.read_csv(os.path.join(OUT, "seller_risk_scores.csv"))
    sellers = pd.read_csv(os.path.join(DATA, "sellers.csv"))[
        ["seller_id", "city", "account_tier"]
    ]
    df = df.merge(sellers, on="seller_id", how="left")
    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=TIER_ORDER, ordered=True)
    return df


@st.cache_data
def load_pricing_benchmark():
    """Per-category avg seller listed price, true competitor MEDIAN (from raw
    listings, robust to the price tail), and the avg price gap %."""
    bench = pd.read_csv(os.path.join(OUT, "category_pricing_benchmark.csv"))
    listings = pd.read_csv(os.path.join(DATA, "product_listings.csv"))
    comp_median = (listings.groupby("category")["competitor_price"]
                   .median().round(2).reset_index()
                   .rename(columns={"competitor_price": "competitor_median_price"}))
    return bench.merge(comp_median, on="category")


@st.cache_data
def load_monthly_trend():
    """Monthly GMV per seller in long format (seller_id, month, gmv)."""
    df = pd.read_csv(os.path.join(OUT, "seller_monthly_trend.csv"))
    month_cols = [c for c in df.columns if c != "seller_id"]
    long = df.melt(id_vars="seller_id", value_vars=month_cols,
                   var_name="month", value_name="gmv")
    return long


@st.cache_data
def load_returns():
    """Raw return events tagged with the seller's category (all 5 reasons)."""
    r = pd.read_csv(os.path.join(DATA, "returns.csv"))[["seller_id", "return_reason"]]
    cats = pd.read_csv(os.path.join(DATA, "sellers.csv"))[["seller_id", "category"]]
    return r.merge(cats, on="seller_id", how="left")


@st.cache_data
def load_return_rates():
    """Per-seller order/return counts for category-level return-rate maths."""
    return pd.read_csv(os.path.join(OUT, "seller_return_rates.csv"))[
        ["seller_id", "category", "total_orders", "total_returns"]
    ]


def no_data(msg="No data matches the current filters."):
    st.info(msg)


# ---------------------------------------------------------------------------
# Sidebar filters — apply to every section
# ---------------------------------------------------------------------------
risk = load_risk_scores()

st.sidebar.header("Filters")
all_categories = sorted(risk["category"].unique())
sel_categories = st.sidebar.multiselect(
    "Category", options=all_categories, default=all_categories,
    help="Filter every chart and table by seller category.",
)
sel_tiers = st.sidebar.multiselect(
    "Risk tier", options=TIER_ORDER, default=TIER_ORDER,
    help="Filter every chart and table by churn risk tier.",
)
# Guard against empty selections blanking the whole page.
sel_categories = sel_categories or all_categories
sel_tiers = sel_tiers or TIER_ORDER

f = risk[risk["category"].isin(sel_categories) & risk["risk_tier"].isin(sel_tiers)].copy()
sel_ids = set(f["seller_id"])  # drives the trend / returns charts too


# ---------------------------------------------------------------------------
# Header + global KPIs
# ---------------------------------------------------------------------------
st.title("Marketplace Seller Intelligence Platform")
st.caption("Churn-risk early-warning system for marketplace seller operations · "
           "scored point-in-time as of 2024-03-31")

total_sellers = len(f)
gmv_30d_total = f["gmv_30d"].sum()
high_df = f[f["risk_tier"] == "High"]
revenue_at_risk = high_df["gmv_30d"].sum()
pct_at_risk = (revenue_at_risk / gmv_30d_total * 100) if gmv_30d_total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Active sellers", f"{total_sellers:,}")
c2.metric("Platform GMV (30d)", f"₹{gmv_30d_total:,.0f}")
c3.metric("At-risk sellers (High)", f"{len(high_df):,}")
c4.metric("Revenue at risk", f"₹{revenue_at_risk:,.0f}", f"{pct_at_risk:.1f}% of GMV",
          delta_color="inverse")

st.divider()

tab_risk, tab_rev, tab_ret, tab_price, tab_watch = st.tabs([
    "🎯 Risk Overview", "💰 Revenue & Growth", "↩️ Returns & Quality",
    "🏷️ Pricing", "📋 Watchlist",
])


# ===========================================================================
# TAB 1 — RISK OVERVIEW
# ===========================================================================
with tab_risk:
    # --- Seller risk map ----------------------------------------------------
    st.subheader("Seller risk map — GMV vs return rate")
    if f.empty:
        no_data()
    else:
        fig_map = px.scatter(
            f, x="return_rate", y="gmv_30d", color="risk_tier", size="orders_30d",
            size_max=28, color_discrete_map=TIER_COLORS,
            category_orders={"risk_tier": TIER_ORDER},
            custom_data=["seller_id", "category", "churn_probability"],
            labels={"return_rate": "Return rate (%)", "gmv_30d": "GMV last 30 days (₹)",
                    "risk_tier": "Risk tier", "orders_30d": "Orders (30d)"},
        )
        fig_map.update_traces(hovertemplate=(
            "<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>"
            "Return rate: %{x:.1f}%<br>GMV 30d: ₹%{y:,.0f}<br>"
            "Churn prob: %{customdata[2]:.1%}<extra></extra>"))
        fig_map.update_layout(height=480, legend_title_text="Risk tier")
        st.plotly_chart(fig_map, width="stretch")

    # --- Portfolio composition: tier mix + churn-probability spread ----------
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Risk tier distribution")
        if f.empty:
            no_data()
        else:
            tier_counts = (f["risk_tier"].value_counts()
                           .reindex(TIER_ORDER).fillna(0).reset_index())
            tier_counts.columns = ["risk_tier", "count"]
            fig_tier = px.pie(
                tier_counts, names="risk_tier", values="count", hole=0.55,
                color="risk_tier", color_discrete_map=TIER_COLORS,
                category_orders={"risk_tier": TIER_ORDER},
            )
            fig_tier.update_traces(textinfo="label+percent")
            fig_tier.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_tier, width="stretch")
    with col_b:
        st.subheader("Churn-probability spread")
        if f.empty:
            no_data()
        else:
            fig_hist = px.histogram(
                f, x="churn_probability", nbins=25, color="risk_tier",
                color_discrete_map=TIER_COLORS, category_orders={"risk_tier": TIER_ORDER},
                labels={"churn_probability": "Churn probability"},
            )
            # Tier thresholds (0.4 Medium, 0.7 High) as reference lines.
            fig_hist.add_vline(x=0.4, line_dash="dash", line_color="gray")
            fig_hist.add_vline(x=0.7, line_dash="dash", line_color="gray")
            fig_hist.update_layout(height=360, legend_title_text="Risk tier",
                                   yaxis_title="Sellers")
            st.plotly_chart(fig_hist, width="stretch")

    # --- Intervention workload ----------------------------------------------
    st.subheader("Recommended-action workload")
    st.caption("How many sellers fall into each playbook action — for staffing interventions.")
    if f.empty:
        no_data()
    else:
        act = (f["recommended_action"].value_counts().reset_index())
        act.columns = ["recommended_action", "count"]
        fig_act = px.bar(
            act.sort_values("count"), x="count", y="recommended_action",
            orientation="h", text="count",
            labels={"count": "Sellers", "recommended_action": ""},
        )
        fig_act.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_act, width="stretch")


# ===========================================================================
# TAB 2 — REVENUE & GROWTH
# ===========================================================================
with tab_rev:
    # --- Monthly GMV momentum (stacked by category) -------------------------
    st.subheader("Monthly platform GMV momentum")
    st.caption("Historical monthly GMV (2024) for the selected sellers, stacked by category — "
               "shows growth trajectory and seasonality.")
    trend = load_monthly_trend()
    trend = trend[trend["seller_id"].isin(sel_ids)]
    cat_map = dict(zip(risk["seller_id"], risk["category"]))
    trend = trend.assign(category=trend["seller_id"].map(cat_map))
    if trend.empty:
        no_data()
    else:
        monthly = (trend.groupby(["month", "category"], as_index=False)["gmv"].sum()
                   .sort_values("month"))
        fig_mom = px.area(
            monthly, x="month", y="gmv", color="category",
            labels={"month": "Month", "gmv": "GMV (₹)", "category": "Category"},
        )
        fig_mom.update_layout(height=400, legend_title_text="Category")
        st.plotly_chart(fig_mom, width="stretch")

    col_c, col_d = st.columns(2)
    # --- Revenue at risk by category ----------------------------------------
    with col_c:
        st.subheader("Revenue at risk by category")
        if f.empty:
            no_data()
        else:
            by_cat = f.groupby("category").agg(
                total_gmv=("gmv_30d", "sum"),
                at_risk_gmv=("gmv_30d", lambda s: s[f.loc[s.index, "risk_tier"] == "High"].sum()),
            ).reset_index()
            cat_long = by_cat.melt(id_vars="category", var_name="series", value_name="gmv")
            cat_long["series"] = cat_long["series"].map({
                "total_gmv": "Total GMV", "at_risk_gmv": "At-risk (High) GMV"})
            fig_rar = px.bar(
                cat_long, x="gmv", y="category", color="series", orientation="h",
                barmode="group",
                color_discrete_map={"Total GMV": "#1f77b4", "At-risk (High) GMV": "#d62728"},
                labels={"gmv": "GMV (₹)", "category": "", "series": ""},
            )
            fig_rar.update_layout(height=380, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_rar, width="stretch")
    # --- GMV by account tier -------------------------------------------------
    with col_d:
        st.subheader("GMV by account tier")
        if f.empty:
            no_data()
        else:
            by_tier = (f.groupby("account_tier")["gmv_30d"].sum()
                       .reindex(ACCOUNT_TIER_ORDER).fillna(0).reset_index())
            fig_at = px.bar(
                by_tier, x="account_tier", y="gmv_30d", color="account_tier",
                color_discrete_map=ACCOUNT_TIER_COLORS,
                category_orders={"account_tier": ACCOUNT_TIER_ORDER},
                labels={"account_tier": "", "gmv_30d": "GMV (₹)"}, text_auto=".2s",
            )
            fig_at.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_at, width="stretch")

    # --- Top sellers on the watchlist by GMV --------------------------------
    st.subheader("Top sellers at risk by GMV (High + Medium)")
    st.caption("Where retention effort protects the most revenue — biggest GMV among at-risk sellers.")
    watch_gmv = f[f["risk_tier"].isin(["High", "Medium"])].nlargest(10, "gmv_30d")
    if watch_gmv.empty:
        no_data("No High/Medium sellers in the current selection.")
    else:
        fig_top = px.bar(
            watch_gmv.sort_values("gmv_30d"), x="gmv_30d", y="seller_id",
            color="risk_tier", orientation="h", color_discrete_map=TIER_COLORS,
            category_orders={"risk_tier": TIER_ORDER},
            custom_data=["category", "churn_probability"],
            labels={"gmv_30d": "GMV last 30 days (₹)", "seller_id": "", "risk_tier": "Risk tier"},
        )
        fig_top.update_traces(hovertemplate=(
            "<b>%{y}</b><br>Category: %{customdata[0]}<br>"
            "GMV 30d: ₹%{x:,.0f}<br>Churn prob: %{customdata[1]:.1%}<extra></extra>"))
        fig_top.update_layout(height=420, legend_title_text="Risk tier")
        st.plotly_chart(fig_top, width="stretch")


# ===========================================================================
# TAB 3 — RETURNS & QUALITY
# ===========================================================================
with tab_ret:
    col_e, col_f = st.columns(2)
    # --- Return-reason composition ------------------------------------------
    with col_e:
        st.subheader("Return reasons")
        st.caption("What drives returns — size/quality issues point to different fixes.")
        rr = load_returns()
        rr = rr[rr["seller_id"].isin(sel_ids) & rr["category"].isin(sel_categories)]
        if rr.empty:
            no_data()
        else:
            reasons = rr["return_reason"].value_counts().reset_index()
            reasons.columns = ["return_reason", "count"]
            fig_rsn = px.pie(reasons, names="return_reason", values="count", hole=0.5)
            fig_rsn.update_traces(textinfo="percent")
            fig_rsn.update_layout(height=380, legend_title_text="Reason")
            st.plotly_chart(fig_rsn, width="stretch")
    # --- Return rate by category --------------------------------------------
    with col_f:
        st.subheader("Return rate by category")
        st.caption("Categories with structurally higher returns need tighter quality controls.")
        rates = load_return_rates()
        rates = rates[rates["seller_id"].isin(sel_ids) & rates["category"].isin(sel_categories)]
        if rates.empty or rates["total_orders"].sum() == 0:
            no_data()
        else:
            agg = rates.groupby("category").agg(
                orders=("total_orders", "sum"), returns=("total_returns", "sum")).reset_index()
            agg["return_rate"] = (100.0 * agg["returns"] / agg["orders"]).round(1)
            fig_rrc = px.bar(
                agg.sort_values("return_rate"), x="return_rate", y="category",
                orientation="h", color="return_rate", color_continuous_scale="Reds",
                text="return_rate",
                labels={"return_rate": "Return rate (%)", "category": ""},
            )
            fig_rrc.update_traces(texttemplate="%{text}%")
            fig_rrc.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_rrc, width="stretch")


# ===========================================================================
# TAB 4 — PRICING
# ===========================================================================
with tab_price:
    pricing = load_pricing_benchmark()
    pricing = pricing[pricing["category"].isin(sel_categories)]

    st.subheader("Seller pricing vs competition by category")
    if pricing.empty:
        no_data()
    else:
        pm = pricing.melt(id_vars="category",
                          value_vars=["avg_listed_price", "competitor_median_price"],
                          var_name="series", value_name="price")
        pm["series"] = pm["series"].map({
            "avg_listed_price": "Avg seller listed price",
            "competitor_median_price": "Competitor median price"})
        fig_price = px.bar(
            pm, x="price", y="category", color="series", orientation="h", barmode="group",
            color_discrete_map={"Avg seller listed price": "#1f77b4",
                                "Competitor median price": "#9467bd"},
            labels={"price": "Price (₹)", "category": "", "series": ""},
        )
        fig_price.update_layout(height=400, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_price, width="stretch")

    # --- Price gap %: above (uncompetitive) vs below market ------------------
    st.subheader("Average price gap vs competitor (%)")
    st.caption("Positive = priced above competitors (conversion risk); negative = below (margin left on table).")
    if pricing.empty:
        no_data()
    else:
        gap = pricing[["category", "price_gap_pct"]].copy()
        gap["direction"] = gap["price_gap_pct"].apply(lambda v: "Above competitor" if v >= 0 else "Below competitor")
        fig_gap = px.bar(
            gap.sort_values("price_gap_pct"), x="price_gap_pct", y="category",
            orientation="h", color="direction",
            color_discrete_map={"Above competitor": "#d62728", "Below competitor": "#2ca02c"},
            labels={"price_gap_pct": "Price gap (%)", "category": "", "direction": ""},
        )
        fig_gap.update_layout(height=360, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_gap, width="stretch")


# ===========================================================================
# TAB 5 — WATCHLIST
# ===========================================================================
with tab_watch:
    st.subheader("At-risk seller watchlist (High + Medium)")
    watch = (f[f["risk_tier"].isin(["High", "Medium"])]
             .sort_values("churn_probability", ascending=False)
             .loc[:, ["seller_id", "category", "gmv_30d", "return_rate",
                      "churn_probability", "risk_tier", "recommended_action"]])
    if watch.empty:
        no_data("No High or Medium risk sellers in the current selection.")
    else:
        st.dataframe(
            watch, width="stretch", hide_index=True,
            column_config={
                "seller_id": st.column_config.TextColumn("Seller ID"),
                "category": st.column_config.TextColumn("Category"),
                "gmv_30d": st.column_config.NumberColumn("GMV (30d)", format="₹%.0f"),
                "return_rate": st.column_config.NumberColumn("Return rate", format="%.1f%%"),
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"),
                "risk_tier": st.column_config.TextColumn("Risk tier"),
                "recommended_action": st.column_config.TextColumn("Recommended action"),
            },
        )
        st.caption(f"{len(watch)} sellers on the watchlist · click any column header to re-sort.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Built with Python · XGBoost · Streamlit | Hariharan Balaji"
    "</div>",
    unsafe_allow_html=True,
)
