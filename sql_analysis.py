"""
sql_analysis.py
===============
Loads the 4 marketplace CSVs into a SQLite database (marketplace.db) and runs
five analytical SQL queries that a marketplace / seller-ops team would use to
spot at-risk sellers, benchmark pricing, and build trend features.

Each result is kept as a pandas DataFrame and exported to ./outputs/<name>.csv.

IMPORTANT — date anchoring:
The synthetic order history ends on 2024-06-30, not "today". So "last 30/60/90
days" is measured relative to MAX(order_date) in the data (the as-of date),
not the real-world clock. Otherwise every rolling window would be empty.

Run:  python sql_analysis.py
"""

import os
import sqlite3
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "outputs"
DB_PATH = "marketplace.db"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Load CSVs -> SQLite
# ----------------------------------------------------------------------------
def load_database():
    """Read the 4 CSVs and write them as tables into a fresh SQLite db."""
    conn = sqlite3.connect(DB_PATH)

    tables = {
        "sellers": "sellers.csv",
        "orders": "orders.csv",
        "returns": "returns.csv",
        "product_listings": "product_listings.csv",
    }
    for table, fname in tables.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        df.to_sql(table, conn, if_exists="replace", index=False)

    # Helpful indexes — these queries join/filter heavily on seller_id and dates.
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_seller ON orders(seller_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_returns_seller ON returns(seller_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_listings_seller ON product_listings(seller_id)")
    conn.commit()
    return conn


def report(name, df):
    """Print shape + head and export the dataframe to /outputs."""
    df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
    print(f"\n{'=' * 70}\n{name}  ->  shape={df.shape}")
    print(df.head().to_string(index=False))


def main():
    conn = load_database()

    # As-of date = latest order in the dataset. All rolling windows hang off this.
    anchor = pd.read_sql("SELECT MAX(order_date) AS d FROM orders", conn)["d"].iloc[0]
    print(f"As-of (anchor) date = {anchor}")

    # ------------------------------------------------------------------------
    # Query A — seller_gmv_summary
    # ------------------------------------------------------------------------
    # Business question: "How much revenue is each seller driving right now, and
    # is it growing or shrinking?" Rolling 30/60/90-day GMV plus a month-over-
    # month delta is the core health vitals card for the marketplace team — a
    # sudden MoM drop is the earliest signal of a seller starting to disengage.
    query_a = f"""
    WITH bounds AS (
        SELECT
            date('{anchor}')                AS as_of,
            date('{anchor}', '-30 days')    AS d30,
            date('{anchor}', '-60 days')    AS d60,
            date('{anchor}', '-90 days')    AS d90,
            strftime('%Y-%m', '{anchor}')                       AS cur_month,
            strftime('%Y-%m', date('{anchor}', 'start of month', '-1 day')) AS prev_month
    )
    SELECT
        s.seller_id,
        s.seller_name,
        s.category,
        -- rolling-window GMV (windows are cumulative: 90d includes 30d)
        COALESCE(SUM(CASE WHEN o.order_date > b.d30 THEN o.gmv END), 0) AS gmv_last_30d,
        COALESCE(SUM(CASE WHEN o.order_date > b.d60 THEN o.gmv END), 0) AS gmv_last_60d,
        COALESCE(SUM(CASE WHEN o.order_date > b.d90 THEN o.gmv END), 0) AS gmv_last_90d,
        COALESCE(SUM(CASE WHEN o.order_date > b.d30 THEN 1 ELSE 0 END), 0) AS orders_last_30d,
        -- month-over-month GMV change %: latest calendar month vs the one before
        COALESCE(SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.cur_month THEN o.gmv END), 0)  AS gmv_cur_month,
        COALESCE(SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.prev_month THEN o.gmv END), 0) AS gmv_prev_month,
        ROUND(
            CASE
                WHEN SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.prev_month THEN o.gmv END) > 0
                THEN 100.0 * (
                    SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.cur_month  THEN o.gmv END) -
                    SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.prev_month THEN o.gmv END)
                ) / SUM(CASE WHEN strftime('%Y-%m', o.order_date) = b.prev_month THEN o.gmv END)
                ELSE NULL
            END, 1
        ) AS gmv_mom_change_pct
    FROM sellers s
    CROSS JOIN bounds b
    LEFT JOIN orders o ON o.seller_id = s.seller_id
    GROUP BY s.seller_id, s.seller_name, s.category
    ORDER BY gmv_last_30d DESC
    """
    seller_gmv_summary = pd.read_sql(query_a, conn)
    report("seller_gmv_summary", seller_gmv_summary)

    # ------------------------------------------------------------------------
    # Query B — seller_return_rates
    # ------------------------------------------------------------------------
    # Business question: "Which sellers have a return problem, and why?"
    # Return rate is a direct proxy for customer dissatisfaction and erodes
    # profitability (reverse logistics cost). Splitting returns into quality /
    # size / damage tells ops whether the fix is supplier quality, size-chart
    # accuracy, or packaging — a high return rate is also a leading churn signal.
    query_b = """
    SELECT
        s.seller_id,
        s.seller_name,
        s.category,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COUNT(r.return_id)         AS total_returns,
        ROUND(
            100.0 * COUNT(r.return_id) / NULLIF(COUNT(DISTINCT o.order_id), 0), 1
        ) AS return_rate_pct,
        SUM(CASE WHEN r.return_reason = 'quality issue' THEN 1 ELSE 0 END) AS quality_returns,
        SUM(CASE WHEN r.return_reason = 'size issue'    THEN 1 ELSE 0 END) AS size_returns,
        SUM(CASE WHEN r.return_reason = 'damaged'       THEN 1 ELSE 0 END) AS damage_returns
    FROM sellers s
    LEFT JOIN orders  o ON o.seller_id = s.seller_id
    LEFT JOIN returns r ON r.order_id  = o.order_id
    GROUP BY s.seller_id, s.seller_name, s.category
    ORDER BY return_rate_pct DESC
    """
    seller_return_rates = pd.read_sql(query_b, conn)
    report("seller_return_rates", seller_return_rates)

    # ------------------------------------------------------------------------
    # Query C — category_pricing_benchmark
    # ------------------------------------------------------------------------
    # Business question: "Are we priced competitively per category vs rival
    # platforms?" If our listed prices sit well above competitors, we lose the
    # buy-box / conversion; well below, we leave margin on the table. The price
    # gap % guides category-level pricing and seller-incentive decisions.
    # NOTE: SQLite has no PERCENTILE/MEDIAN function, so avg + competitor + gap
    # are computed in SQL and the median listed price is added via pandas.
    query_c = """
    SELECT
        category,
        ROUND(AVG(listed_price), 2)     AS avg_listed_price,
        ROUND(AVG(competitor_price), 2) AS avg_competitor_price,
        ROUND(
            100.0 * (AVG(listed_price) - AVG(competitor_price)) / AVG(competitor_price), 2
        ) AS price_gap_pct,
        COUNT(*) AS num_listings
    FROM product_listings
    GROUP BY category
    ORDER BY price_gap_pct DESC
    """
    category_pricing_benchmark = pd.read_sql(query_c, conn)
    # Median listed price per category (computed in pandas — see note above).
    median_listed = (
        pd.read_sql("SELECT category, listed_price FROM product_listings", conn)
        .groupby("category")["listed_price"].median().round(2)
        .reset_index().rename(columns={"listed_price": "median_listed_price"})
    )
    category_pricing_benchmark = category_pricing_benchmark.merge(
        median_listed, on="category"
    )
    # Reorder so avg / median sit together.
    category_pricing_benchmark = category_pricing_benchmark[[
        "category", "avg_listed_price", "median_listed_price",
        "avg_competitor_price", "price_gap_pct", "num_listings",
    ]]
    report("category_pricing_benchmark", category_pricing_benchmark)

    # ------------------------------------------------------------------------
    # Query D — seller_monthly_trend
    # ------------------------------------------------------------------------
    # Business question: "What does each seller's recent GMV trajectory look
    # like?" Six months of monthly GMV in WIDE format (one column per month) is
    # the feature matrix for trend/slope features used in churn modelling — a
    # downward staircase across the columns is the classic pre-churn pattern.
    query_d = f"""
    SELECT
        o.seller_id,
        strftime('%Y-%m', o.order_date) AS order_month,
        SUM(o.gmv) AS monthly_gmv
    FROM orders o
    -- last 6 calendar months inclusive of the anchor month: -5 months back from
    -- the anchor's month start gives exactly Jan..Jun 2024 (6 columns, not 7).
    WHERE o.order_date >= date('{anchor}', 'start of month', '-5 months')
    GROUP BY o.seller_id, order_month
    """
    monthly_long = pd.read_sql(query_d, conn)
    # Pivot long -> wide: one column per month. SQLite can't pivot dynamically,
    # so we reshape in pandas. Missing months -> 0 GMV (seller had no sales).
    seller_monthly_trend = (
        monthly_long
        .pivot(index="seller_id", columns="order_month", values="monthly_gmv")
        .fillna(0)
        .round(2)
        .reset_index()
    )
    seller_monthly_trend.columns.name = None
    report("seller_monthly_trend", seller_monthly_trend)

    # ------------------------------------------------------------------------
    # Query E — seller_catalogue_health
    # ------------------------------------------------------------------------
    # Business question: "Is each seller's catalogue well-priced and deep
    # enough?" Catalogue depth (total listings) plus the share of items priced
    # above/below competitors flags sellers who are either uncompetitive (mostly
    # above market -> low conversion) or potentially under-pricing. Thin, badly
    # priced catalogues correlate with low GMV and eventual churn.
    query_e = """
    SELECT
        s.seller_id,
        s.seller_name,
        s.account_tier,
        COUNT(pl.listing_id)            AS total_listings,
        ROUND(AVG(pl.listed_price), 2)  AS avg_listed_price,
        ROUND(
            100.0 * SUM(CASE WHEN pl.listed_price > pl.competitor_price THEN 1 ELSE 0 END)
            / NULLIF(COUNT(pl.listing_id), 0), 1
        ) AS pct_priced_above_competitor,
        ROUND(
            100.0 * SUM(CASE WHEN pl.listed_price < pl.competitor_price THEN 1 ELSE 0 END)
            / NULLIF(COUNT(pl.listing_id), 0), 1
        ) AS pct_priced_below_competitor
    FROM sellers s
    LEFT JOIN product_listings pl ON pl.seller_id = s.seller_id
    GROUP BY s.seller_id, s.seller_name, s.account_tier
    ORDER BY total_listings DESC
    """
    seller_catalogue_health = pd.read_sql(query_e, conn)
    report("seller_catalogue_health", seller_catalogue_health)

    conn.close()
    print(f"\n{'=' * 70}\nAll outputs written to ./{OUT_DIR}")


if __name__ == "__main__":
    main()
