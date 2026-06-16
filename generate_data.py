"""
generate_data.py
================
Generates synthetic-but-realistic data for a marketplace seller analytics
project that mimics Myntra's marketplace-team workflows.

Produces 4 CSVs in ./data:
    1. sellers.csv           - seller master (500 sellers)
    2. orders.csv            - ~18 months of order history
    3. returns.csv           - return events linked to orders
    4. product_listings.csv  - current catalogue per seller

Design goal: the data should let a downstream model learn what a churning
seller looks like (declining orders + rising returns in the months before
they go inactive), so the patterns below are deliberately correlated with
the `is_active` label rather than purely random.

Run:  python generate_data.py
"""

import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
# A single fixed seed makes the whole pipeline deterministic. We also create a
# dedicated NumPy Generator (np.random.default_rng) which is the modern,
# preferred API over the legacy np.random.* global state. Using one Generator
# threaded through every step keeps the run fully reproducible.
SEED = 42
np.random.seed(SEED)          # seeds legacy API (used by pandas sampling etc.)
rng = np.random.default_rng(SEED)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------------
CATEGORIES = ["Ethnic Wear", "Footwear", "Western Wear", "Accessories", "Sports"]

# Ethnic Wear is weighted highest because, for an India-first marketplace like
# Myntra, ethnic/festive apparel is the largest seller segment. Sports is the
# smallest niche. Probabilities must sum to 1.
CATEGORY_WEIGHTS = [0.35, 0.22, 0.20, 0.13, 0.10]

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Surat"]
# City weights loosely reflect seller-hub concentration. Surat is included
# specifically because it is a major textile/ethnic-wear manufacturing hub.
CITY_WEIGHTS = [0.24, 0.22, 0.18, 0.12, 0.12, 0.12]

# Per-category baseline return rates. Apparel that depends on fit (ethnic,
# western) returns far more than accessories or sports gear. These match the
# rates requested in the brief and are realistic for fashion ecommerce.
CATEGORY_RETURN_RATE = {
    "Ethnic Wear": 0.22,
    "Western Wear": 0.18,
    "Footwear": 0.15,
    "Accessories": 0.10,
    "Sports": 0.08,
}

# Price ranges (₹) per category. Ethnic wear skews premium (festive/occasion
# wear), accessories are cheapest, footwear has the widest spread.
CATEGORY_PRICE_RANGE = {
    "Ethnic Wear": (800, 2500),
    "Footwear": (500, 3000),
    "Western Wear": (600, 2000),
    "Accessories": (200, 800),
    "Sports": (400, 2000),
}

RETURN_REASONS = ["size issue", "quality issue", "wrong item", "damaged", "changed mind"]
# Size issues dominate fashion returns; "damaged in transit" is rarest.
RETURN_REASON_WEIGHTS = [0.40, 0.20, 0.12, 0.10, 0.18]

# Order window: Jan 2023 -> Jun 2024 inclusive (18 months).
ORDER_MONTHS = pd.period_range("2023-01", "2024-06", freq="M")

# Monthly seasonality multipliers applied to every seller's order volume.
# Oct-Nov spikes for the festive/wedding + Big Fashion Sale season; Feb-Mar
# dips in the post-festive lull. Values are multipliers around 1.0.
SEASONALITY = {
    1: 0.95,   # Jan  - post-holiday normal
    2: 0.80,   # Feb  - dip
    3: 0.82,   # Mar  - dip
    4: 0.95,
    5: 1.00,
    6: 1.05,   # Jun  - mid-year sale bump
    7: 1.00,
    8: 1.05,
    9: 1.10,   # Sep  - festive build-up begins
    10: 1.45,  # Oct  - festive peak (Diwali / Big Fashion Sale)
    11: 1.40,  # Nov  - festive continues
    12: 1.10,  # Dec  - year-end / wedding season
}


# ----------------------------------------------------------------------------
# 1. sellers.csv
# ----------------------------------------------------------------------------
def generate_sellers(n=500):
    """500 sellers with id, name, category, city, join_date, tier, active flag."""
    seller_ids = [f"MNT-{i:04d}" for i in range(1, n + 1)]

    categories = rng.choice(CATEGORIES, size=n, p=CATEGORY_WEIGHTS)
    cities = rng.choice(CITIES, size=n, p=CITY_WEIGHTS)

    # Readable but synthetic seller names. Real marketplaces have a long tail
    # of small "<City> <Category> House"-style trade names.
    name_suffix = rng.choice(
        ["Trends", "House", "Bazaar", "Collection", "Mart", "Studio", "Emporium", "Hub"],
        size=n,
    )
    seller_names = [
        f"{city} {cat.split()[0]} {suf}"
        for city, cat, suf in zip(cities, categories, name_suffix)
    ]

    # join_date uniformly spread across Jan 2022 - Dec 2023. Sellers must exist
    # before the order window starts (Jan 2023) OR join partway through it.
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2023-12-31")
    span_days = (end - start).days
    join_offsets = rng.integers(0, span_days + 1, size=n)
    join_dates = [start + pd.Timedelta(days=int(d)) for d in join_offsets]

    # Account tier 20/50/30 Gold/Silver/Bronze. Pyramid shape: few top sellers,
    # a large middle, and a meaningful base of small sellers.
    account_tiers = rng.choice(["Gold", "Silver", "Bronze"], size=n, p=[0.20, 0.50, 0.30])

    # 85% active / 15% churned. The 15% inactive sellers are our churn labels;
    # downstream models predict this flag from order + return behaviour.
    is_active = rng.choice([1, 0], size=n, p=[0.85, 0.15])

    df = pd.DataFrame({
        "seller_id": seller_ids,
        "seller_name": seller_names,
        "category": categories,
        "city": cities,
        "join_date": pd.to_datetime(join_dates).date,
        "account_tier": account_tiers,
        "is_active": is_active,
    })
    return df


# ----------------------------------------------------------------------------
# 2. orders.csv
# ----------------------------------------------------------------------------
def generate_orders(sellers):
    """~18 months of orders. Active sellers steady; inactive sellers decline
    and stop in their final 2-3 months before churning."""
    rows = []
    order_counter = 1

    for _, s in sellers.iterrows():
        sid = s["seller_id"]
        cat = s["category"]
        active = s["is_active"]
        join = pd.Timestamp(s["join_date"])

        # Gold sellers run higher baseline volume than Bronze.
        tier_boost = {"Gold": 1.35, "Silver": 1.0, "Bronze": 0.75}[s["account_tier"]]

        # Each seller has a personal baseline order rate. Base 15-60/month is the
        # brief's range; we centre it and let tier + noise spread it out.
        base_rate = rng.uniform(15, 60) * tier_boost

        # For churned sellers, pick a "go-inactive" month somewhere in the back
        # half of the window so we can taper their volume to zero before it.
        if active == 0:
            # index into ORDER_MONTHS where the seller effectively stops
            churn_idx = rng.integers(len(ORDER_MONTHS) - 5, len(ORDER_MONTHS))
        else:
            churn_idx = None

        for m_idx, period in enumerate(ORDER_MONTHS):
            month_start = period.to_timestamp()
            # Seller can't have orders before they joined.
            if month_start < join.to_period("M").to_timestamp():
                continue

            season = SEASONALITY[period.month]
            rate = base_rate * season

            if churn_idx is not None:
                if m_idx > churn_idx:
                    # Fully inactive after churn month: no orders.
                    continue
                # Decline ramp over the final 3 months before going inactive.
                months_to_churn = churn_idx - m_idx
                if months_to_churn <= 2:
                    # 0 months left -> ~15% volume, 1 -> ~40%, 2 -> ~70%.
                    decline = {0: 0.15, 1: 0.40, 2: 0.70}[months_to_churn]
                    rate *= decline

            # Poisson is the natural distribution for counts of independent
            # events (orders) in a fixed period; mean = our computed rate.
            n_orders = rng.poisson(max(rate, 0.1))
            if n_orders <= 0:
                continue

            days_in_month = period.days_in_month
            for _ in range(n_orders):
                day = rng.integers(1, days_in_month + 1)
                order_date = month_start + pd.Timedelta(days=int(day) - 1)

                # Items per order: mostly 1-2, occasionally a basket of several.
                # Geometric-ish via clipped Poisson keeps it small and positive.
                num_items = int(np.clip(rng.poisson(1.4) + 1, 1, 8))

                # GMV per item drawn from the category price band, then x items.
                # Lognormal-ish spread would also work; uniform-per-item with
                # item count gives a realistic right-skewed order-value curve.
                lo, hi = CATEGORY_PRICE_RANGE[cat]
                per_item = rng.uniform(lo, hi)
                gmv = round(per_item * num_items, 2)

                rows.append({
                    "order_id": f"ORD-{order_counter:07d}",
                    "seller_id": sid,
                    "order_date": order_date.date(),
                    "gmv": gmv,
                    "num_items": num_items,
                    "category": cat,
                })
                order_counter += 1

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. returns.csv
# ----------------------------------------------------------------------------
def generate_returns(orders, sellers):
    """Return events linked to orders. Probability of return depends on the
    order's category and is inflated for at-risk / inactive sellers, whose
    quality problems are part of why they churn."""
    active_map = dict(zip(sellers["seller_id"], sellers["is_active"]))

    # Per-order return probability = category base rate, scaled up for the
    # sellers that ultimately churn (poor fulfilment -> more returns -> churn).
    cat_rate = orders["category"].map(CATEGORY_RETURN_RATE).to_numpy()
    inactive_mask = orders["seller_id"].map(active_map).to_numpy() == 0
    # Inactive sellers carry ~1.6x the return rate of healthy sellers.
    prob = cat_rate * np.where(inactive_mask, 1.6, 1.0)
    prob = np.clip(prob, 0, 0.95)

    draws = rng.random(len(orders))
    returned = draws < prob

    returned_orders = orders[returned].reset_index(drop=True)

    n = len(returned_orders)
    reasons = rng.choice(RETURN_REASONS, size=n, p=RETURN_REASON_WEIGHTS)

    # Returns happen a few days to ~3 weeks after the order (delivery + try-on
    # + initiate-return window). Right-skewed: most returns come quickly.
    lag_days = np.clip(rng.poisson(7, size=n) + 1, 1, 30)
    return_dates = [
        (pd.Timestamp(od) + pd.Timedelta(days=int(l))).date()
        for od, l in zip(returned_orders["order_date"], lag_days)
    ]

    df = pd.DataFrame({
        "return_id": [f"RET-{i:07d}" for i in range(1, n + 1)],
        "order_id": returned_orders["order_id"],
        "seller_id": returned_orders["seller_id"],
        "return_date": return_dates,
        "return_reason": reasons,
    })
    return df


# ----------------------------------------------------------------------------
# 4. product_listings.csv
# ----------------------------------------------------------------------------
def generate_listings(sellers):
    """Current catalogue per seller. Each seller lists several products in
    their category with a Myntra price and a competing-platform price."""
    rows = []
    listing_counter = 1

    descriptors = ["Premium", "Classic", "Designer", "Casual", "Festive",
                   "Everyday", "Signature", "Trendy", "Handcrafted", "Sport"]
    product_nouns = {
        "Ethnic Wear": ["Kurta Set", "Saree", "Lehenga", "Anarkali", "Sherwani"],
        "Footwear": ["Sneakers", "Sandals", "Loafers", "Heels", "Running Shoes"],
        "Western Wear": ["Dress", "Jeans", "T-Shirt", "Jacket", "Skirt"],
        "Accessories": ["Handbag", "Belt", "Wallet", "Sunglasses", "Scarf"],
        "Sports": ["Track Pants", "Jersey", "Gym Bag", "Yoga Mat", "Cap"],
    }

    for _, s in sellers.iterrows():
        cat = s["category"]
        lo, hi = CATEGORY_PRICE_RANGE[cat]

        # Catalogue size varies; Gold sellers list deeper catalogues.
        base_listings = {"Gold": 25, "Silver": 12, "Bronze": 6}[s["account_tier"]]
        n_listings = max(1, int(rng.poisson(base_listings)))

        for _ in range(n_listings):
            desc = rng.choice(descriptors)
            noun = rng.choice(product_nouns[cat])
            listed_price = round(rng.uniform(lo, hi), 2)

            # Competitor price is the same item on a rival platform: centred on
            # our price but nudged +/-15%, so price-competitiveness analysis has
            # signal (some sellers over-/under-priced vs market).
            factor = rng.normal(1.0, 0.08)
            factor = float(np.clip(factor, 0.85, 1.15))
            competitor_price = round(listed_price * factor, 2)

            rows.append({
                "listing_id": f"LST-{listing_counter:07d}",
                "seller_id": s["seller_id"],
                "product_name": f"{desc} {noun}",
                "category": cat,
                "listed_price": listed_price,
                "competitor_price": competitor_price,
            })
            listing_counter += 1

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    sellers = generate_sellers(500)
    orders = generate_orders(sellers)
    returns = generate_returns(orders, sellers)
    listings = generate_listings(sellers)

    sellers.to_csv(os.path.join(DATA_DIR, "sellers.csv"), index=False)
    orders.to_csv(os.path.join(DATA_DIR, "orders.csv"), index=False)
    returns.to_csv(os.path.join(DATA_DIR, "returns.csv"), index=False)
    listings.to_csv(os.path.join(DATA_DIR, "product_listings.csv"), index=False)

    print("Synthetic data written to ./data")
    print(f"  sellers.csv ............ {len(sellers):>7,} rows")
    print(f"  orders.csv ............. {len(orders):>7,} rows")
    print(f"  returns.csv ............ {len(returns):>7,} rows")
    print(f"  product_listings.csv ... {len(listings):>7,} rows")

    # Quick sanity check: returns as a share of orders should land near the
    # blended category base rate (~17%), a bit higher due to inactive inflation.
    overall_return_rate = len(returns) / len(orders) if len(orders) else 0
    print(f"\n  overall return rate .... {overall_return_rate:.1%}")
    print(f"  active sellers ......... {sellers['is_active'].sum()} / {len(sellers)}")


if __name__ == "__main__":
    main()
