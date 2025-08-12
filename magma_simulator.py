
# app.py — MGTN$ Economy Calibrator
# Run with: streamlit run app.py
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MGTN$ Economy Calibrator", layout="wide")

# ---------------------------- Helpers ----------------------------
def kaiko_depth(users, k, beta):
    users = np.asarray(users, dtype=float)
    return k * np.power(np.maximum(users, 1.0), beta)  # avoid 0**beta

def tier_ratio(depth_05, depth_10, depth_20, choose="1.0"):
    # ratios relative to 2% depth
    r05 = depth_05 / depth_20 if depth_20 else 0.0
    r10 = depth_10 / depth_20 if depth_20 else 0.0
    if choose == "0.5":
        return r05
    if choose == "1.0":
        return r10
    return 1.0

def cap_per_user(buildings, users_per_building, k, beta, ratio):
    u = np.asarray(buildings) * users_per_building
    d2 = kaiko_depth(u, k, beta)
    target = d2 * ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(u>0, target / u, 0.0)
    return cap

def ema(series, span=14):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def dao_cap_policy(ema_depth, users_series, cadence_days=14, trigger_pct=0.05,
                   max_step_pct=0.10, cap_floor=1.0, cap_ceil=100.0):
    # Instant cap per user (target) = EMA depth / users
    target = np.where(users_series>0, ema_depth / users_series, 0.0)
    cap = np.zeros_like(target, dtype=float)
    anchor = ema_depth[0]
    cap_val = np.clip(target[0], cap_floor, cap_ceil)
    for t in range(len(target)):
        if t>0 and t % cadence_days == 0:
            if anchor > 0 and abs(ema_depth[t] - anchor) / anchor >= trigger_pct:
                desired = np.clip(target[t], cap_floor, cap_ceil)
                delta = desired - cap_val
                step = np.sign(delta) * min(abs(delta), max_step_pct * max(cap_val, 1e-9))
                cap_val += step
                anchor = ema_depth[t]
        cap_val = float(np.clip(cap_val, cap_floor, cap_ceil))
        cap[t] = cap_val
    return cap

def build_emission_schedule(months, r0=0.15, rf=0.02, half_life_years=6.0):
    """Return monthly emission rate array scaled so that half-life ≈ half_life_years (no reinjection)."""
    t_years = np.linspace(0, months/12, months)
    # base curve moves from r0 toward rf
    # use simple exp toward rf and then scale to match half-life
    k = -math.log(1e-4 / max(r0-rf, 1e-9)) / max(t_years[-1], 1e-6)  # reach rf near the horizon
    annual_rate = rf + (r0 - rf) * np.exp(-k * t_years)
    monthly_rate = annual_rate / 12.0

    # bisection scale for half-life
    def balance_after(mths, scale=1.0):
        bal = 1.0
        for i in range(mths):
            bal -= bal * monthly_rate[i] * scale
        return bal

    target_month = int(round(half_life_years * 12))
    lo, hi = 0.05, 5.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        bal_mid = balance_after(target_month, scale=mid)
        if bal_mid > 0.5:
            lo = mid
        else:
            hi = mid
    scale = 0.5*(lo+hi)
    return monthly_rate * scale, annual_rate * scale

def simulate_pool(total_pool, monthly_rate, monthly_injection):
    bal = np.zeros_like(monthly_rate, dtype=float)
    x = float(total_pool)
    for i in range(len(monthly_rate)):
        emitted = x * monthly_rate[i]
        x = x - emitted + (monthly_injection[i] if i < len(monthly_injection) else 0.0)
        bal[i] = x
    return bal

def normalize_weights(weights):
    s = sum(weights)
    if s <= 0:
        return [1/len(weights)]*len(weights)
    return [w/s for w in weights]

def ladder_alpha(depth_min, depth_max, n_slices, targets, total_at_2pct):
    edges = np.linspace(depth_min, depth_max, n_slices + 1)
    mids = 0.5*(edges[1:] + edges[:-1])
    ratios = []
    alphas = np.linspace(-1.5, 3.0, 1001)
    best = None
    best_alpha = None
    for a in alphas:
        raw = np.power(mids, -a)
        w = raw / raw.sum()
        dollars = w * total_at_2pct
        def cum(thr):
            return dollars[mids<=thr].sum()
        err = (cum(0.5) - targets[0.5])**2 + (cum(1.0) - targets[1.0])**2
        if (best is None) or (err < best):
            best = err
            best_alpha = a
    return best_alpha

# ---------------------------- Sidebar Controls ----------------------------
st.sidebar.header("Global Inputs")

# Pools
primary_pool = st.sidebar.number_input("Primary Pool size (MGTN$)", value=10_000_000_000, step=100_000_000)
reward_pool  = st.sidebar.number_input("Reward Pool size (MGTN$)",  value=3_000_000_000, step=50_000_000)

# Revenue sharing / buybacks
buyback_pct = st.sidebar.slider("Revenue Sharing → Buyback rate (%)", 0, 50, 5) / 100.0

# Emissions
r0 = st.sidebar.slider("Initial annual emission rate (%)", 1, 40, 15) / 100.0
rf = st.sidebar.slider("Final annual emission rate (%) by Year 15", 0, 10, 2) / 100.0
half_life = st.sidebar.slider("Target half-life (years)", 1, 10, 6)
horizon_years = st.sidebar.slider("Horizon (years) for reward simulation", 5, 20, 15)

# Kaiko depth model
st.sidebar.markdown("---")
st.sidebar.subheader("Market Depth Model (Kaiko-style)")
k_ratio = st.sidebar.number_input("k (scaling)", value=1450, step=50)
beta = st.sidebar.slider("β (sublinear exponent)", 0.1, 1.0, 0.6, 0.05)
users_per_building = st.sidebar.slider("Users per building", 1, 10, 3)
max_buildings = st.sidebar.slider("Max buildings (x-axis range)", 100, 20000, 5000, step=100)

# Tier ratios
st.sidebar.caption("Tier 1 cumulative depth targets (of per-CEX allocation)")
d05 = st.sidebar.number_input("±0.5% depth (% of per-CEX)", value=4.0, step=0.5) / 100.0
d10 = st.sidebar.number_input("±1.0% depth (% of per-CEX)", value=12.0, step=0.5) / 100.0
d20 = st.sidebar.number_input("±2.0% depth (% of per-CEX)", value=36.0, step=0.5) / 100.0
cap_threshold = st.sidebar.radio("Cap is tied to which depth?", ["±1.0%", "±2.0%"], index=0)
ratio_choice = "1.0" if cap_threshold == "±1.0%" else "2.0"
ratio = tier_ratio(d05, d10, d20, choose=("1.0" if ratio_choice=="1.0" else "2.0"))

# DAO policy
st.sidebar.markdown("---")
st.sidebar.subheader("DAO Cap Policy")
cadence = st.sidebar.slider("Governance cadence (days)", 7, 60, 14, step=1)
trigger = st.sidebar.slider("EMA trigger threshold (%)", 1, 25, 5, step=1) / 100.0
max_step = st.sidebar.slider("Max cap step per decision (%)", 1, 50, 10, step=1) / 100.0
cap_floor = st.sidebar.number_input("Cap floor (MRT$ per user)", value=5.0, step=1.0)
cap_ceil = st.sidebar.number_input("Cap ceiling (MRT$ per user)", value=60.0, step=5.0)

# MM & CEX ladder
st.sidebar.markdown("---")
st.sidebar.subheader("MM Loan & CEX split")
loan_amount = st.sidebar.number_input("MM loan amount (USD)", value=3_000_000, step=100_000)
st.sidebar.caption("Weights (will be normalized):")
w_binance = st.sidebar.slider("Binance weight", 0.0, 1.0, 0.40, 0.05)
w_okx     = st.sidebar.slider("OKX weight",     0.0, 1.0, 0.25, 0.05)
w_coin    = st.sidebar.slider("Coinbase weight",0.0, 1.0, 0.20, 0.05)
w_bybit   = st.sidebar.slider("Bybit weight",   0.0, 1.0, 0.15, 0.05)

# Ladder
st.sidebar.subheader("Power-law Ladder (Tier 1)")
n_slices = st.sidebar.slider("Number of slices", 10, 60, 30, step=5)
depth_min = st.sidebar.number_input("Min depth (%)", value=0.05)
depth_max = st.sidebar.number_input("Max depth (%)", value=2.0)

# ---------------------------- Revenues Data ----------------------------
st.markdown(\"\"\"
### Revenues (editable)
Provide annual USD revenues by region (2026–2030). Values are evenly distributed across months (can be extended in code to seasonality).
\"\"\")
default_rev = pd.DataFrame({
    \"Region\": [\"EU\", \"USA\", \"UAE\"],
    \"2026\": [936_616, 1_023_689, 1_261_364],
    \"2027\": [4_886_217, 6_400_741, 3_795_668],
    \"2028\": [12_643_021, 21_055_270, 11_215_809],
    \"2029\": [28_004_583, 49_464_421, 32_570_042],
    \"2030\": [63_344_647, 117_463_586, 94_944_280],
})
rev_df = st.data_editor(default_rev, num_rows=\"dynamic\", use_container_width=True)
years = [c for c in rev_df.columns if c != \"Region\"]
annual_totals = rev_df[years].sum().to_numpy()
# 60 months for 5y
monthly_rev_5y = np.repeat(annual_totals/12.0, 12)

# Extend revenues to horizon: hold flat at Year-5 monthly
months_total = horizon_years * 12
steady = monthly_rev_5y[-1] if len(monthly_rev_5y)>0 else 0.0
monthly_revenue_series = np.concatenate([monthly_rev_5y, np.full(max(0, months_total-len(monthly_rev_5y)), steady)])

# ---------------------------- Tabs ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([\"Revenues & Pool\", \"DAO Cap\", \"Market Depth vs Buildings\", \"MM CEX & Ladder\", \"Downloads\"])

with tab1:
    st.markdown(\"#### Monthly Revenues by Region (USD) & Cumulative\")
    # Build monthly by region for the first 60 months
    month_axis_5y = np.arange(len(monthly_rev_5y))
    fig, ax = plt.subplots(figsize=(8,4))
    for idx, row in rev_df.iterrows():
        series = np.repeat(row[years].to_numpy()/12.0, 12)
        ax.plot(month_axis_5y, series, label=f\"{row['Region']} (monthly)\")
    ax.set_xlabel(\"Months since Jan 2026\"); ax.set_ylabel(\"USD\"); ax.legend(); ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(month_axis_5y, np.cumsum(np.sum([np.repeat(rev_df.iloc[i][years].to_numpy()/12.0, 12) for i in range(len(rev_df))], axis=0)), color=\"black\", linewidth=2, label=\"Cumulative\")
    ax2.set_ylabel(\"Cumulative (USD)\")
    st.pyplot(fig, clear_figure=True)

    st.markdown(\"#### Reward Pool Depletion (with Buybacks)\")
    months = months_total
    monthly_rate, annual_rate = build_emission_schedule(months, r0=r0, rf=rf, half_life_years=half_life)
    buybacks = monthly_revenue_series * buyback_pct  # 1 MGTN$ = $1
    bal_with_rev = simulate_pool(reward_pool, monthly_rate, buybacks)
    bal_no_rev   = simulate_pool(reward_pool, monthly_rate, np.zeros_like(buybacks))

    t_years = np.arange(months)/12.0
    fig2, ax = plt.subplots(figsize=(8,4))
    ax.plot(t_years, bal_no_rev/1e9, '--', label=\"No revenue sharing\")
    ax.plot(t_years, bal_with_rev/1e9, label=f\"Buyback {int(buyback_pct*100)}%\") 
    ax.set_xlabel(\"Years\"); ax.set_ylabel(\"Pool Balance (B MGTN$)\"); ax.grid(alpha=0.2); ax.legend(loc=\"upper right\")
    ax2 = ax.twinx(); ax2.plot(t_years, (annual_rate)*100, alpha=0.5, label=\"Annual emission rate\"); ax2.set_ylabel(\"Annual Rate (%)\")
    st.pyplot(fig2, clear_figure=True)

with tab2:
    st.markdown(\"#### DAO-Driven Cap per User (EMA-triggered)\")
    # Simulate 1-year daily EMA and cap policy
    days = 365
    # generate a synthetic depth ±1% evolution from monthly Kaiko target
    # Build a buildings trajectory (linear) to create users
    buildings = np.linspace(max_buildings*0.2, max_buildings*0.6, days)
    users = buildings * users_per_building
    # target depth from Kaiko (1% or 2% ratio)
    d2 = kaiko_depth(users, k_ratio, beta)
    d_target = d2 * (d10/d20 if ratio_choice==\"1.0\" else 1.0)
    # add some noise to get daily depth
    rng = np.random.default_rng(42)
    daily_depth = np.clip(d_target + rng.normal(0, d_target*0.03, size=days), 1e3, None)
    depth_ema = ema(daily_depth, span=14)
    cap_series = dao_cap_policy(depth_ema, users, cadence_days=cadence, trigger_pct=trigger,
                                max_step_pct=max_step, cap_floor=cap_floor, cap_ceil=cap_ceil)

    fig3, ax = plt.subplots(figsize=(8,4))
    ax.plot(daily_depth, alpha=0.35, label=\"Daily depth ±{}\".format(\"1%\" if ratio_choice==\"1.0\" else \"2%\"))
    ax.plot(depth_ema, label=\"EMA14d (USD)\")
    ax.set_xlabel(\"Day\"); ax.set_ylabel(\"Depth (USD)\"); ax.grid(alpha=0.2); ax.legend(loc=\"upper left\")
    ax2 = ax.twinx(); ax2.step(np.arange(days), cap_series, where='post', label=\"DAO Cap per User (MRT$)\"); ax2.set_ylabel(\"Cap per User (MRT$)\"); ax2.legend(loc=\"upper right\")
    st.pyplot(fig3, clear_figure=True)

with tab3:
    st.markdown(\"#### Depth Curves vs Buildings and Cap per User\")
    b = np.linspace(1, max_buildings, 200)
    u = b * users_per_building
    d2 = kaiko_depth(u, k_ratio, beta)
    r05, r10 = d05/d20, d10/d20
    fig4, ax = plt.subplots(figsize=(8,4))
    ax.plot(b, d2*r05, label=\"Depth ±0.5%\" )
    ax.plot(b, d2*r10, label=\"Depth ±1.0%\" )
    ax.plot(b, d2,      label=\"Depth ±2.0%\" )
    ax.set_xlabel(\"Buildings\"); ax.set_ylabel(\"Depth (USD)\"); ax.grid(alpha=0.2); ax.legend(loc=\"upper left\")
    cap_curve = cap_per_user(b, users_per_building, k_ratio, beta, (r10 if ratio_choice==\"1.0\" else 1.0))
    ax2 = ax.twinx(); ax2.plot(b, cap_curve, '--', label=\"Cap per user (MRT$)\"); ax2.set_ylabel(\"MRT$ per user\"); ax2.legend(loc=\"upper right\")
    st.pyplot(fig4, clear_figure=True)

with tab4:
    st.markdown(\"#### MM Loan Split across CEXs\")
    names = [\"Binance\",\"OKX\",\"Coinbase\",\"Bybit\"]
    weights = normalize_weights([w_binance, w_okx, w_coin, w_bybit])
    alloc = [loan_amount*w for w in weights]
    fig5, ax = plt.subplots(figsize=(7,3))
    ax.bar(names, alloc)
    for i, val in enumerate(alloc):
        ax.text(i, val, f\"${val:,.0f}\", ha=\"center\", va=\"bottom\", fontsize=9)
    ax.set_ylabel(\"USD allocated\"); ax.grid(axis='y', alpha=0.2)
    st.pyplot(fig5, clear_figure=True)

    st.markdown(\"#### Power-law Ladder Calibration (Tier 1)\")
    per_cex_alloc = alloc[0]  # first CEX
    targets = {0.5: d05*per_cex_alloc, 1.0: d10*per_cex_alloc, 2.0: d20*per_cex_alloc}
    alpha = ladder_alpha(depth_min, depth_max, n_slices, targets, total_at_2pct=targets[2.0])

    edges = np.linspace(depth_min, depth_max, n_slices+1)
    mids = 0.5*(edges[1:]+edges[:-1])
    raw = np.power(mids, -alpha)
    w = raw / raw.sum()
    dollars = w * targets[2.0]

    # plots
    fig6, ax = plt.subplots(figsize=(7,3))
    ax.bar(mids, dollars, width=(edges[1]-edges[0]))
    ax.set_xlabel(\"Depth (% from mid)\"); ax.set_ylabel(\"USD per slice (both sides)\"); ax.set_title(f\"First CEX ladder (alpha = {alpha:.3f})\"); ax.grid(alpha=0.2)
    st.pyplot(fig6, clear_figure=True)

    # cumulative check
    def cum(thr): return dollars[mids<=thr].sum()
    fig7, ax = plt.subplots(figsize=(7,3))
    sorted_m = np.sort(mids)
    cum_curve = [dollars[mids<=t].sum() for t in sorted_m]
    ax.plot(sorted_m, cum_curve, label=\"Achieved cumulative $\" )
    for thr in [0.5,1.0,2.0]:
        ax.axvline(thr, linestyle=\"--\"); ax.axhline(targets[thr], linestyle=\"--\")
        ax.text(thr, targets[thr], f\" {thr:.1f}% → ${targets[thr]:,.0f}\", va=\"bottom\", fontsize=9)
    ax.set_xlabel(\"Depth threshold (%)\"); ax.set_ylabel(\"Cumulative USD\"); ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig7, clear_figure=True)

with tab5:
    st.markdown(\"#### Downloads\")
    # Prepare data
    out_rev = pd.DataFrame({\"Month\": np.arange(len(monthly_revenue_series))+1, \"Monthly_Revenue_USD\": monthly_revenue_series})
    st.download_button(\"Download monthly revenues CSV\", data=out_rev.to_csv(index=False), file_name=\"monthly_revenues.csv\", mime=\"text/csv\")
    out_params = {
        \"primary_pool\": primary_pool, \"reward_pool\": reward_pool, \"buyback_pct\": buyback_pct,
        \"r0\": r0, \"rf\": rf, \"half_life\": half_life, \"k_ratio\": k_ratio, \"beta\": beta,
        \"users_per_building\": users_per_building, \"tier1_ratios\": {\"0.5%\": d05, \"1.0%\": d10, \"2.0%\": d20},
        \"loan_amount\": loan_amount, \"cex_weights\": dict(zip([\"Binance\",\"OKX\",\"Coinbase\",\"Bybit\"], weights)),
        \"ladder\": {\"n_slices\": n_slices, \"depth_min\": depth_min, \"depth_max\": depth_max, \"alpha\": float(alpha)},
        \"dao_policy\": {\"cadence_days\": cadence, \"trigger_pct\": trigger, \"max_step_pct\": max_step, \"cap_floor\": cap_floor, \"cap_ceil\": cap_ceil},
    }
    st.download_button(\"Download current parameters (JSON)\", data=pd.Series(out_params).to_json(), file_name=\"params.json\", mime=\"application/json\")

