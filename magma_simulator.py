
# app_v3.py — MGTN$ Economy Calibrator (v3, with tooltips)
# Run with: streamlit run app_v3.py

import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MGTN$ Economy Calibrator v3", layout="wide")

# ============================ Helpers ============================

def kaiko_depth(users, k, beta):
    users = np.asarray(users, dtype=float)
    return k * np.power(np.maximum(users, 1.0), beta)

def tier_ratio(depth_05, depth_10, depth_20, choose="1.0"):
    r05 = depth_05 / depth_20 if depth_20 else 0.0
    r10 = depth_10 / depth_20 if depth_20 else 0.0
    if choose == "0.5":
        return r05
    if choose == "1.0":
        return r10
    return 1.0

def cap_per_user(buildings, users_per_building, k, beta, depth_ratio):
    u = np.asarray(buildings) * users_per_building
    d2 = kaiko_depth(u, k, beta)
    target = d2 * depth_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(u > 0, target / u, 0.0)
    return cap

def ema(series, span=14):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def dao_cap_policy(ema_depth, users_series, cadence_days=14, trigger_pct=0.05,
                   max_step_pct=0.10, cap_floor=1.0, cap_ceil=100.0):
    target = np.where(users_series > 0, ema_depth / users_series, 0.0)
    cap = np.zeros_like(target, dtype=float)
    anchor = float(ema_depth[0])
    cap_val = float(np.clip(target[0], cap_floor, cap_ceil))
    for t in range(len(target)):
        if t > 0 and t % cadence_days == 0:
            if anchor > 0 and abs(ema_depth[t] - anchor) / anchor >= trigger_pct:
                desired = float(np.clip(target[t], cap_floor, cap_ceil))
                delta = desired - cap_val
                step = math.copysign(min(abs(delta), max_step_pct * max(cap_val, 1e-9)), delta)
                cap_val += step
                anchor = float(ema_depth[t])
        cap_val = float(np.clip(cap_val, cap_floor, cap_ceil))
        cap[t] = cap_val
    return cap

def build_emission_schedule(months, r0=0.15, rf=0.02, half_life_years=6.0):
    t_years = np.linspace(0, months/12, months)
    k = -math.log(1e-4 / max(r0 - rf, 1e-9)) / max(t_years[-1], 1e-6)
    annual_rate = rf + (r0 - rf) * np.exp(-k * t_years)
    monthly_rate = annual_rate / 12.0

    def balance_after(mths, scale):
        bal = 1.0
        for i in range(mths):
            bal -= bal * monthly_rate[i] * scale
        return bal

    target_month = int(round(half_life_years * 12))
    lo, hi = 0.05, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if balance_after(target_month, mid) > 0.5:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
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
        return [1/len(weights)] * len(weights)
    return [w/s for w in weights]

def ladder_alpha(depth_min, depth_max, n_slices, targets, total_at_2pct):
    edges = np.linspace(depth_min, depth_max, n_slices + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    alphas = np.linspace(-1.5, 3.0, 1001)
    best_err, best_alpha = None, None
    for a in alphas:
        raw = np.power(mids, -a)
        w = raw / raw.sum()
        dollars = w * total_at_2pct
        cum = lambda thr: dollars[mids <= thr].sum()
        err = (cum(0.5) - targets[0.5])**2 + (cum(1.0) - targets[1.0])**2
        if best_err is None or err < best_err:
            best_err, best_alpha = err, a
    return float(best_alpha)

# ============================ Sidebar ============================

st.title("MGTN$ Economy Calibrator — v3")
st.caption("Tune pools, buybacks, emissions, market depth, DAO cap policy, and MM ladder. Hover the (?) icons for help.")

st.sidebar.header("Global Inputs")

# Pools
primary_pool = st.sidebar.number_input(
    "Primary Pool size (MGTN$)",
    value=10_000_000_000, step=100_000_000,
    help="Supply available for MRT$→MGTN$ conversions (primary swap / bonding curve)."
)
reward_pool  = st.sidebar.number_input(
    "Reward Pool size (MGTN$)",
    value=3_000_000_000, step=50_000_000,
    help="Pool distributed to holders as Inflationary Rewards (IR). Depletes per emission schedule; replenished by buybacks."
)

# Revenue sharing / buybacks
buyback_pct = st.sidebar.slider(
    "Revenue Sharing → Buyback rate (%)", 0, 50, 5, help="Percent of JV revenues converted to MGTN$ by the MM and re-injected into the Reward Pool each month."
) / 100.0

# Emissions
r0 = st.sidebar.slider("Initial annual emission rate (%)", 1, 40, 15, help="Starting annual emission rate from the Reward Pool (year 1).") / 100.0
rf = st.sidebar.slider("Final annual emission rate (%) by Year 15", 0, 10, 2, help="Asymptotic annual emission rate by the end of horizon.") / 100.0
half_life = st.sidebar.slider("Target half-life (years)", 1, 10, 6, help="Time for the Reward Pool to fall to 50% with zero buybacks. Used to scale the schedule.")
horizon_years = st.sidebar.slider("Horizon (years) for reward simulation", 5, 20, 15, help="Simulation horizon for pool depletion and buybacks.")

# Kaiko depth model
st.sidebar.markdown("---")
st.sidebar.subheader("Market Depth Model (Kaiko)")
k_ratio = st.sidebar.number_input("k (scaling)", value=1450, step=50, help="Scaling factor for depth: D2%(u) = k * u^β.")
beta = st.sidebar.slider("β (sublinear exponent)", 0.1, 1.0, 0.6, 0.05, help="Sublinear exponent for depth growth vs. users (0.6 typical).")
users_per_building = st.sidebar.slider("Users per building", 1, 10, 3, help="Map buildings to users: users = buildings × this value.")
max_buildings = st.sidebar.slider("Max buildings (x-axis range)", 100, 20000, 5000, step=100, help="Upper x-axis bound for buildings plots.")

# Tier ratios
st.sidebar.caption("Tier-1 cumulative depth targets (% of per-CEX allocation)")
d05 = st.sidebar.number_input("±0.5% depth (%)", value=4.0, step=0.5, help="Cumulative depth within ±0.5% of mid, as % of per-CEX allocation.") / 100.0
d10 = st.sidebar.number_input("±1.0% depth (%)", value=12.0, step=0.5, help="Cumulative depth within ±1.0% of mid, as % of per-CEX allocation.") / 100.0
d20 = st.sidebar.number_input("±2.0% depth (%)", value=36.0, step=0.5, help="Cumulative depth within ±2.0% of mid, as % of per-CEX allocation.") / 100.0
cap_threshold = st.sidebar.radio("Cap is tied to which depth?", ["±1.0%", "±2.0%"], index=0, help="Choose which depth threshold the DAO cap enforces against.")
ratio_choice = "1.0" if cap_threshold == "±1.0%" else "2.0"

# DAO policy
st.sidebar.markdown("---")
st.sidebar.subheader("DAO Cap Policy")
cadence = st.sidebar.slider("Governance cadence (days)", 7, 60, 14, step=1, help="How often the DAO can update the cap per user.")
trigger = st.sidebar.slider("EMA trigger threshold (%)", 1, 25, 5, step=1, help="Minimum EMA change from the last anchor to trigger an update.") / 100.0
max_step = st.sidebar.slider("Max cap step per decision (%)", 1, 50, 10, step=1, help="At each decision, cap can move toward the target by at most this percent of its current value.") / 100.0
cap_floor = st.sidebar.number_input("Cap floor (MRT$ per user)", value=5.0, step=1.0, help="Lower bound for the per-user cap. Prevents cap from shrinking to near-zero during thin liquidity.")
cap_ceil = st.sidebar.number_input("Cap ceiling (MRT$ per user)", value=60.0, step=5.0, help="Upper bound for the per-user cap. Prevents overly large caps during high-liquidity regimes.")

# MM & CEX ladder
st.sidebar.markdown("---")
st.sidebar.subheader("MM Loan & CEX split")
loan_amount = st.sidebar.number_input("MM loan amount (USD)", value=3_000_000, step=100_000, help="USD loan deployed by MM across CEXs to meet depth targets.")
st.sidebar.caption("Weights (normalized to 100%)")
w_binance = st.sidebar.slider("Binance weight", 0.0, 1.0, 0.40, 0.05, help="Share of MM loan allocated to Binance.")
w_okx     = st.sidebar.slider("OKX weight",     0.0, 1.0, 0.25, 0.05, help="Share of MM loan allocated to OKX.")
w_coin    = st.sidebar.slider("Coinbase weight",0.0, 1.0, 0.20, 0.05, help="Share of MM loan allocated to Coinbase.")
w_bybit   = st.sidebar.slider("Bybit weight",   0.0, 1.0, 0.15, 0.05, help="Share of MM loan allocated to Bybit.")

# Ladder
st.sidebar.subheader("Power-law Ladder (Tier-1)")
n_slices = st.sidebar.slider("Number of slices", 10, 60, 30, step=5, help="Number of depth buckets from min to max depth.")
depth_min = st.sidebar.number_input("Min depth (%)", value=0.05, help="Lower bound of ladder quoting range (from mid).")
depth_max = st.sidebar.number_input("Max depth (%)", value=2.0, help="Upper bound of ladder quoting range (from mid).")

# ---------------------------- Navigation ----------------------------
page = st.radio(
    "Navigation",
    ["Dashboard", "Revenues & Reward Pool", "DAO Cap Impact", "Depth vs Buildings", "MM & Ladder", "Downloads"],
    horizontal=True,
    help="Switch between analysis pages."
)

# ============================ Revenues Table ============================
st.markdown("### Revenues (editable)")
st.caption("Annual USD revenues by region (2026–2030). Evenly distributed per month (no seasonality).")
default_rev = pd.DataFrame({
    "Region": ["EU", "USA", "UAE"],
    "2026": [936_616, 1_023_689, 1_261_364],
    "2027": [4_886_217, 6_400_741, 3_795_668],
    "2028": [12_643_021, 21_055_270, 11_215_809],
    "2029": [28_004_583, 49_464_421, 32_570_042],
    "2030": [63_344_647, 117_463_586, 94_944_280],
})
rev_df = st.data_editor(default_rev, num_rows="dynamic", use_container_width=True)
years_cols = [c for c in rev_df.columns if c != "Region"]
annual_totals = rev_df[years_cols].sum().to_numpy()
monthly_rev_5y = np.repeat(annual_totals / 12.0, 12)  # 60 months
horizon_months = st.sidebar.slider("Horizon months (override)", 12, 240, int(15*12), step=12, help="For long simulations (reward pool).")
months_total = horizon_months
steady = monthly_rev_5y[-1] if len(monthly_rev_5y) else 0.0
monthly_revenue_series = np.concatenate([monthly_rev_5y, np.full(max(0, months_total - len(monthly_rev_5y)), steady)])

# ============================ Pages ============================

if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Primary Pool (MGTN$)", f"{primary_pool:,.0f}")
    col2.metric("Reward Pool (MGTN$)", f"{reward_pool:,.0f}")
    col3.metric("Buyback rate", f"{int(buyback_pct*100)}%")
    col4.metric("Half-life target", f"{half_life} years")

    monthly_rate, annual_rate = build_emission_schedule(months_total, r0=r0, rf=rf, half_life_years=half_life)
    buybacks = monthly_revenue_series * buyback_pct
    bal_with_rev = simulate_pool(reward_pool, monthly_rate, buybacks)
    t_years = np.arange(months_total) / 12.0
    st.markdown("#### Reward Pool Depletion (Quick View)")
    with st.expander("What am I looking at?"):
        st.write("Balance of the 3B MGTN$ Reward Pool over time with buybacks applied. Lower half-life or higher buybacks bend the curve upward.")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_years, bal_with_rev / 1e9, label=f"Buyback {int(buyback_pct*100)}%")
    ax.set_xlabel("Years"); ax.set_ylabel("Pool Balance (B MGTN$)"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### DAO Cap vs Depth Headroom (Quick View)")
    days = 365
    b_series = np.linspace(max_buildings * 0.2, max_buildings * 0.6, days)
    u_series = b_series * users_per_building
    d2 = kaiko_depth(u_series, k_ratio, beta)
    r10 = d10 / d20
    d_target = d2 * (r10 if ratio_choice == "1.0" else 1.0)
    noise = np.random.default_rng(42).normal(0, d_target * 0.03, size=days)
    daily_depth = np.clip(d_target + noise, 1e3, None)
    depth_ema = ema(daily_depth, span=14)
    cap_series = dao_cap_policy(depth_ema, u_series, cadence_days=cadence, trigger_pct=trigger,
                                max_step_pct=max_step, cap_floor=cap_floor, cap_ceil=cap_ceil)
    allowed = cap_series * u_series
    headroom = depth_ema - allowed
    breach_days = int((headroom < 0).sum())
    colA, colB, colC = st.columns(3)
    colA.metric("Current cap per user (MRT$)", f"{cap_series[-1]:.2f}")
    colB.metric("Headroom now (USD)", f"{headroom[-1]:,.0f}")
    colC.metric("Days breaching depth", f"{breach_days} / 365")
    with st.expander("What does the DAO cap change?"):
        st.write(
            "It converts a depth target (±1% or ±2%) into a per-user limit so that if all users sell, aggregate sales remain under market depth. "
            "Cadence/trigger make changes infrequent; step/floor/ceiling add guardrails."
        )
    fig2, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(depth_ema, label="Depth target (EMA)")
    ax.plot(allowed, label="Max allowed @ DAO cap")
    ax.fill_between(np.arange(days), allowed, depth_ema, where=(depth_ema>allowed), alpha=0.2, label="Headroom")
    ax.set_xlabel("Day"); ax.set_ylabel("USD"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig2, clear_figure=True)

elif page == "Revenues & Reward Pool":
    st.subheader("Monthly Revenues by Region (USD) & Cumulative")
    with st.expander("What am I looking at?"):
        st.write("Editable monthly revenue paths by region (flat within each year), plus cumulative. Drives buyback injections.")
    month_axis_5y = np.arange(len(np.repeat(annual_totals / 12.0, 12)))
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, row in rev_df.iterrows():
        series = np.repeat(row[years_cols].to_numpy() / 12.0, 12)
        ax.plot(month_axis_5y, series, label=f"{row['Region']} (monthly)")
    ax.set_xlabel("Months since Jan 2026"); ax.set_ylabel("USD"); ax.grid(alpha=0.2); ax.legend()
    ax2 = ax.twinx()
    stacked = np.sum([np.repeat(rev_df.iloc[i][years_cols].to_numpy()/12.0, 12) for i in range(len(rev_df))], axis=0)
    ax2.plot(month_axis_5y, np.cumsum(stacked), color="black", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative (USD)")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Reward Pool Depletion (with Buybacks)")
    with st.expander("What am I looking at?"):
        st.write("Reward Pool balance with and without buybacks. Blue line (right axis) is the annual emission rate used to compute monthly payouts.")
    monthly_rate, annual_rate = build_emission_schedule(months_total, r0=r0, rf=rf, half_life_years=half_life)
    buybacks = monthly_revenue_series * buyback_pct
    bal_with_rev = simulate_pool(reward_pool, monthly_rate, buybacks)
    bal_no_rev   = simulate_pool(reward_pool, monthly_rate, np.zeros_like(buybacks))
    t_years = np.arange(months_total) / 12.0
    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_years, bal_no_rev / 1e9, "--", label="No revenue sharing")
    ax.plot(t_years, bal_with_rev / 1e9, label=f"Buyback {int(buyback_pct*100)}%")
    ax.set_xlabel("Years"); ax.set_ylabel("Pool Balance (B MGTN$)"); ax.grid(alpha=0.2); ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(t_years, annual_rate * 100, alpha=0.5, label="Annual emission rate")
    ax2.set_ylabel("Annual Rate (%)")
    st.pyplot(fig2, clear_figure=True)

elif page == "DAO Cap Impact":
    st.subheader("How the DAO Cap Changes the System")
    st.caption("Cap per user is computed from the EMA depth target so that aggregate sales ≤ depth; guardrails: cadence, trigger, step, floor/ceiling.")
    days = 365
    b0 = st.slider("Start buildings", 100, max_buildings, int(max_buildings*0.2), step=50, help="Initial building count for the simulation window.")
    b1 = st.slider("End buildings", b0, max_buildings, int(max_buildings*0.6), step=50, help="Final building count for the simulation window.")
    b_series = np.linspace(b0, b1, days)
    u_series = b_series * users_per_building
    d2 = kaiko_depth(u_series, k_ratio, beta)
    r10 = d10 / d20
    d_target = d2 * (r10 if ratio_choice == "1.0" else 1.0)
    noise = np.random.default_rng(1).normal(0, d_target * 0.03, size=days)
    daily_depth = np.clip(d_target + noise, 1e3, None)
    depth_ema = ema(daily_depth, span=14)
    cap_series = dao_cap_policy(depth_ema, u_series, cadence_days=cadence, trigger_pct=trigger,
                                max_step_pct=max_step, cap_floor=cap_floor, cap_ceil=cap_ceil)
    allowed = cap_series * u_series
    headroom = depth_ema - allowed
    breach_days = int((headroom < 0).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current cap per user (MRT$)", f"{cap_series[-1]:.2f}")
    col2.metric("Allowed aggregate now (USD)", f"{allowed[-1]:,.0f}")
    col3.metric("Depth (EMA) now (USD)", f"{depth_ema[-1]:,.0f}")
    col4.metric("Days breaching depth", f"{breach_days} / 365")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(depth_ema, label="Depth target (EMA)")
    ax.plot(allowed, label="Max allowed @ DAO cap")
    ax.fill_between(np.arange(days), allowed, depth_ema, where=(depth_ema>allowed), alpha=0.25, label="Headroom")
    ax.set_xlabel("Day"); ax.set_ylabel("USD"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig, clear_figure=True)

elif page == "Depth vs Buildings":
    st.subheader("Depth Curves vs Buildings and Cap per User")
    with st.expander("What am I looking at?"):
        st.write("Kaiko depth curves (±0.5/1/2%) vs buildings, plus the implied cap per user that guarantees aggregate sales ≤ chosen depth.")
    b = np.linspace(1, max_buildings, 200)
    u = b * users_per_building
    d2 = kaiko_depth(u, k_ratio, beta)
    r05 = d05 / d20
    r10 = d10 / d20
    fig4, ax = plt.subplots(figsize=(9, 4))
    ax.plot(b, d2 * r05, label="Depth ±0.5%")
    ax.plot(b, d2 * r10, label="Depth ±1.0%")
    ax.plot(b, d2, label="Depth ±2.0%")
    ax.set_xlabel("Buildings"); ax.set_ylabel("Depth (USD)"); ax.grid(alpha=0.2); ax.legend(loc="upper left")
    cap_curve = cap_per_user(b, users_per_building, k_ratio, beta, (r10 if ratio_choice == "1.0" else 1.0))
    ax2 = ax.twinx()
    ax2.plot(b, cap_curve, "--", label="Cap per user (MRT$)")
    ax2.set_ylabel("MRT$ per user"); ax2.legend(loc="upper right")
    st.pyplot(fig4, clear_figure=True)

elif page == "MM & Ladder":
    st.subheader("MM Loan Split across CEXs")
    with st.expander("What am I looking at?"):
        st.write("Loan allocation across CEXs; used to compute per-exchange depth targets and ladder calibration.")
    names = ["Binance", "OKX", "Coinbase", "Bybit"]
    weights = normalize_weights([w_binance, w_okx, w_coin, w_bybit])
    alloc = [loan_amount * w for w in weights]
    fig5, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(names, alloc)
    for i, val in enumerate(alloc):
        ax.text(i, val, f"${val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("USD allocated"); ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig5, clear_figure=True)

    st.subheader("Power-law Ladder Calibration (Tier-1)")
    with st.expander("What am I looking at?"):
        st.write("USD per slice allocated across depth buckets such that cumulative depth matches Tier-1 targets at ±0.5% and ±1.0%.")
    per_cex_alloc = alloc[0]
    targets = {0.5: d05 * per_cex_alloc, 1.0: d10 * per_cex_alloc, 2.0: d20 * per_cex_alloc}
    alpha = ladder_alpha(depth_min, depth_max, n_slices, targets, total_at_2pct=targets[2.0])

    edges = np.linspace(depth_min, depth_max, n_slices + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    raw = np.power(mids, -alpha)
    w = raw / raw.sum()
    dollars = w * targets[2.0]

    fig6, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(mids, dollars, width=(edges[1] - edges[0]))
    ax.set_xlabel("Depth (% from mid)"); ax.set_ylabel("USD per slice (both sides)")
    ax.set_title(f"First CEX ladder (alpha = {alpha:.3f})"); ax.grid(alpha=0.2)
    st.pyplot(fig6, clear_figure=True)

    fig7, ax = plt.subplots(figsize=(9, 3.5))
    sorted_m = np.sort(mids)
    cum_curve = [dollars[mids <= t].sum() for t in sorted_m]
    ax.plot(sorted_m, cum_curve, label="Achieved cumulative $")
    for thr in [0.5, 1.0, 2.0]:
        ax.axvline(thr, linestyle="--"); ax.axhline(targets[thr], linestyle="--")
        ax.text(thr, targets[thr], f" {thr:.1f}% → ${targets[thr]:,.0f}", va="bottom", fontsize=9)
    ax.set_xlabel("Depth threshold (%)"); ax.set_ylabel("Cumulative USD"); ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig7, clear_figure=True)

elif page == "Downloads":
    st.subheader("Downloads")
    out_rev = pd.DataFrame({"Month": np.arange(len(monthly_revenue_series)) + 1,
                            "Monthly_Revenue_USD": monthly_revenue_series})
    st.download_button("Download monthly revenues CSV",
                       data=out_rev.to_csv(index=False),
                       file_name="monthly_revenues.csv",
                       mime="text/csv")
    out_params = {
        "primary_pool": primary_pool,
        "reward_pool": reward_pool,
        "buyback_pct": buyback_pct,
        "emissions": {"r0": r0, "rf": rf, "half_life": half_life, "horizon_months": months_total},
        "kaiko": {"k_ratio": k_ratio, "beta": beta, "users_per_building": users_per_building},
        "tier1_ratios": {"0.5%": d05, "1.0%": d10, "2.0%": d20, "cap_threshold": cap_threshold},
        "dao_policy": {"cadence_days": cadence, "trigger_pct": trigger,
                       "max_step_pct": max_step, "cap_floor": cap_floor, "cap_ceil": cap_ceil},
        "mm": {"loan_amount": loan_amount,
               "cex_weights": dict(zip(["Binance", "OKX", "Coinbase", "Bybit"], normalize_weights([w_binance, w_okx, w_coin, w_bybit])))}
    }
    st.download_button("Download current parameters (JSON)",
                       data=json.dumps(out_params, indent=2),
                       file_name="params.json",
                       mime="application/json")
