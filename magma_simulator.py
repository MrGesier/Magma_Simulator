# app_v4.py — MGTN$ Economy Calibrator (v4: price dynamics + cadence)
# Run with: streamlit run app_v4.py

import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MGTN$ Economy Calibrator v4", layout="wide")

# ============================ Helpers ============================

def kaiko_depth(users, k, beta):
    users = np.asarray(users, dtype=float)
    return k * np.power(np.maximum(users, 1.0), beta)

def cap_per_user_from_depth(depth_usd, users, price_ref, floor, ceil):
    """Return cap per user in MRT$, given depth in USD, users, and a price reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(users > 0, depth_usd / (users * np.maximum(price_ref, 1e-9)), 0.0)
    return np.clip(cap, floor, ceil)

def ema(series, span=14):
    return pd.Series(series).ewm(span=span, adjust=False).mean().to_numpy()

def dao_cap_policy_from_series(cap_target_series, cadence_days=14, trigger_pct=0.05,
                               max_step_pct=0.10, cap_floor=1.0, cap_ceil=100.0):
    """
    Stepwise per-user cap (MRT$) updated at a fixed governance cadence.
    Only changes if the instantaneous target cap moved >= trigger_pct from the last anchor.
    Changes are limited to max_step_pct per decision and then clipped to [floor, ceil].
    """
    cap = np.zeros_like(cap_target_series, dtype=float)
    anchor = float(cap_target_series[0])
    cap_val = float(np.clip(cap_target_series[0], cap_floor, cap_ceil))
    for t in range(len(cap_target_series)):
        if t > 0 and t % cadence_days == 0:
            if anchor > 0 and abs(cap_target_series[t] - anchor) / anchor >= trigger_pct:
                desired = float(np.clip(cap_target_series[t], cap_floor, cap_ceil))
                delta = desired - cap_val
                step = math.copysign(min(abs(delta), max_step_pct * max(cap_val, 1e-9)), delta)
                cap_val += step
                anchor = float(cap_target_series[t])
        cap_val = float(np.clip(cap_val, cap_floor, cap_ceil))
        cap[t] = cap_val
    return cap

def build_emission_schedule(months, r0=0.15, rf=0.02, half_life_years=6.0):
    """
    Monthly emission rates that start near r0 (annual) and asymptotically approach rf (annual).
    Rates are scaled so the pool half-life (with zero buybacks) ≈ half_life_years.
    """
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
    """Return pool balance over time given per-month rate and per-month token injections."""
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
    """Search alpha for a power-law ladder matching cumulative targets at 0.5% and 1.0%."""
    edges = np.linspace(depth_min, depth_max, n_slices + 1)
    mids  = 0.5 * (edges[1:] + edges[:-1])
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

def gbm_price_path(days, p0=1.0, mu=0.0, sigma=0.2, seed=42):
    """Daily geometric Brownian motion: dP/P = mu*dt + sigma*sqrt(dt)*dW with dt=1/252."""
    rng = np.random.default_rng(seed)
    dt = 1/252
    shocks = rng.normal(0, 1, size=days)
    ret = (mu * dt) + (sigma * np.sqrt(dt) * shocks)
    logP = np.log(p0) + np.cumsum(ret)
    return np.exp(logP)

def monthly_from_daily(series_daily, months, days_per_month=30, how="sum"):
    """Aggregate a daily series into months of 30 days."""
    arr = np.asarray(series_daily, dtype=float)
    out = []
    ptr = 0
    for _ in range(months):
        chunk = arr[ptr:ptr+days_per_month]
        out.append(chunk.sum() if how == "sum" else chunk.mean())
        ptr += days_per_month
    return np.array(out)

# ============================ Sidebar ============================

st.title("MGTN$ Economy Calibrator — v4")
st.caption("Adds price dynamics, price-aware buybacks, and price-aware DAO cap with configurable cadence.")

st.sidebar.header("Global Inputs")

# Pools
primary_pool = st.sidebar.number_input("Primary Pool size (MGTN$)", value=10_000_000_000, step=100_000_000)
reward_pool  = st.sidebar.number_input("Reward Pool size (MGTN$)",  value=3_000_000_000, step=50_000_000)

# Price model
st.sidebar.markdown("---")
st.sidebar.subheader("MGTN$ Price Model (GBM)")
p0 = st.sidebar.number_input("Initial price (USD/MGTN$)", value=1.00, step=0.05, format="%.2f")
mu = st.sidebar.slider("Annual drift μ (%)", -50, 50, 0, step=1) / 100.0
sigma = st.sidebar.slider("Annual volatility σ (%)", 1, 200, 60, step=1) / 100.0
price_seed = st.sidebar.number_input("Random seed", value=42, step=1)

# Revenue sharing / buybacks
st.sidebar.markdown("---")
st.sidebar.subheader("Revenue Sharing → Buybacks")
buyback_pct = st.sidebar.slider("Buyback rate (% of revenues)", 0, 50, 5) / 100.0
buyback_cadence = st.sidebar.slider("Buyback cadence (days)", 1, 90, 30, step=1)
price_ref_for_cap = st.sidebar.selectbox("Price reference for DAO cap", ["EMA14d price", "Last price"], index=0)

# Emissions
st.sidebar.markdown("---")
st.sidebar.subheader("Reward Emissions")
r0 = st.sidebar.slider("Initial annual emission rate (%)", 1, 40, 15) / 100.0
rf = st.sidebar.slider("Final annual emission rate (%) by Year 15", 0, 10, 2) / 100.0
half_life = st.sidebar.slider("Target half-life (years)", 1, 10, 6)
horizon_years = st.sidebar.slider("Horizon (years) for reward simulation", 5, 20, 15)

# Kaiko depth model
st.sidebar.markdown("---")
st.sidebar.subheader("Market Depth Model (Kaiko)")
k_ratio = st.sidebar.number_input("k (scaling)", value=1450, step=50)
beta = st.sidebar.slider("β (sublinear exponent)", 0.1, 1.0, 0.6, 0.05)
users_per_building = st.sidebar.slider("Users per building", 1, 10, 3)
max_buildings = st.sidebar.slider("Max buildings (x-axis range)", 100, 20000, 5000, step=100)

# Tier ratios
st.sidebar.caption("Tier-1 cumulative depth targets (% of per-CEX allocation)")
d05 = st.sidebar.number_input("±0.5% depth (%)", value=4.0, step=0.5) / 100.0
d10 = st.sidebar.number_input("±1.0% depth (%)", value=12.0, step=0.5) / 100.0
d20 = st.sidebar.number_input("±2.0% depth (%)", value=36.0, step=0.5) / 100.0
cap_threshold = st.sidebar.radio("Cap is tied to which depth?", ["±1.0%", "±2.0%"], index=0)
ratio_choice = "1.0" if cap_threshold == "±1.0%" else "2.0"

# DAO policy
st.sidebar.markdown("---")
st.sidebar.subheader("DAO Cap Policy")
cadence = st.sidebar.slider("Governance cadence (days)", 7, 60, 14, step=1)
trigger = st.sidebar.slider("EMA trigger threshold (%)", 1, 25, 5, step=1) / 100.0
max_step = st.sidebar.slider("Max cap step per decision (%)", 1, 50, 10, step=1) / 100.0
cap_floor = st.sidebar.number_input("Cap floor (MRT$ per user)", value=5.0, step=1.0)
cap_ceil = st.sidebar.number_input("Cap ceiling (MRT$ per user)", value=60.0, step=5.0)

# ============================ Revenues Input ============================

st.markdown("### Revenues (editable)")
st.caption("Annual USD revenues by region (2026–2030). Evenly distributed per month; extended at Year-5 monthly level thereafter.")
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
months_total = horizon_years * 12
steady = monthly_rev_5y[-1] if len(monthly_rev_5y) else 0.0
monthly_revenue_series = np.concatenate([monthly_rev_5y, np.full(max(0, months_total - len(monthly_rev_5y)), steady)])

# ============================ Price & Buybacks (Daily) ============================

days_per_month = 30
days = months_total * days_per_month
price_path = gbm_price_path(days, p0=p0, mu=mu, sigma=sigma, seed=price_seed)
price_ema = ema(price_path, span=14)  # 14-day EMA
price_ref_series = price_ema if price_ref_for_cap == "EMA14d price" else price_path

# Build daily buyback USD stream and convert to tokens on chosen cadence
daily_usd_rev = np.repeat(monthly_revenue_series / days_per_month, days_per_month)
daily_buyback_accum = 0.0
daily_token_inj = np.zeros(days)
for t in range(days):
    daily_usd = daily_usd_rev[t] * buyback_pct
    daily_buyback_accum += daily_usd
    if t % buyback_cadence == 0:
        # Execute buyback: convert accumulated USD at today's price
        tokens = daily_buyback_accum / max(price_path[t], 1e-9)
        daily_token_inj[t] += tokens
        daily_buyback_accum = 0.0
# Flush any leftover on last day
if daily_buyback_accum > 0:
    daily_token_inj[-1] += daily_buyback_accum / max(price_path[-1], 1e-9)

# Convert to monthly token injections (sum of daily tokens within each month)
monthly_injection_tokens = monthly_from_daily(daily_token_inj, months_total, days_per_month=days_per_month, how="sum")

# ============================ Reward Pool Simulation (Monthly) ============================

monthly_rate, annual_rate = build_emission_schedule(months_total, r0=r0, rf=rf, half_life_years=half_life)
bal_with_rev = simulate_pool(reward_pool, monthly_rate, monthly_injection_tokens)
bal_no_rev   = simulate_pool(reward_pool, monthly_rate, np.zeros_like(monthly_injection_tokens))

# ============================ DAO Cap Impact with Price (Daily) ============================

# Daily depth using Kaiko based on buildings/users growth
b_series = np.linspace(max_buildings * 0.2, max_buildings * 0.6, days)
u_series = b_series * users_per_building
d2_daily = kaiko_depth(u_series, k_ratio, beta)
r10 = d10 / d20
depth_target_daily = d2_daily * (r10 if ratio_choice == "1.0" else 1.0)
depth_ema_daily = ema(depth_target_daily, span=14)

# Per-user instantaneous cap in MRT$ derived from depth and price reference
cap_target_per_user = cap_per_user_from_depth(depth_ema_daily, u_series, price_ref_series, cap_floor, cap_ceil)
cap_per_user_policy = dao_cap_policy_from_series(
    cap_target_per_user, cadence_days=cadence, trigger_pct=trigger,
    max_step_pct=max_step, cap_floor=cap_floor, cap_ceil=cap_ceil
)
allowed_total_usd = cap_per_user_policy * u_series * price_ref_series  # max aggregate in USD at the cap

# ============================ UI Pages ============================

page = st.radio(
    "Navigation",
    ["Dashboard", "Price & Buybacks", "Revenues & Reward Pool", "DAO Cap Impact", "Depth vs Buildings", "MM & Ladder", "Downloads"],
    horizontal=True
)

if page == "Dashboard":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Primary Pool (MGTN$)", f"{primary_pool:,.0f}")
    col2.metric("Reward Pool (MGTN$)", f"{reward_pool:,.0f}")
    col3.metric("Buyback rate", f"{int(buyback_pct*100)}%")
    col4.metric("Half-life target", f"{half_life} years")

    t_years = np.arange(months_total) / 12.0
    st.markdown("#### Reward Pool Depletion (Price-aware buybacks)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t_years, bal_no_rev / 1e9, "--", label="No revenue sharing")
    ax.plot(t_years, bal_with_rev / 1e9, label=f"Buybacks @ {int(buyback_pct*100)}%")
    ax.set_xlabel("Years"); ax.set_ylabel("Pool Balance (B MGTN$)"); ax.grid(alpha=0.2); ax.legend(loc="upper right")
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### DAO Cap vs Depth Headroom (Price-aware)")
    breach_days = int((depth_ema_daily - allowed_total_usd < 0).sum())
    colA, colB, colC = st.columns(3)
    colA.metric("Current price (USD)", f"{price_path[-1]:.3f}")
    colB.metric("Cap per user now (MRT$)", f"{cap_per_user_policy[-1]:.2f}")
    colC.metric("Days breaching depth", f"{breach_days} / {days}")

    fig2, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(depth_ema_daily, label="Depth target (EMA, USD)")
    ax.plot(allowed_total_usd, label="Max allowed @ DAO cap (USD)")
    ax.fill_between(np.arange(days), allowed_total_usd, depth_ema_daily,
                    where=(depth_ema_daily > allowed_total_usd), alpha=0.25, label="Headroom")
    ax.set_xlabel("Day"); ax.set_ylabel("USD"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig2, clear_figure=True)

elif page == "Price & Buybacks":
    st.subheader("Price Path (GBM) and Buyback Cadence")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_path, label="Price (daily)")
    ax.plot(price_ema, label="Price EMA14d")
    ax.set_xlabel("Day"); ax.set_ylabel("USD/MGTN$"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Token injections from buybacks (executed at cadence days)**")
    days_axis = np.arange(days)
    fig2, ax = plt.subplots(figsize=(10, 3.5))
    ax.stem(days_axis, daily_token_inj, use_line_collection=True, label="Token injection (daily)")
    ax.set_xlabel("Day"); ax.set_ylabel("Tokens injected (MGTN$)"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig2, clear_figure=True)

    st.caption("USD accumulates daily from revenues × buyback%. On cadence days, the MM converts the accumulated USD to tokens at the prevailing price, then reinjects into the Reward Pool.")

elif page == "Revenues & Reward Pool":
    st.subheader("Monthly Revenues by Region (USD) & Cumulative")
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

    t_years = np.arange(months_total) / 12.0
    st.subheader("Reward Pool Depletion (with/without buybacks)")
    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_years, bal_no_rev / 1e9, "--", label="No revenue sharing")
    ax.plot(t_years, bal_with_rev / 1e9, label=f"Buybacks @ {int(buyback_pct*100)}% (price-aware)")
    ax.set_xlabel("Years"); ax.set_ylabel("Pool Balance (B MGTN$)"); ax.grid(alpha=0.2); ax.legend(loc="upper right")
    ax2 = ax.twinx()
    ax2.plot(t_years, (annual_rate * 100)[:len(t_years)], alpha=0.5, label="Annual emission rate")
    ax2.set_ylabel("Annual Rate (%)")
    st.pyplot(fig2, clear_figure=True)

elif page == "DAO Cap Impact":
    st.subheader("How the DAO Cap Uses Price")
    st.write("Per-user cap = depth target (USD) / (users × price reference). DAO cadence/trigger/step + floor/ceiling produce the stepwise policy.")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cap_target_per_user, label="Instant target cap (MRT$/user)")
    ax.step(np.arange(days), cap_per_user_policy, where="post", label="DAO cap policy (MRT$/user)")
    ax.set_xlabel("Day"); ax.set_ylabel("MRT$ per user"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig, clear_figure=True)

    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(depth_ema_daily, label="Depth target (EMA, USD)")
    ax.plot(allowed_total_usd, label="Max allowed @ DAO cap (USD)")
    ax.fill_between(np.arange(days), allowed_total_usd, depth_ema_daily,
                    where=(depth_ema_daily > allowed_total_usd), alpha=0.25, label="Headroom")
    ax.set_xlabel("Day"); ax.set_ylabel("USD"); ax.grid(alpha=0.2); ax.legend()
    st.pyplot(fig2, clear_figure=True)

elif page == "Depth vs Buildings":
    st.subheader("Depth Curves vs Buildings (monthly view)")
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
    st.pyplot(fig4, clear_figure=True)

elif page == "MM & Ladder":
    st.subheader("MM Loan Split across CEXs")
    names = ["Binance", "OKX", "Coinbase", "Bybit"]
    weights = normalize_weights([0.40, 0.25, 0.20, 0.15])  # defaults; feel free to replace with sidebar if needed
    # If you want live weights, replace the line above by:
    # weights = normalize_weights([w_binance, w_okx, w_coin, w_bybit])
    loan_amount = 3_000_000
    alloc = [loan_amount * w for w in weights]
    fig5, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(names, alloc)
    for i, val in enumerate(alloc):
        ax.text(i, val, f"${val:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("USD allocated"); ax.grid(axis="y", alpha=0.2)
    st.pyplot(fig5, clear_figure=True)

    st.subheader("Power-law Ladder Calibration (Tier-1)")
    d05, d10, d20 = 0.04, 0.12, 0.36
    depth_min, depth_max, n_slices = 0.05, 2.0, 30
    per_cex_alloc = alloc[0]
    targets = {0.5: d05 * per_cex_alloc, 1.0: d10 * per_cex_alloc, 2.0: d20 * per_cex_alloc}
    alpha = ladder_alpha(depth_min, depth_max, n_slices, targets, total_at_2pct=targets[2.0])

    edges = np.linspace(depth_min, depth_max, n_slices + 1)
    mids  = 0.5 * (edges[1:] + edges[:-1])
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
        "price": {"p0": p0, "mu": mu, "sigma": sigma, "seed": price_seed,
                  "price_ref_for_cap": price_ref_for_cap, "buyback_cadence_days": buyback_cadence},
        "buyback_pct": buyback_pct,
        "emissions": {"r0": r0, "rf": rf, "half_life": half_life, "horizon_years": horizon_years},
        "kaiko": {"k_ratio": k_ratio, "beta": beta, "users_per_building": users_per_building},
        "tier1_ratios": {"0.5%": d05, "1.0%": d10, "2.0%": d20, "cap_threshold": cap_threshold},
        "dao_policy": {"cadence_days": cadence, "trigger_pct": trigger,
                       "max_step_pct": max_step, "cap_floor": cap_floor, "cap_ceil": cap_ceil},
    }
    st.download_button("Download current parameters (JSON)",
                       data=json.dumps(out_params, indent=2),
                       file_name="params.json",
                       mime="application/json")
