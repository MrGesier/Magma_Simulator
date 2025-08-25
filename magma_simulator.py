# app_magma_flowdesk_buyback_split.py
# Run: streamlit run app_magma_flowdesk_buyback_split.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# --- compatibility for older Streamlit (column_config may not exist) -----------
try:
    from streamlit import column_config as cc
    _HAS_COLCONF = True
except Exception:
    _HAS_COLCONF = False

st.set_page_config(page_title="Magma — Overflow + Buybacks (split absorption)", layout="wide")
st.title("Unlock → Absorption (Dyn. Liquidity vs Buybacks) → Overflow Impact on Price")

DT = 1/365.0
N  = 100  # Monte Carlo sims (median price)

# ===================== Sidebar: Core ===========================================
st.sidebar.header("Core")
TOTAL_SUPPLY  = st.sidebar.number_input(
    "Total Supply (tokens)", value=1_000_000_000, step=100_000_000,
    help="Offre totale de tokens (ex. 1, 5 ou 10 milliards)."
)
INITIAL_PRICE = st.sidebar.number_input(
    "Initial Price (USD)", value=0.008, step=0.001, format="%.4f",
    help="Prix de listing à TGE (USD par token)."
)
INIT_CIRC_PCT = st.sidebar.slider(
    "Initial Circulating % (listing)", 0.0, 100.0, 12.2, 0.1,
    help="Pourcentage de l’offre totale déjà en circulation au listing (incl. TGE + allocations initiales)."
)
DAYS = st.sidebar.number_input(
    "Horizon (days)", value=730, min_value=180, step=30,
    help="Durée de simulation en jours (ex. 730 ≈ 24 mois)."
)

# ===================== Price regimes (GBM) =====================================
st.sidebar.header("GBM (base regime)")
MU_BASE    = st.sidebar.number_input(
    "μ_base (annualized)", value=0.25, step=0.01,
    help="Drift annualisé du prix (régime normal) utilisé dans le GBM."
)
SIGMA_BASE = st.sidebar.number_input(
    "σ_base (annualized)", value=0.60, step=0.01,
    help="Volatilité annualisée (régime normal) utilisée dans le GBM."
)

st.sidebar.header("Overflow Stress Window")
OVERFLOW_WINDOW_DAYS = st.sidebar.number_input(
    "Window length (days)", value=14, min_value=1, step=1,
    help="Durée (en jours) d’un épisode de stress déclenché par un overflow."
)
MU_OVERFLOW = st.sidebar.number_input(
    "μ_overflow (annualized)", value=-0.20, step=0.05,
    help="Drift annualisé appliqué pendant les fenêtres d’overflow."
)
SIGMA_MULT_OVERFLOW = st.sidebar.number_input(
    "σ multiplier (overflow)", value=1.5, step=0.1,
    help="Multiplicateur appliqué à σ_base pendant un overflow."
)

st.sidebar.header("Bear-Market Windows (optional)")
BEAR_INPUT = st.sidebar.text_area(
    "Bear windows [(start,end, mu, sigma_mult), ...]", value="",
    help="Liste d’intervalles où l’on impose un régime baissier spécifique. "
         "Exemple : [(60,180,-0.35,1.3)] = du jour 60 au 180, μ=-0.35 et σ=1.3×σ_base."
)
try:
    BEAR_WINDOWS = eval(BEAR_INPUT) if BEAR_INPUT.strip() else []
    if not isinstance(BEAR_WINDOWS, list):
        raise ValueError
except Exception:
    st.sidebar.error("Invalid bear windows. Example: [(60,180,-0.35,1.3)]")
    BEAR_WINDOWS = []

# ===================== Liquidity (absorber) ====================================
st.sidebar.header("Dynamic Liquidity (absorber)")
INITIAL_DEPOSIT = st.sidebar.number_input(
    "Initial Liquidity Deposit (USD)", value=500_000, step=100_000,
    help="Dépôt de liquidité initial (absorbeur n°1)."
)
LIQ_WINDOWS_STR = st.sidebar.text_area(
    "Dyn. Liquidity Windows  [(start_day,end_day,USD), ...]",
    value="[(30, 540, 1_000_000)]",
    help="Fenêtres de liquidité dynamique. La somme (USD) est répartie uniformément entre start_day et end_day inclus."
)
try:
    LIQ_WINDOWS = eval(LIQ_WINDOWS_STR) if LIQ_WINDOWS_STR.strip() else []
    if not isinstance(LIQ_WINDOWS, list):
        raise ValueError
except Exception:
    st.sidebar.error("Invalid format. Example: [(30,540,1_000_000)].")
    LIQ_WINDOWS = []

# ===================== Buybacks from revenues ==================================
st.sidebar.header("Buybacks from Revenues")
st.sidebar.caption("Les buybacks sont un absorbeur séparé de la liquidité dynamique.")
START_YEAR = st.sidebar.number_input(
    "Simulation start year", value=2026, step=1,
    help="Année de départ de la table de revenus."
)
DEFAULT_BUYBACK_PCT = st.sidebar.slider(
    "Default Buyback % of revenues", 0.0, 100.0, 5.0, 0.5,
    help="Pourcentage des revenus alloué aux rachats (hors fenêtres spécifiques)."
)/100.0
FX_EURUSD = st.sidebar.number_input(
    "EUR→USD FX", value=1.08, step=0.01,
    help="Taux de conversion EUR→USD appliqué aux revenus."
)

st.sidebar.subheader("Buyback Windows (manual triggers)")
st.sidebar.caption("Format: [(start_day,end_day,pct), ...]. Remplace le pourcentage par défaut sur ces périodes.")
BUYBACK_WINDOWS_STR = st.sidebar.text_area(
    "[(start_day,end_day,pct), ...]", value="",
    help="Fenêtres de rachat manuelles (pct exprimé en fraction : 0.05 = 5%)."
)
try:
    BUYBACK_WINDOWS = eval(BUYBACK_WINDOWS_STR) if BUYBACK_WINDOWS_STR.strip() else []
    if not isinstance(BUYBACK_WINDOWS, list):
        raise ValueError
except Exception:
    st.sidebar.error("Invalid buyback windows. Example: [(0,364,0.05),(365,729,0.10)]")
    BUYBACK_WINDOWS = []

# ---------------- Revenues table (editable) ------------------------------------
st.write("#### Annual Revenues (editable) → daily buybacks")
st.caption(
    "Conversion EU/USA/UAE (EUR) en USD/jour **progressifs et continus** : "
    "• Année 1 part de 0/jour et monte linéairement. "
    "• Années suivantes **démarrent au niveau de fin** de l’année précédente. "
    "Pour chaque année, la **somme des 365 jours = total annuel** (ancres respectées)."
)

rev_cols = ["Year", "EU (EUR)", "USA (EUR)", "UAE (EUR)"]
rev_data = [
    [2026, 1_101_901, 1_204_340, 1_483_958],
    [2027, 5_748_490, 7_530_284, 4_465_492],
    [2028,14_874_143,24_770_906,13_195_069],
    [2029,32_946_569,58_193_436,38_317_697],
    [2030,74_523_115,138_192_454,111_699_153],
]
_rev_df = pd.DataFrame(rev_data, columns=rev_cols)
if _HAS_COLCONF:
    rev_df = st.data_editor(
        _rev_df, use_container_width=True, num_rows="dynamic",
        column_config={
            "Year": cc.NumberColumn("Year", help="Année fiscale (AAAA)."),
            "EU (EUR)": cc.NumberColumn("EU (EUR)", help="Revenus annuels Union Européenne en EUR."),
            "USA (EUR)": cc.NumberColumn("USA (EUR)", help="Revenus annuels USA en EUR."),
            "UAE (EUR)": cc.NumberColumn("UAE (EUR)", help="Revenus annuels EAU en EUR."),
        },
    )
else:
    st.caption("Tip: for column tooltips, use Streamlit ≥ 1.25.")
    rev_df = st.data_editor(_rev_df, use_container_width=True)

# >>>>>>>>>>>>>>> NEW: revenues progressifs ET CONTINUS par année <<<<<<<<<<<<<<<
def build_daily_revenue_usd_progressive_continuous(
    T: int, start_year: int, df_rev: pd.DataFrame, eurusd: float
) -> np.ndarray:
    """
    Série quotidienne des revenus en USD, continue d'une année à l'autre.
    Année 1 : part de 0 et grimpe linéairement.
    Année n>1 : démarre à la valeur de fin de l'année n-1 (continuité),
    puis grimpe linéairement. Pour D=365, si A est le total annuel en USD,
    et r_start la valeur journalière du 1er janvier :
        r_end = 2*A/D - r_start
    On interpole sur D jours : r(d) = r_start + (r_end - r_start) * d/(D-1).
    La somme discrète vaut ~ D*(r_start + r_end)/2 = A (exacte avec D constant).
    """
    daily = np.zeros(T, dtype=float)
    if df_rev.empty:
        return daily

    df_sorted = df_rev.sort_values("Year")
    years = [int(y) for y in df_sorted["Year"].tolist() if int(y) >= start_year]
    if not years:
        return daily

    r_start_next = 0.0  # année 1 démarre à 0/jour

    for year in years:
        idx_start = (year - start_year) * 365
        idx_end   = idx_start + 365
        if idx_start >= T:
            break

        row = df_sorted[df_sorted["Year"] == year].iloc[0]
        total_eur = float(row["EU (EUR)"] + row["USA (EUR)"] + row["UAE (EUR)"])
        total_usd = total_eur * float(eurusd)

        D = 365
        r_start = float(r_start_next)
        r_end   = (2.0 * total_usd) / D - r_start  # garantit l'aire annuelle = total_usd

        # sécurité minimale : si jamais r_end < 0 (rare si revenus croissent), on écrase à 0
        # et on met un profil plat = total_usd/D pour éviter des valeurs négatives.
        if r_end < 0:
            r_start = 0.0
            r_end   = (2.0 * total_usd) / D

        days_here = max(0, min(idx_end, T) - idx_start)
        if days_here <= 0:
            r_start_next = r_end
            continue

        if D > 1:
            for d in range(days_here):
                x = d / (D - 1)
                rate = r_start + (r_end - r_start) * x
                daily[idx_start + d] += max(rate, 0.0)
        else:
            daily[idx_start] += total_usd

        # continuité pour l'année suivante
        r_start_next = r_end

    return daily

def apply_buyback_windows(revenue_daily_usd: np.ndarray, default_pct: float, windows: list, T: int) -> np.ndarray:
    """% par défaut partout, remplacé sur les fenêtres [(start,end,pct), ...]."""
    buyback = revenue_daily_usd * float(default_pct)
    for (s, e, pct) in windows:
        s = max(int(s), 0); e = min(int(e), T-1)
        if e >= s:
            buyback[s:e+1] = revenue_daily_usd[s:e+1] * float(pct)
    return buyback

# ====== Revenus journaliers progressifs & continus, puis buybacks =====
revenue_daily_usd = build_daily_revenue_usd_progressive_continuous(
    int(DAYS), START_YEAR, rev_df, FX_EURUSD
)
buyback_daily     = apply_buyback_windows(revenue_daily_usd, DEFAULT_BUYBACK_PCT, BUYBACK_WINDOWS, int(DAYS))

# ===================== Static depth (Flowdesk calibration) ======================
st.sidebar.header("Static Depth (Flowdesk calibration)")
st.sidebar.caption("La static depth est un SEUIL (non-absorbant) ajusté par loi de puissance MC→Depth@2%.")
MC_points_str    = st.sidebar.text_input(
    "MC points (USD)",    "44e6, 68e6, 92e6, 116e6",
    help="Points de calibration Market Cap en USD, séparés par des virgules."
)
Depth_points_str = st.sidebar.text_input(
    "Depth@2% (USD)",     "0.9e6, 1.3e6, 1.8e6, 2.3e6",
    help="Profondeur à ±2% correspondante (USD), séparée par des virgules."
)
try:
    MC_points    = np.array([float(x) for x in MC_points_str.split(",")])
    Depth_points = np.array([float(x) for x in Depth_points_str.split(",")])
    assert len(MC_points) == len(Depth_points) and len(MC_points) >= 2
except Exception:
    st.sidebar.error("Bad calibration points; using defaults.")
    MC_points    = np.array([44e6, 68e6, 92e6, 116e6])
    Depth_points = np.array([0.9e6, 1.3e6, 1.8e6, 2.3e6])

coeffs    = np.polyfit(np.log(MC_points), np.log(Depth_points), 1)
beta_fit  = float(coeffs[0])
k_fit     = float(np.exp(coeffs[1]))
def static_depth(market_cap_usd: float) -> float:
    return float(k_fit * (max(market_cap_usd, 0.0) ** beta_fit))

# ===================== Allocations (blue-sheet style) ===========================
st.write("#### Allocations & Triggering (blue-sheet style, editable)")
alloc_cols = [
    "Category","Alloc %","TGE %","Lockup (mo)","Vesting (mo)","Entry Price",
    "Default SP %","Triggered SP %","Trigger ROI %","Sellable?"
]
alloc_preset = [
    ["Team",                 10.0,  0.0, 12, 36, 0.0000,   5.0, 50.0, 100.0, False],
    ["Advisors",              1.0,  0.0,  3,  9, 0.0000,  10.0, 60.0, 100.0, False],
    ["Private 1 (Nexera)",    0.6, 10.0,  1, 11, 0.0015,  20.0, 95.0, 110.0, True ],
    ["Private 2 (F&F)",       8.0,  5.0,  1, 11, 0.0023,  20.0, 95.0, 110.0, True ],
    ["Seed",                  6.0,  9.1,  0, 23, 0.0033,  20.0, 90.0, 110.0, True ],
    ["Institutionals (A)",    8.0,  1.8,  0, 23, 0.0045,  25.0, 90.0, 100.0, True ],
    ["ICO#1",                 1.0, 25.0,  1,  9, 0.0050,  40.0, 90.0,  90.0, True ],
    ["Airdrop",               1.0, 20.0,  1,  5, 0.0000,  50.0, 95.0, 100.0, True ],
    ["Rewards Conversation", 20.0,  1.8,  0,120, 0.0000,  30.0, 70.0,  80.0, True ],
    ["Rewards Staking",      10.0,  1.8,  0, 60, 0.0000,  20.0, 60.0,  80.0, True ],
    ["Liquidity (CEX/DEX)",  10.0,  1.7,  0,  0, 0.0080,   0.0,  0.0,   0.0, False],
    ["Ecosystem + Marketing",15.0,  1.7,  0, 60, 0.0000,  15.0, 60.0,  90.0, True ],
    ["Foundation (DAO)",      5.0,100.0,  0,  0, 0.0000,   0.0,  0.0,   0.0, False],
]
_alloc_df = pd.DataFrame(alloc_preset, columns=alloc_cols)
if _HAS_COLCONF:
    df = st.data_editor(
        _alloc_df, use_container_width=True, num_rows="dynamic",
        column_config={
            "Category": cc.TextColumn("Category", help="Nom de l’allocation/pool."),
            "Alloc %": cc.NumberColumn("Alloc %", help="% de l’offre totale attribué à ce pool."),
            "TGE %": cc.NumberColumn("TGE %", help="% de ce pool débloqué au TGE (jour 0)."),
            "Lockup (mo)": cc.NumberColumn("Lockup (mo)", help="Période de lock (mois) avant le début du vesting."),
            "Vesting (mo)": cc.NumberColumn("Vesting (mo)", help="Durée de vesting linéaire (mois) après lock."),
            "Entry Price": cc.NumberColumn("Entry Price", help="Prix d’entrée / coût par token pour ce pool, si pertinent."),
            "Default SP %": cc.NumberColumn("Default SP %", help="% vendable des unlocks (sell pressure) en régime normal."),
            "Triggered SP %": cc.NumberColumn("Triggered SP %", help="% vendable si ROI déclenche le mode ‘triggered’."),
            "Trigger ROI %": cc.NumberColumn("Trigger ROI %", help="Seuil de ROI (%) qui active la colonne ‘Triggered SP %’."),
            "Sellable?": cc.CheckboxColumn("Sellable?", help="Ce pool génère-t-il de la pression vendeuse ?"),
        },
    )
else:
    df = st.data_editor(_alloc_df, use_container_width=True)

# ===================== Build unlocks & circulating ==============================
T = int(DAYS)
t = np.arange(T)
C = len(df)

unlocked = np.zeros((C, T), dtype=float)
for i, row in df.iterrows():
    alloc_tokens = float(row["Alloc %"])/100.0 * TOTAL_SUPPLY
    tge         = float(row["TGE %"])/100.0
    lock_days   = int(row["Lockup (mo)"]) * 30
    vest_days   = int(row["Vesting (mo)"]) * 30
    if tge > 0:
        unlocked[i, 0] += alloc_tokens * tge
    if vest_days > 0:
        daily = (alloc_tokens * (1.0 - tge)) / vest_days
        start, end = lock_days, min(T, lock_days + vest_days)
        if end > start:
            unlocked[i, start:end] += daily

sellable = df["Sellable?"].astype(bool).to_numpy()
entry    = df["Entry Price"].astype(float).to_numpy()
d_sp     = (df["Default SP %"].astype(float).to_numpy())/100.0
t_sp     = (df["Triggered SP %"].astype(float).to_numpy())/100.0
thr      =  df["Trigger ROI %"].astype(float).to_numpy()

circ = np.zeros(T)
circ[0] = (INIT_CIRC_PCT/100.0) * TOTAL_SUPPLY + unlocked[:, 0].sum()
for d in range(1, T):
    circ[d] = circ[d-1] + unlocked[:, d].sum()

# ===================== One simulation (split absorption) =======================
def run_one(seed: int):
    rng = np.random.default_rng(seed)
    price = np.zeros(T); price[0] = INITIAL_PRICE

    queue_tokens_by_cat = np.zeros(C)  # leftovers queue

    # Outputs
    realized_usd_by_cat = np.zeros((C, T))
    desired_usd         = np.zeros(T)
    absorbed_liq_usd    = np.zeros(T)  # by dynamic liquidity
    absorbed_bb_usd     = np.zeros(T)  # by buybacks
    overflow_usd        = np.zeros(T)
    static_depth_series = np.zeros(T)
    dyn_liq_series      = np.zeros(T)  # dynamic liquidity line (no buybacks)
    buyback_series      = np.zeros(T)  # buybacks per day (for plotting)

    overflow_days_left = 0

    for day in range(T):
        p  = price[day-1] if day > 0 else price[0]
        mc = p * circ[day]

        # (1) Static depth (threshold, not absorber)
        depth_s = static_depth(mc)
        static_depth_series[day] = depth_s

        # (2) Dynamic liquidity (absorber) — deposit + windows (NO buybacks here)
        Ldyn_liq = 0.0
        if day == 0:
            Ldyn_liq += INITIAL_DEPOSIT
        for (s, e, usd) in LIQ_WINDOWS:
            if int(s) <= day <= int(e):
                Ldyn_liq += float(usd)
        dyn_liq_series[day] = Ldyn_liq

        # (2b) Buybacks (absorber) — separate component
        Ldyn_bb = float(buyback_daily[day])
        buyback_series[day] = Ldyn_bb

        # (3) Triggering → push to queue (ROI safe; only sellable pools)
        Cn = len(entry)
        roi = np.full(Cn, -np.inf, dtype=float)
        valid = (entry > 0.0) & sellable
        roi[valid] = (p / entry[valid] - 1.0) * 100.0

        sp_today = np.where(roi > thr, t_sp, d_sp)
        sp_today = np.where(sellable, sp_today, 0.0)

        to_queue = unlocked[:, day] * sp_today
        queue_tokens_by_cat += to_queue

        # (4) Desired → Absorb split (liquidity first, then buybacks)
        queue_total = float(queue_tokens_by_cat.sum())
        desired = queue_total * p
        desired_usd[day] = desired

        # First absorb with dynamic liquidity
        abs_liq = min(desired, Ldyn_liq)
        residual_after_liq = desired - abs_liq

        # Then absorb with buybacks
        abs_bb = min(residual_after_liq, Ldyn_bb)
        residual = residual_after_liq - abs_bb

        absorbed_liq_usd[day] = abs_liq
        absorbed_bb_usd[day]  = abs_bb

        if residual > depth_s:
            overflow_usd[day] = residual - depth_s
            overflow_days_left = int(OVERFLOW_WINDOW_DAYS)

        # Execute tokens actually sold (abs_liq + abs_bb)
        absorbed_total = abs_liq + abs_bb
        tokens_to_sell = absorbed_total / max(p, 1e-12)
        if queue_total > 0.0 and tokens_to_sell > 0.0:
            ratio = tokens_to_sell / queue_total
            exec_tokens = queue_tokens_by_cat * ratio
            realized_usd_by_cat[:, day] = exec_tokens * p
            queue_tokens_by_cat -= exec_tokens  # leftovers remain in queue

        # (5) μ/σ of the day: bear > overflow > base
        mu_eff, sigma_eff = MU_BASE, SIGMA_BASE
        in_bear = False
        for (bs, be, bmu, bmult) in BEAR_WINDOWS:
            if int(bs) <= day <= int(be):
                mu_eff    = float(bmu)
                sigma_eff = SIGMA_BASE * float(bmult)
                in_bear   = True
                break
        if not in_bear and overflow_days_left > 0:
            mu_eff    = MU_OVERFLOW
            sigma_eff = SIGMA_BASE * SIGMA_MULT_OVERFLOW
            overflow_days_left -= 1

        # (6) GBM update
        z = rng.normal()
        price_next = p * np.exp((mu_eff - 0.5 * sigma_eff**2) * DT + sigma_eff * np.sqrt(DT) * z)
        price[day] = max(price_next, 1e-12)

    return (price, realized_usd_by_cat, desired_usd, absorbed_liq_usd, absorbed_bb_usd,
            overflow_usd, static_depth_series, dyn_liq_series, buyback_series)

# ===================== Monte Carlo & aggregates =================================
all_prices = np.zeros((T, N))
sum_realized = np.zeros((C, T))
sum_desired  = np.zeros(T)
sum_abs_liq  = np.zeros(T)
sum_abs_bb   = np.zeros(T)
sum_overflow = np.zeros(T)
static_depth_ref = None
dyn_liq_ref      = None
buyback_ref      = None

for n in range(N):
    (p, realized, desired, a_liq, a_bb, overflow, depth_s, Ldyn_liq, Lbb) = run_one(42+n)
    all_prices[:, n] = p
    sum_realized += realized
    sum_desired  += desired
    sum_abs_liq  += a_liq
    sum_abs_bb   += a_bb
    sum_overflow += overflow
    if static_depth_ref is None:
        static_depth_ref = depth_s
        dyn_liq_ref      = Ldyn_liq
        buyback_ref      = Lbb

median_price = np.median(all_prices, axis=1)
avg_realized = sum_realized / N
avg_desired  = sum_desired  / N
avg_abs_liq  = sum_abs_liq  / N
avg_abs_bb   = sum_abs_bb   / N
avg_overflow = sum_overflow / N

# ===================== Graph 1: Realized by category + price ====================
st.write("### Graph 1 — Realized (absorbed) by Category vs Median Price")
st.caption("Realized = exécutions via **Dynamic Liquidity + Buybacks** (la décomposition est visible dans le Graph 2).")
use_log = st.checkbox("Log-scale for USD/day axis (Graph 1)", value=False, help="Active un axe Y logarithmique pour mieux lire les pics.")
clip_99 = st.checkbox("Clip left axis at 99th percentile (reduce spikes)", value=True, help="Coupe l’axe gauche au 99e percentile pour réduire l’effet des spikes.")
smooth_left = st.checkbox("Smooth stacked sum (display only)", value=True, help="Lisse la somme empilée (affichage uniquement).")

sum_realized_series = avg_realized.sum(axis=0)
sum_realized_series_disp = uniform_filter1d(sum_realized_series, size=7, mode="nearest") if smooth_left else sum_realized_series

fig1, ax1 = plt.subplots(figsize=(13, 6))
ax1.stackplot(t, avg_realized, labels=[str(x) for x in df["Category"]], alpha=0.65)
ax1.set_xlabel("Day")
ax1.set_ylabel("Realized Sells (USD/day)")
if use_log:
    ax1.set_yscale("log")
if clip_99 and not use_log:
    y_top = float(np.percentile(sum_realized_series_disp, 99) * 1.15)
    ax1.set_ylim(0, max(1.0, y_top))
ax1.legend(loc="upper left", ncol=2, fontsize=8)

ax2 = ax1.twinx()
ax2.plot(t, median_price, linewidth=2, label="Median Price (100 sims)")
ax2.set_ylabel("Token Price (USD)")
ax2.legend(loc="upper right")
ax1.set_title("Realized Sales by Category vs Median Price (Overflow windows on μ/σ)")
st.pyplot(fig1)

# ===================== Graph 2: Desired vs Absorption split vs Overflow =========
st.write("### Graph 2 — Desired vs Absorbed (split: Liquidity vs Buybacks) vs Overflow + Depth")
fig2, ax = plt.subplots(figsize=(14, 6))

# Desired (blue fill)
ax.fill_between(t, 0, avg_desired, color="#cfe3ff", alpha=0.8, label="Desired sells (USD/day)")

# Absorption split (stack the two)
ax.fill_between(t, 0, avg_abs_liq, color="#ffcf99", alpha=0.9, label="Absorbed by Dynamic Liquidity")
ax.fill_between(t, avg_abs_liq, avg_abs_liq + avg_abs_bb, color="#ffa466", alpha=0.9, label="Absorbed by Buybacks")

# Threshold line (static depth)
ax.plot(t, static_depth_ref, linestyle="--", color="#1b6ac9", linewidth=2.0,
        label="Static Depth Threshold (Flowdesk-calibrated)")

# Dynamic Liquidity & Buyback lines (capacity cues)
ax.plot(t, dyn_liq_ref, linestyle=":", color="#f39c12", linewidth=2, label="Dynamic Liquidity (absorber)")
ax.plot(t, buyback_ref, linestyle=":", color="#d35400", linewidth=1.6, label="Buybacks (absorber)")

# Overflow hatch (only the part above depth)
residual_avg = np.maximum(avg_desired - (avg_abs_liq + avg_abs_bb), 0.0)
overflow_zone = np.maximum(residual_avg - static_depth_ref, 0.0)
ax.fill_between(t, static_depth_ref, static_depth_ref + overflow_zone,
                where=(overflow_zone > 0),
                facecolor="none", edgecolor="red", hatch="//", linewidth=0.0,
                label="Overflow (triggers μ/σ)")

ax.set_xlabel("Day")
ax.set_ylabel("USD/day")
ax.legend(loc="upper left", ncol=2)
ax.set_title("Desired vs Absorbed vs Overflow — StaticDepth is a threshold (not an absorber)")
st.pyplot(fig2)

# ===================== Optional: Utilization ====================================
if st.checkbox("Show Utilization (Residual / StaticDepth)", help="Affiche l’utilisation de la profondeur statique (Residual / StaticDepth). Overflow quand > 1."):
    util = np.divide(residual_avg, np.maximum(static_depth_ref, 1e-9))
    figU, axU = plt.subplots(figsize=(13, 4))
    axU.plot(t, util, linewidth=2, label="Residual / StaticDepth")
    axU.axhline(1.0, linestyle="--", linewidth=1, label="Overflow threshold")
    util_top = max(1.1, min(5.0, float(np.nanmax(util)*1.1)))
    axU.fill_between(t, 1.0, np.minimum(util, util_top), where=(util>1.0),
                     alpha=0.25, step="pre", label="Overflow zone")
    axU.set_ylim(0, util_top)
    axU.set_xlabel("Day"); axU.set_ylabel("×")
    axU.legend(loc="upper left")
    axU.set_title("Depth Utilization — overflow when > 1")
    st.pyplot(figU)

# ===================== Help (tooltips-style recap) ==============================
with st.expander("🛈 What you’re seeing (quick glossary)"):
    st.markdown("""
- **Revenues (progressifs & continus)** : chaque année est une **rampe linéaire** dont le **départ = fin de l’année précédente**.
  La **somme des 365 jours** égale exactement le **total annuel** fourni.
- **Desired sells** = ce que les holders *veulent* vendre aujourd’hui (unlocks × règles SP par pool ; ROI trigger-safe).
- **Absorbed by Dynamic Liquidity** = exécutions financées par la liquidité dynamique (incl. dépôt initial).
- **Absorbed by Buybacks** = exécutions financées par le budget de rachat du jour (depuis revenus).
- **Static Depth (Flowdesk)** = *seuil uniquement* (n’absorbe pas). Le résiduel au-dessus déclenche une fenêtre μ/σ de stress.
""")

# ===================== Raw Data (optional) ======================================
if st.checkbox("Show Raw Data", help="Affiche un tableau avec les séries agrégées et les découpes par catégorie."):
    out = pd.DataFrame({
        "Day": t,
        "Median Price": np.median(all_prices, axis=1),
        "Desired USD/day": avg_desired,
        "Absorbed by Liquidity USD/day": avg_abs_liq,
        "Absorbed by Buybacks USD/day": avg_abs_bb,
        "Residual USD/day": residual_avg,
        "Overflow USD/day": avg_overflow,
        "Static Depth (USD/day)": static_depth_ref,
        "Dyn Liquidity capacity (USD/day)": dyn_liq_ref,
        "Buyback capacity (USD/day)": buyback_ref,
        "Revenue per day (USD)": revenue_daily_usd[:T],
        "Circulating (tokens)": circ,
    })
    for i, name in enumerate(df["Category"]):
        out[f"Realized USD — {name}"] = avg_realized[i]
    st.dataframe(out, use_container_width=True)
