"""
╔══════════════════════════════════════════════════════════════════╗
║         VaR & ES VERIFICATION SCRIPT  —  KaramFRM               ║
║  Fixed date range so results are fully reproducible              ║
╚══════════════════════════════════════════════════════════════════╝

Run:  python verify_var_es.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# ── Fixed inputs ───────────────────────────────────────────────────
TICKERS     = ["SPY", "BND", "GLD", "QQQ"]
WEIGHTS     = [0.25, 0.25, 0.25, 0.25]       # equal weight
PORT_VALUE  = 100_000
CONFIDENCE  = 0.95
DAYS        = 5
START       = "2020-01-01"                   # fixed — reproducible
END         = "2025-01-01"                   # fixed — reproducible
SIMS        = 10_000

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
SEP  = "─" * 60

def header(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    if detail:
        print(f"         → {detail}")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — DATA FETCH
# ══════════════════════════════════════════════════════════════════
header("STEP 1 — DATA FETCH")
print(f"  Tickers    : {TICKERS}")
print(f"  Date range : {START}  →  {END}  (FIXED)")
print(f"  Weights    : {[f'{w*100:.0f}%' for w in WEIGHTS]}")
print(f"  Sum weights: {sum(WEIGHTS)*100:.0f}%")

prices = pd.DataFrame()
for t in TICKERS:
    raw = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)
    prices[t] = raw["Adj Close"]

log_returns = np.log(prices / prices.shift(1)).dropna()
weights     = np.array(WEIGHTS)

print(f"\n  Trading days loaded : {len(log_returns)}")
print(f"  Expected (~5Y)      : ~1,260")
check("Sufficient data loaded", len(log_returns) > 1000,
      f"{len(log_returns)} days")
check("Weights sum to 1.0", abs(sum(weights) - 1.0) < 1e-9,
      f"Sum = {sum(weights):.10f}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — PORTFOLIO STATISTICS
# ══════════════════════════════════════════════════════════════════
header("STEP 2 — PORTFOLIO STATISTICS")

portfolio_returns  = (log_returns * weights).sum(axis=1)
cov_matrix_annual  = log_returns.cov() * 252
portfolio_std_dev  = float(np.sqrt(weights.T @ cov_matrix_annual @ weights))
portfolio_mean     = float(portfolio_returns.mean())
portfolio_mean_ann = portfolio_mean * 252

print(f"\n  Daily mean return    : {portfolio_mean*100:.4f}%")
print(f"  Annual mean return   : {portfolio_mean_ann*100:.2f}%")
print(f"  Annual volatility    : {portfolio_std_dev*100:.2f}%")
print(f"  Daily vol (approx)   : {portfolio_std_dev/np.sqrt(252)*100:.4f}%")

check("Annual vol is positive",           portfolio_std_dev > 0)
check("Annual vol is realistic (1–40%)",  0.01 < portfolio_std_dev < 0.40,
      f"{portfolio_std_dev*100:.2f}%")
check("Annual return is realistic",       -0.50 < portfolio_mean_ann < 0.50,
      f"{portfolio_mean_ann*100:.2f}%")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — ROLLING RETURNS
# ══════════════════════════════════════════════════════════════════
header("STEP 3 — ROLLING RETURNS")

range_returns        = portfolio_returns.rolling(window=DAYS).sum().dropna()
range_returns_dollar = range_returns * PORT_VALUE

print(f"\n  Rolling window       : {DAYS} days")
print(f"  Observations         : {len(range_returns)}")
print(f"  Min 5D return        : {range_returns.min()*100:.2f}%  (${range_returns.min()*PORT_VALUE:,.0f})")
print(f"  Max 5D return        : {range_returns.max()*100:.2f}%  (${range_returns.max()*PORT_VALUE:,.0f})")
print(f"  Mean 5D return       : {range_returns.mean()*100:.4f}%")

check("Rolling returns are computed",    len(range_returns) > 0)
check("Min return is negative (losses exist)", range_returns.min() < 0,
      f"Min = {range_returns.min()*100:.2f}%")
check("Observations ≥ 1000",             len(range_returns) >= 1000,
      f"{len(range_returns)}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — HISTORICAL SIMULATION
# ══════════════════════════════════════════════════════════════════
header("STEP 4 — HISTORICAL SIMULATION")

percentile_cutoff = 100 * (1 - CONFIDENCE)
raw_percentile    = np.percentile(range_returns, percentile_cutoff)
hist_VaR          = -raw_percentile * PORT_VALUE

losses            = -range_returns * PORT_VALUE
tail_losses       = losses[losses >= hist_VaR]
hist_ES           = float(tail_losses.mean())

breach_count      = int((range_returns < raw_percentile).sum())
expected_breaches = int(len(range_returns) * (1 - CONFIDENCE))

print(f"\n  Confidence level     : {CONFIDENCE*100:.0f}%")
print(f"  Percentile cutoff    : {percentile_cutoff:.1f}th")
print(f"  Raw percentile value : {raw_percentile*100:.4f}%")
print(f"  5-Day VaR            : ${hist_VaR:,.2f}")
print(f"  5-Day ES             : ${hist_ES:,.2f}")
print(f"  ES/VaR ratio         : {hist_ES/hist_VaR:.3f}")
print(f"  Actual breaches      : {breach_count}")
print(f"  Expected breaches    : ~{expected_breaches}  ({(1-CONFIDENCE)*100:.0f}% of {len(range_returns)})")

check("VaR is positive",                  hist_VaR > 0,        f"${hist_VaR:,.2f}")
check("ES is positive",                   hist_ES > 0,         f"${hist_ES:,.2f}")
check("ES > VaR  (always must hold)",     hist_ES > hist_VaR,
      f"ES ${hist_ES:,.2f} > VaR ${hist_VaR:,.2f}")
check("ES/VaR ratio is reasonable (1.1–2.5)", 1.1 < hist_ES/hist_VaR < 2.5,
      f"Ratio = {hist_ES/hist_VaR:.3f}")
check("Breach count ≈ expected",
      abs(breach_count - expected_breaches) <= expected_breaches * 0.3,
      f"Actual {breach_count} vs expected ~{expected_breaches}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — PARAMETRIC (NORMAL)
# ══════════════════════════════════════════════════════════════════
header("STEP 5 — PARAMETRIC (NORMAL DISTRIBUTION)")

z_score   = norm.ppf(1 - CONFIDENCE)
sigma_t   = portfolio_std_dev * np.sqrt(DAYS / 252)   # period vol
mu_t      = portfolio_mean * DAYS                      # period mean

param_VaR = -PORT_VALUE * (z_score * sigma_t - mu_t)
param_ES  = (PORT_VALUE * (norm.pdf(z_score) / (1 - CONFIDENCE)) * sigma_t
             - mu_t * PORT_VALUE)

print(f"\n  z-score ({CONFIDENCE*100:.0f}%)        : {z_score:.4f}")
print(f"  sigma_t (period vol)   : {sigma_t*100:.4f}%")
print(f"  mu_t    (period mean)  : {mu_t*100:.6f}%")
print(f"  5-Day Parametric VaR   : ${param_VaR:,.2f}")
print(f"  5-Day Parametric ES    : ${param_ES:,.2f}")
print(f"  ES/VaR ratio           : {param_ES/param_VaR:.3f}")

# Formula verification — manual expansion
manual_VaR = PORT_VALUE * (-z_score * sigma_t + mu_t)
manual_ES  = PORT_VALUE * (norm.pdf(z_score) / (1 - CONFIDENCE) * sigma_t - mu_t)

print(f"\n  Manual formula check:")
print(f"    VaR = V × (−z × σ_t + μ_t)")
print(f"        = {PORT_VALUE:,} × (−({z_score:.4f}) × {sigma_t:.6f} + {mu_t:.6f})")
print(f"        = ${manual_VaR:,.2f}")
print(f"    ES  = V × (φ(z)/(1−α) × σ_t − μ_t)")
print(f"        = ${manual_ES:,.2f}")

check("z-score is negative for loss-side",  z_score < 0,     f"z = {z_score:.4f}")
check("VaR is positive",                    param_VaR > 0,   f"${param_VaR:,.2f}")
check("ES is positive",                     param_ES > 0,    f"${param_ES:,.2f}")
check("ES > VaR  (always must hold)",       param_ES > param_VaR,
      f"ES ${param_ES:,.2f} > VaR ${param_VaR:,.2f}")
check("Formula matches manual expansion",   abs(param_VaR - manual_VaR) < 0.01,
      f"Diff = ${abs(param_VaR - manual_VaR):.4f}")
check("ES/VaR ratio reasonable (1.1–2.0)", 1.1 < param_ES/param_VaR < 2.0,
      f"Ratio = {param_ES/param_VaR:.3f}")

# Multi-confidence monotonicity check
print(f"\n  Monotonicity check (VaR₉₀ < VaR₉₅ < VaR₉₉):")
conf_levels = [0.90, 0.95, 0.99]
var_levels  = []
for cl in conf_levels:
    z  = norm.ppf(1 - cl)
    v  = -PORT_VALUE * (z * sigma_t - mu_t)
    var_levels.append(v)
    print(f"    VaR {cl*100:.0f}% = ${v:,.2f}")

check("VaR₉₀ < VaR₉₅ < VaR₉₉",
      var_levels[0] < var_levels[1] < var_levels[2],
      f"${var_levels[0]:,.0f} < ${var_levels[1]:,.0f} < ${var_levels[2]:,.0f}")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════
header("STEP 6 — MONTE CARLO SIMULATION")

np.random.seed(42)
z_scores     = np.random.normal(0, 1, SIMS)
scenario_pnl = PORT_VALUE * (
    portfolio_mean * DAYS
    + portfolio_std_dev * np.sqrt(DAYS / 252) * z_scores
)

mc_VaR   = -np.percentile(scenario_pnl, (1 - CONFIDENCE) * 100)
mc_tail  = scenario_pnl[scenario_pnl <= -mc_VaR]
mc_ES    = float(-mc_tail.mean())

print(f"\n  Simulations          : {SIMS:,}")
print(f"  Seed                 : 42  (fixed → reproducible)")
print(f"  Sim mean P&L         : ${scenario_pnl.mean():,.2f}")
print(f"  Sim std P&L          : ${scenario_pnl.std():,.2f}")
print(f"  5-Day MC VaR         : ${mc_VaR:,.2f}")
print(f"  5-Day MC ES          : ${mc_ES:,.2f}")
print(f"  ES/VaR ratio         : {mc_ES/mc_VaR:.3f}")
print(f"  Tail observations    : {len(mc_tail)}  (expected ~{int(SIMS*(1-CONFIDENCE))})")

# Verify MC sim std matches theoretical
expected_std = PORT_VALUE * portfolio_std_dev * np.sqrt(DAYS / 252)
actual_std   = scenario_pnl.std()

print(f"\n  Theoretical P&L std  : ${expected_std:,.2f}")
print(f"  Simulated P&L std    : ${actual_std:,.2f}")
print(f"  Difference           : {abs(actual_std - expected_std)/expected_std*100:.2f}%")

check("VaR is positive",                 mc_VaR > 0,       f"${mc_VaR:,.2f}")
check("ES is positive",                  mc_ES > 0,        f"${mc_ES:,.2f}")
check("ES > VaR  (always must hold)",    mc_ES > mc_VaR,
      f"ES ${mc_ES:,.2f} > VaR ${mc_VaR:,.2f}")
check("Simulated std ≈ theoretical (within 5%)",
      abs(actual_std - expected_std) / expected_std < 0.05,
      f"{abs(actual_std-expected_std)/expected_std*100:.2f}% diff")
check("Tail count ≈ expected",
      abs(len(mc_tail) - int(SIMS*(1-CONFIDENCE))) <= 50,
      f"{len(mc_tail)} vs expected {int(SIMS*(1-CONFIDENCE))}")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — CROSS-METHOD COMPARISON
# ══════════════════════════════════════════════════════════════════
header("STEP 7 — CROSS-METHOD COMPARISON")

print(f"""
  {'Method':<25} {'VaR':>12} {'ES':>12} {'ES/VaR':>10}
  {SEP}
  {'Historical Simulation':<25} ${hist_VaR:>10,.2f} ${hist_ES:>10,.2f} {hist_ES/hist_VaR:>9.3f}
  {'Parametric (Normal)':<25} ${param_VaR:>10,.2f} ${param_ES:>10,.2f} {param_ES/param_VaR:>9.3f}
  {'Monte Carlo (10k)':<25} ${mc_VaR:>10,.2f} ${mc_ES:>10,.2f} {mc_ES/mc_VaR:>9.3f}
""")

# MC and Parametric should be close (both assume normality effectively)
mc_vs_param_pct = abs(mc_VaR - param_VaR) / param_VaR * 100

print(f"  MC vs Parametric VaR divergence  : {mc_vs_param_pct:.2f}%")
print(f"  (Expected < 5% with 10,000 sims)")

check("All methods: ES > VaR",
      hist_ES > hist_VaR and param_ES > param_VaR and mc_ES > mc_VaR)
check("MC ≈ Parametric VaR (within 5%)",  mc_vs_param_pct < 5.0,
      f"{mc_vs_param_pct:.2f}% divergence")
check("Historical VaR in reasonable range of Parametric",
      0.5 < hist_VaR / param_VaR < 2.0,
      f"Ratio = {hist_VaR/param_VaR:.3f}")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
header("FINAL SUMMARY")

print(f"""
  Input Parameters
  ─────────────────────────────────────────────────
  Tickers      : {', '.join(TICKERS)}
  Weights      : {[f'{w*100:.0f}%' for w in WEIGHTS]}
  Portfolio    : ${PORT_VALUE:,}
  Date range   : {START} → {END}
  Time horizon : {DAYS} days
  Confidence   : {CONFIDENCE*100:.0f}%

  Results
  ─────────────────────────────────────────────────
  Historical   VaR = ${hist_VaR:>10,.2f}   ES = ${hist_ES:>10,.2f}
  Parametric   VaR = ${param_VaR:>10,.2f}   ES = ${param_ES:>10,.2f}
  Monte Carlo  VaR = ${mc_VaR:>10,.2f}   ES = ${mc_ES:>10,.2f}

  Portfolio Statistics
  ─────────────────────────────────────────────────
  Annual Vol   : {portfolio_std_dev*100:.2f}%
  Annual Return: {portfolio_mean_ann*100:.2f}%
""")

print("  All checks complete. Review ✅ / ❌ above for any issues.\n")
