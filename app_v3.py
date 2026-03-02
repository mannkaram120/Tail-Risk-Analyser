import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from scipy.stats import norm, gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tail Risk Analyzer | KaramFRM",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#F5F2EB; color:#1a1a1a; }
.stApp { background:#F5F2EB; }

/* ══ SIDEBAR ══════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background:#EDEAE0;
    border-right:1px solid #D0CCC4;
    padding-top: 0.5rem;
}

/* All sidebar text — dark, readable */
[data-testid="stSidebar"] * {
    font-family:'Space Mono',monospace !important;
    color:#1a1a1a !important;
}

/* Section headers — bold, uppercase, 15px equivalent */
.sidebar-section {
    font-family:'Space Mono',monospace;
    font-size:0.82rem;
    font-weight:700;
    letter-spacing:0.18em;
    color:#1a1a1a;
    text-transform:uppercase;
    margin-top:24px;
    margin-bottom:8px;
    padding-bottom:4px;
    border-bottom:1px solid #C8C4BC;
}

/* Sub-labels (e.g. "Stock Tickers") */
.sidebar-sublabel {
    font-family:'Space Mono',monospace;
    font-size:0.72rem;
    font-weight:700;
    letter-spacing:0.1em;
    color:#3a3a3a;
    text-transform:uppercase;
    margin-bottom:6px;
    margin-top:14px;
}

/* Streamlit native widget labels */
[data-testid="stSidebar"] label {
    font-size:0.72rem !important;
    font-weight:700 !important;
    letter-spacing:0.1em !important;
    color:#1a1a1a !important;
    text-transform:uppercase !important;
}

/* ── Ticker tag ── */
.ticker-tag {
    display:inline-flex; align-items:center;
    background:#FFFFFF;
    border:1.5px solid #7A8C5E;
    color:#1a1a1a;
    font-family:'Space Mono',monospace;
    font-size:0.72rem;
    padding:4px 10px;
    border-radius:4px;
    font-weight:700;
    margin-top:4px;
}

/* ── Text input — white background, dark text ── */
[data-testid="stSidebar"] input[type="text"] {
    background-color:#FFFFFF !important;
    color:#1a1a1a !important;
    border:1.5px solid #DCDCDC !important;
    border-radius:6px !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.76rem !important;
    padding:8px 12px !important;
}
[data-testid="stSidebar"] input[type="text"]::placeholder {
    color:#888 !important;
    font-style:italic !important;
}

/* ── Number input ── */
[data-testid="stSidebar"] input[type="number"] {
    background:#FFFFFF !important;
    border:1.5px solid #DCDCDC !important;
    border-radius:6px !important;
    color:#1a1a1a !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.76rem !important;
}

/* ── Ticker row spacing ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap:0px !important;
    margin-bottom:-8px !important;
    align-items:center !important;
    min-height:34px !important;
}

/* ── × remove button ── */
[data-testid="stSidebar"] button[kind="secondary"] {
    padding:0px 7px !important;
    min-height:26px !important;
    height:26px !important;
    font-size:0.9rem !important;
    line-height:1 !important;
    border:1px solid #DCDCDC !important;
    border-radius:4px !important;
    background:#FFFFFF !important;
    color:#555 !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    border-color:#C0392B !important;
    color:#C0392B !important;
}

/* ── Quick-add & main buttons ── */
.stButton > button {
    font-family:'Space Mono',monospace !important;
    font-size:0.70rem !important;
    background:#FFFFFF !important;
    border:1px solid #DCDCDC !important;
    color:#1a1a1a !important;
    border-radius:4px !important;
    padding:5px 10px !important;
    transition:all 0.15s ease !important;
}
.stButton > button:hover {
    border-color:#7A8C5E !important;
    color:#7A8C5E !important;
    background:#F5F2EB !important;
}

/* ── Sliders ── */
div[data-testid="stSlider"] * {
    color:#1a1a1a !important;
    font-family:'Space Mono',monospace !important;
}
div[data-testid="stSlider"] p {
    font-size:0.72rem !important;
    font-weight:700 !important;
    letter-spacing:0.08em !important;
    color:#1a1a1a !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] label {
    color:#1a1a1a !important;
    font-size:0.76rem !important;
    font-weight:500 !important;
}

/* ══ MAIN SECTION LABEL ═══════════════════════════════════════════════════════ */
.section-label {
    font-family:'Space Mono',monospace;
    font-size:0.62rem;
    font-weight:700;
    letter-spacing:0.15em;
    color:#7A8C5E;
    text-transform:uppercase;
    margin-bottom:0.3rem;
    margin-top:0.7rem;
}

/* ══ METRIC CARDS ════════════════════════════════════════════════════════════ */
.metric-card {
    background:#FFFFFF;
    border:1px solid #E0DDD5;
    border-radius:6px;
    padding:18px 20px;
    font-family:'Space Mono',monospace;
}
.metric-label { font-size:0.58rem; letter-spacing:0.12em; color:#555; text-transform:uppercase; margin-bottom:6px; }
.metric-value { font-size:1.55rem; font-weight:700; color:#1a1a1a; line-height:1; }
.metric-sub   { font-size:0.62rem; color:#888; margin-top:5px; }
.metric-var   { border-top:3px solid #C0392B; }
.metric-es    { border-top:3px solid #7B241C; }
.metric-vol   { border-top:3px solid #333; }
.metric-loss  { border-top:3px solid #888; }

/* ══ COMPARISON TABLE ════════════════════════════════════════════════════════ */
.compare-table { width:100%; font-family:'Space Mono',monospace; font-size:0.78rem; border-collapse:collapse; }
.compare-table th { background:#1a1a1a; color:#F5F2EB; padding:12px 16px; text-align:left; font-size:0.62rem; letter-spacing:0.1em; text-transform:uppercase; }
.compare-table td { padding:12px 16px; border-bottom:1px solid #E8E4DC; color:#1a1a1a; }
.compare-table tr:nth-child(even) td { background:#F9F7F2; }
.compare-table tr:hover td { background:#F0EDE5; }

/* ══ INFO BOX ════════════════════════════════════════════════════════════════ */
.info-box { background:#FAFAF7; border-left:3px solid #7A8C5E; padding:12px 16px; font-size:0.78rem; color:#333; border-radius:0 4px 4px 0; margin:12px 0; font-family:'Inter',sans-serif; line-height:1.6; }

/* ══ MISC ════════════════════════════════════════════════════════════════════ */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:2rem; padding-bottom:2rem; }
hr { border:none; border-top:1px solid #D4CFC4; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "tickers" not in st.session_state:
    st.session_state.tickers = ["SPY", "BND", "GLD", "QQQ"]

QUICK_ADD = ["SPY", "QQQ", "BND", "GLD", "AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "META", "BRK-B"]

# ── Colour palette (industry standard — minimal) ───────────────────────────────
C_HIST    = "#BFBFBF"       # neutral gray — full distribution
C_TAIL    = "rgba(192,57,43,0.30)"  # soft red transparent — tail shading
C_VAR     = "#C0392B"       # red — VaR line
C_ES      = "#7B241C"       # dark red — ES line
C_KDE     = "#555555"       # dark gray — KDE overlay
C_NORM    = "#1a1a1a"       # black — normal curve (parametric)
C_GRID    = "#EEEBE4"
C_BG      = "#FFFFFF"
C_PLOTBG  = "#FAFAF8"

# ── Base Plotly layout ─────────────────────────────────────────────────────────
def base_layout(title_text, x_title, height=500):
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8F9FA",
        font=dict(family="Space Mono, monospace", size=12, color="#1a1a1a"),
        margin=dict(l=80, r=40, t=80, b=75),
        height=height,
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(size=19, color="#1a1a1a", family="Space Mono, monospace"),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            title=dict(text=f"<b>{x_title}</b>", font=dict(size=14, color="#1a1a1a")),
            showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
            zeroline=False,
            showline=True, linecolor="#CCCCCC", linewidth=1,
            tickfont=dict(size=12, color="#333333"),
            tickprefix="$", tickformat=",.0f",
            showspikes=False,
            mirror=False,
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            title=dict(text="<b>Probability Density</b>", font=dict(size=14, color="#1a1a1a")),
            showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
            zeroline=False,
            showline=True, linecolor="#CCCCCC", linewidth=1,
            tickfont=dict(size=12, color="#333333"),
            mirror=False,
            ticks="outside",
            ticklen=4,
        ),
        legend=dict(
            orientation="h",
            font=dict(size=12, color="#1a1a1a"),
            bgcolor="rgba(255,255,255,0.0)",
            borderwidth=0,
            x=1.0, y=1.06,
            xanchor="right", yanchor="bottom",
        ),
        showlegend=True,
        barmode="overlay",
        shapes=[],
        annotations=[],
    )

# ── Unified chart builder ──────────────────────────────────────────────────────
def build_risk_chart(
    title, x_data, var_val, es_val,
    x_label, days, confidence_pct,
    kde_overlay=True, normal_curve=False,
    portfolio_mean=None, portfolio_std_dev=None, portfolio_value=None,
):
    """
    Professional risk chart:
    - Neutral gray histogram
    - Semi-transparent red tail shading
    - Red VaR line (thick)
    - Dark red ES line (medium)
    - KDE or normal curve overlay
    - Single summary annotation box (top-right)
    """
    fig = go.Figure()

    x_arr = np.array(x_data)
    x_min, x_max = x_arr.min(), x_arr.max()

    # 1. Full distribution histogram — neutral gray
    fig.add_trace(go.Histogram(
        x=x_arr,
        nbinsx=60,
        histnorm="probability density",
        marker=dict(color=C_HIST, line=dict(color="#999", width=0.4)),
        opacity=0.75,
        name="P&L Distribution",
        hovertemplate="P&L: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
    ))

    # 2. Tail region — soft red overlay histogram
    tail = x_arr[x_arr <= -var_val]
    if len(tail) > 1:
        fig.add_trace(go.Histogram(
            x=tail,
            nbinsx=25,
            histnorm="probability density",
            marker=dict(color=C_TAIL, line=dict(color="rgba(192,57,43,0.5)", width=0.4)),
            opacity=1.0,
            name="Tail Region",
            hovertemplate="Loss: $%{x:,.0f}<br>Density: %{y:.6f}<extra></extra>",
        ))

    # 3. KDE overlay — smooth density curve
    if kde_overlay and len(x_arr) > 10:
        kde = gaussian_kde(x_arr, bw_method=0.3)
        x_range = np.linspace(x_min * 1.1, x_max * 1.1, 500)
        kde_vals = kde(x_range)
        fig.add_trace(go.Scatter(
            x=x_range, y=kde_vals,
            mode="lines",
            line=dict(color=C_KDE, width=2),
            name="KDE",
            hoverinfo="skip",
        ))

    # 4. Normal curve overlay for parametric
    if normal_curve and portfolio_mean is not None:
        mu  = portfolio_mean * days * portfolio_value
        sig = portfolio_std_dev * np.sqrt(days / 252) * portfolio_value
        x_range = np.linspace(x_min * 1.1, x_max * 1.1, 500)
        norm_vals = norm.pdf(x_range, mu, sig)
        fig.add_trace(go.Scatter(
            x=x_range, y=norm_vals,
            mode="lines",
            line=dict(color=C_NORM, width=2, dash="dot"),
            name="Normal Curve",
            hoverinfo="skip",
        ))

    # 5. VaR line — red, thick
    fig.add_vline(
        x=-var_val,
        line=dict(color=C_VAR, width=2.5, dash="dash"),
    )

    # 6. ES line — dark red, slightly thinner
    fig.add_vline(
        x=-es_val,
        line=dict(color=C_ES, width=2, dash="solid"),
    )

    # 7. Single summary box — top right, clean institutional style
    mean_pct = (portfolio_mean * 252 * 100) if portfolio_mean is not None else 0
    vol_pct  = (portfolio_std_dev * 100) if portfolio_std_dev is not None else 0

    summary = (
        f"<b>{days}-Day VaR ({confidence_pct}%):   ${var_val:>10,.0f}</b><br>"
        f"<b>{days}-Day ES  ({confidence_pct}%):   ${es_val:>10,.0f}</b>"
    )
    if portfolio_mean is not None:
        summary += (
            f"<br>─────────────────────────────<br>"
            f"Ann. Return:    {mean_pct:+.2f}%<br>"
            f"Ann. Vol:       {vol_pct:.2f}%"
        )

    fig.add_annotation(
        x=0.99, y=0.98,
        xref="paper", yref="paper",
        text=summary,
        showarrow=False,
        align="left",
        bgcolor="#FFFFFF",
        bordercolor="#CCCCCC",
        borderwidth=1,
        borderpad=12,
        font=dict(family="Space Mono, monospace", size=12, color="#1a1a1a"),
        xanchor="right", yanchor="top",
        opacity=0.97,
    )

    layout = base_layout(title, x_label)
    fig.update_layout(**layout)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### KARAM.")
    st.markdown(
        '<p style="font-family:Space Mono,monospace;font-size:0.78rem;color:#1a1a1a;font-weight:700;margin:0 0 8px 0;letter-spacing:0.12em;">TAIL RISK ANALYZER</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<p class="sidebar-section">// Asset Selection</p>', unsafe_allow_html=True)

    # ── Table header ──
    st.markdown("""
    <div style="display:grid;grid-template-columns:2fr 1.4fr 0.6fr;gap:4px;
                padding:6px 4px;border-bottom:2px solid #1a1a1a;margin-bottom:4px;">
        <span style="font-family:Space Mono,monospace;font-size:0.65rem;font-weight:700;
                     letter-spacing:0.12em;color:#1a1a1a;text-transform:uppercase;">Ticker</span>
        <span style="font-family:Space Mono,monospace;font-size:0.65rem;font-weight:700;
                     letter-spacing:0.12em;color:#1a1a1a;text-transform:uppercase;">Weight (%)</span>
        <span style="font-family:Space Mono,monospace;font-size:0.65rem;font-weight:700;
                     letter-spacing:0.12em;color:#1a1a1a;text-transform:uppercase;">Remove</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Table rows: Ticker | Weight input | × ──
    n_tickers = len(st.session_state.tickers)
    default_w = round(100.0 / n_tickers, 1) if n_tickers > 0 else 100.0

    raw_weights = {}
    for t in list(st.session_state.tickers):
        col_name, col_w, col_x = st.columns([2, 1.4, 0.6])
        with col_name:
            st.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:0.76rem;font-weight:700;'
                f'color:#1a1a1a;padding:6px 0 0 2px;">{t}</div>',
                unsafe_allow_html=True
            )
        with col_w:
            raw_weights[t] = st.number_input(
                f"w_{t}", min_value=0.0, max_value=100.0,
                value=default_w, step=0.5, format="%.1f",
                key=f"weight_{t}",
                label_visibility="collapsed",
            )
        with col_x:
            if st.button("✕", key=f"remove_{t}"):
                st.session_state.tickers.remove(t)
                st.rerun()

    # ── Total weight indicator ──
    total_w = sum(raw_weights.values())
    color   = "#7A8C5E" if abs(total_w - 100.0) <= 0.11 else "#C0392B"
    symbol  = "✓" if abs(total_w - 100.0) <= 0.11 else "⚠"
    st.markdown(
        f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;font-weight:700;'
        f'color:{color};text-align:right;padding:6px 4px 2px 0;border-top:1px solid #C8C4BC;margin-top:4px;">'
        f'{symbol} Total: {total_w:.1f}%</div>',
        unsafe_allow_html=True
    )

    # ── Add ticker input ──
    st.markdown('<p class="sidebar-sublabel" style="margin-top:14px;">ADD TICKER</p>', unsafe_allow_html=True)
    new_ticker = st.text_input(
        "Add ticker", value="", placeholder="e.g. TSLA  →  press Enter",
        key="ticker_input", label_visibility="collapsed"
    )
    if new_ticker:
        t = new_ticker.strip().upper()
        if t and t not in st.session_state.tickers:
            st.session_state.tickers.append(t)
        st.rerun()

    st.markdown('<p class="sidebar-sublabel" style="margin-top:14px;">QUICK ADD</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, qt in enumerate(QUICK_ADD):
        if cols[i % 3].button(qt, key=f"qa_{qt}"):
            if qt not in st.session_state.tickers:
                st.session_state.tickers.append(qt)
            st.rerun()

    st.markdown("---")
    st.markdown('<p class="sidebar-section">// Parameters</p>', unsafe_allow_html=True)

    portfolio_value = st.number_input(
        "PORTFOLIO VALUE ($)", min_value=1000, max_value=10_000_000,
        value=100_000, step=10_000,
    )
    lookback_years = st.select_slider(
        "LOOKBACK PERIOD", options=[1, 2, 3, 5, 7, 10], value=5,
        format_func=lambda x: f"{x}Y"
    )
    days = st.select_slider(
        "TIME HORIZON", options=list(range(1, 31)), value=5,
        format_func=lambda x: f"{x}D"
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-section">// Risk Settings</p>', unsafe_allow_html=True)

    confidence_pct = st.select_slider(
        "CONFIDENCE LEVEL",
        options=[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.9],
        value=95,
        format_func=lambda x: f"{x}%"
    )
    confidence = confidence_pct / 100

    method = st.radio(
        "METHOD",
        options=["Historical", "Parametric", "Monte Carlo", "All (Compare)"],
        index=3,
    )

    st.markdown("---")
    run = st.button("▶  CALCULATE RISK", key="run_btn", use_container_width=True)

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<p class="section-label">// Quantitative Risk Tool</p>
<h1 style="font-family:'Space Mono',monospace;font-size:3rem;line-height:1.05;margin:0;font-weight:700;letter-spacing:-0.03em;color:#1a1a1a;">TAIL RISK</h1>
<h1 style="font-family:'Space Mono',monospace;font-size:3rem;line-height:1.05;margin:0;font-weight:700;letter-spacing:-0.03em;color:#B5B0A4;">ANALYZER</h1>
<p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#555;margin-top:0.8rem;letter-spacing:0.05em;">Real market data via yFinance &nbsp;·&nbsp; Historical &nbsp;·&nbsp; Parametric &nbsp;·&nbsp; Monte Carlo</p>
""", unsafe_allow_html=True)
st.markdown("---")

if not run:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;height:260px;">
        <p style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#B5B0A4;letter-spacing:0.12em;">
            CONFIGURE ASSETS AND CLICK CALCULATE RISK
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data fetch ─────────────────────────────────────────────────────────────────
tickers = st.session_state.tickers
if not tickers:
    st.warning("Please add at least one ticker.")
    st.stop()

# Resolve weights — normalise if not exactly 100 due to rounding, block if invalid
raw_w   = {t: st.session_state.get(f"weight_{t}", 100.0 / len(tickers)) for t in tickers}
total_w = sum(raw_w.values())

if total_w <= 0:
    st.error("Portfolio weights must sum to 100%. Please adjust the weights in the sidebar.")
    st.stop()

# Auto-normalise to exactly 1.0 (handles minor floating point drift)
weights_dict = {t: raw_w[t] / total_w for t in tickers}

if abs(total_w - 100.0) > 0.11:
    st.error(f"⚠ Portfolio weights sum to **{total_w:.1f}%** — they must equal 100%. Please fix in the sidebar before calculating.")
    st.stop()

enddate   = dt.datetime.now()
startdate = enddate - dt.timedelta(days=365 * lookback_years)

with st.spinner("Fetching market data..."):
    try:
        adj_close_df = pd.DataFrame()
        for ticker in tickers:
            data = yf.download(ticker, start=startdate, end=enddate,
                               auto_adjust=False, progress=False)
            if data.empty:
                st.error(f"❌ Could not fetch data for **{ticker}**. Please check the ticker symbol.")
                st.stop()
            adj_close_df[ticker] = data['Adj Close']

        log_returns       = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
        weights           = np.array([weights_dict[t] for t in tickers])
        portfolio_returns  = (log_returns * weights).sum(axis=1)
        range_returns      = portfolio_returns.rolling(window=days).sum().dropna()
        cov_matrix_annual  = log_returns.cov() * 252
        portfolio_std_dev  = np.sqrt(weights.T @ cov_matrix_annual @ weights)
        portfolio_mean     = portfolio_returns.mean()

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# ── Calculations ───────────────────────────────────────────────────────────────
def calc_historical(range_returns, portfolio_value, confidence):
    VaR    = -np.percentile(range_returns, 100 * (1 - confidence)) * portfolio_value
    losses = -range_returns * portfolio_value
    ES     = losses[losses >= VaR].mean()
    return float(VaR), float(ES)

def calc_parametric(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days):
    z   = norm.ppf(1 - confidence)
    VaR = -portfolio_value * (z * portfolio_std_dev * np.sqrt(days / 252) - portfolio_mean * days)
    ES  = (portfolio_value * (norm.pdf(z) / (1 - confidence)) * portfolio_std_dev * np.sqrt(days / 252)
           - portfolio_mean * days * portfolio_value)
    return float(VaR), float(ES)

def calc_montecarlo(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days, sims=10000):
    np.random.seed(42)
    z_scores     = np.random.normal(0, 1, sims)
    scenario_pnl = portfolio_value * (
        portfolio_mean * days
        + portfolio_std_dev * np.sqrt(days / 252) * z_scores
    )
    VaR  = -np.percentile(scenario_pnl, 100 * (1 - confidence))
    tail = scenario_pnl[scenario_pnl <= -VaR]
    ES   = -tail.mean()
    return float(VaR), float(ES), scenario_pnl

hist_VaR,  hist_ES             = calc_historical(range_returns, portfolio_value, confidence)
param_VaR, param_ES            = calc_parametric(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days)
mc_VaR,    mc_ES, mc_scenarios = calc_montecarlo(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days)

worst_loss = float(-range_returns.min() * portfolio_value)
annual_vol = portfolio_std_dev * 100
mean_return_ann = portfolio_mean * 252 * 100

display_VaR = {"Historical": hist_VaR, "Parametric": param_VaR,
               "Monte Carlo": mc_VaR,  "All (Compare)": hist_VaR}[method]
display_ES  = {"Historical": hist_ES,  "Parametric": param_ES,
               "Monte Carlo": mc_ES,   "All (Compare)": hist_ES}[method]

# ── Metric cards ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card metric-var">
        <div class="metric-label">Value at Risk ({confidence_pct}%)</div>
        <div class="metric-value">${display_VaR:,.0f}</div>
        <div class="metric-sub">{days}-day horizon</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card metric-es">
        <div class="metric-label">Expected Shortfall ({confidence_pct}%)</div>
        <div class="metric-value">${display_ES:,.0f}</div>
        <div class="metric-sub">avg loss beyond VaR</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card metric-vol">
        <div class="metric-label">Annual Volatility</div>
        <div class="metric-value">{annual_vol:.2f}%</div>
        <div class="metric-sub">portfolio σ p.a.</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card metric-loss">
        <div class="metric-label">Worst Historical Loss</div>
        <div class="metric-value">${worst_loss:,.0f}</div>
        <div class="metric-sub">{days}-day window</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Portfolio allocation pill row ──────────────────────────────────────────────
alloc_pills = " &nbsp;·&nbsp; ".join(
    [f"<b>{t}</b> {weights_dict[t]*100:.1f}%" for t in tickers]
)
st.markdown(
    f'<p style="font-family:Space Mono,monospace;font-size:0.68rem;color:#555;letter-spacing:0.06em;">'
    f'ALLOCATION &nbsp;→&nbsp; {alloc_pills}</p>',
    unsafe_allow_html=True
)
if method == "All (Compare)":
    st.markdown('<p class="section-label">// Multi-Method Comparison</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <table class="compare-table">
        <thead>
            <tr>
                <th>Method</th>
                <th>VaR ({confidence_pct}%)</th>
                <th>ES ({confidence_pct}%)</th>
                <th>Δ vs Historical VaR</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Historical Simulation</td>
                <td>${hist_VaR:,.2f}</td>
                <td>${hist_ES:,.2f}</td>
                <td>— baseline</td>
            </tr>
            <tr>
                <td>Parametric (Normal)</td>
                <td>${param_VaR:,.2f}</td>
                <td>${param_ES:,.2f}</td>
                <td>{((param_VaR - hist_VaR)/hist_VaR*100):+.1f}%</td>
            </tr>
            <tr>
                <td>Monte Carlo (10,000 sims)</td>
                <td>${mc_VaR:,.2f}</td>
                <td>${mc_ES:,.2f}</td>
                <td>{((mc_VaR - hist_VaR)/hist_VaR*100):+.1f}%</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart comparison ──
    methods   = ["Historical", "Parametric", "Monte Carlo"]
    var_vals  = [hist_VaR, param_VaR, mc_VaR]
    es_vals   = [hist_ES,  param_ES,  mc_ES]

    y_min  = min(var_vals + es_vals)
    y_max  = max(var_vals + es_vals)
    y_pad  = (y_max - y_min) * 0.45   # headroom above tallest bar for labels
    y_floor = max(0, y_min * 0.80)     # start axis at 80% of lowest value

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name=f"VaR ({confidence_pct}%)",
        x=methods, y=var_vals,
        marker_color=C_VAR,
        marker_line=dict(color="#8B1A1A", width=1),
        text=[f"<b>${v:,.0f}</b>" for v in var_vals],
        textposition="outside",
        textfont=dict(size=13, family="Space Mono, monospace", color="#1a1a1a"),
    ))
    fig_bar.add_trace(go.Bar(
        name=f"ES ({confidence_pct}%)",
        x=methods, y=es_vals,
        marker_color=C_ES,
        marker_line=dict(color="#4A0E0E", width=1),
        text=[f"<b>${v:,.0f}</b>" for v in es_vals],
        textposition="outside",
        textfont=dict(size=13, family="Space Mono, monospace", color="#1a1a1a"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8F9FA",
        font=dict(family="Space Mono, monospace", size=12, color="#1a1a1a"),
        margin=dict(l=80, r=40, t=80, b=65),
        height=420,
        title=dict(
            text=f"<b>VaR & ES by Method  |  {confidence_pct}% Confidence  |  {days}-Day Horizon</b>",
            font=dict(size=19, color="#1a1a1a"),
            x=0,
        ),
        xaxis=dict(
            showgrid=False,
            linecolor="#CCCCCC", linewidth=1,
            tickfont=dict(size=13, color="#1a1a1a", family="Space Mono, monospace"),
            title=dict(text="<b>Method</b>", font=dict(size=14, color="#1a1a1a")),
            ticks="outside", ticklen=4,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
            tickprefix="$", tickformat=",.0f",
            tickfont=dict(size=12, color="#333333"),
            title=dict(text="<b>Dollar Risk ($)</b>", font=dict(size=14, color="#1a1a1a")),
            linecolor="#CCCCCC", linewidth=1,
            ticks="outside", ticklen=4,
            range=[y_floor, y_max + y_pad],
        ),
        barmode="group",
        bargap=0.35,
        bargroupgap=0.06,
        legend=dict(
            orientation="h",
            font=dict(size=12, color="#1a1a1a"),
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            x=1.0, y=1.06,
            xanchor="right", yanchor="bottom",
        ),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <b>How to read this:</b> Historical VaR uses actual past returns — no distribution assumption.
        Parametric assumes normality (may underestimate fat tails). Monte Carlo simulates 10,000 paths
        from historical mean &amp; volatility. ES is always larger than VaR — it measures the average
        loss <i>given</i> that VaR is breached. A high ES/VaR ratio signals heavy tail risk.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Shared data ────────────────────────────────────────────────────────────────
range_returns_dollar = range_returns * portfolio_value
chart_kwargs = dict(
    days=days, confidence_pct=confidence_pct,
    portfolio_mean=portfolio_mean,
    portfolio_std_dev=portfolio_std_dev,
    portfolio_value=portfolio_value,
)

# ── Historical chart ───────────────────────────────────────────────────────────
if method in ("Historical", "All (Compare)"):
    st.markdown('<p class="section-label">// Historical Simulation</p>', unsafe_allow_html=True)
    fig_h = build_risk_chart(
        title=f"Historical VaR & ES — {days}-Day Horizon  |  {confidence_pct}% Confidence",
        x_data=range_returns_dollar.values,
        var_val=hist_VaR, es_val=hist_ES,
        x_label=f"{days}-Day Portfolio P&L ($)",
        kde_overlay=True, normal_curve=False,
        **chart_kwargs,
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ── Parametric chart ───────────────────────────────────────────────────────────
if method in ("Parametric", "All (Compare)"):
    st.markdown('<p class="section-label">// Parametric (Normal Distribution)</p>', unsafe_allow_html=True)
    fig_p = build_risk_chart(
        title=f"Parametric VaR & ES — {days}-Day Horizon  |  {confidence_pct}% Confidence",
        x_data=range_returns_dollar.values,
        var_val=param_VaR, es_val=param_ES,
        x_label=f"{days}-Day Portfolio P&L ($)",
        kde_overlay=False, normal_curve=True,
        **chart_kwargs,
    )
    st.plotly_chart(fig_p, use_container_width=True)

# ── Monte Carlo chart ──────────────────────────────────────────────────────────
if method in ("Monte Carlo", "All (Compare)"):
    st.markdown('<p class="section-label">// Monte Carlo Simulation — 10,000 Paths</p>', unsafe_allow_html=True)
    fig_mc = build_risk_chart(
        title=f"Monte Carlo VaR & ES — {days}-Day Horizon  |  {confidence_pct}% Confidence",
        x_data=mc_scenarios,
        var_val=mc_VaR, es_val=mc_ES,
        x_label="Simulated Gain / Loss ($)",
        kde_overlay=True, normal_curve=False,
        **chart_kwargs,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#888;letter-spacing:0.1em;">
        KARAM. &nbsp;|&nbsp; TAIL RISK ANALYZER
    </span>
    <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#888;letter-spacing:0.07em;">
        {', '.join(tickers)} &nbsp;·&nbsp; {lookback_years}Y &nbsp;·&nbsp; {days}D &nbsp;·&nbsp; {confidence_pct}% CONFIDENCE
    </span>
</div>
""", unsafe_allow_html=True)
