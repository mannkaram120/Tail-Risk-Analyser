import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from scipy.stats import norm, gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env file from project directory automatically
except ImportError:
    pass

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

/* ══ SIDEBAR — Institutional / Bloomberg-meets-Stripe ═══════════════════════ */
[data-testid="stSidebar"] {
    background:#F0EDE6;
    border-right:1px solid #D6D2CA;
    padding:0;
}
[data-testid="stSidebar"] > div:first-child {
    padding:16px 18px 24px 18px;
}
[data-testid="stSidebar"] * {
    font-family:'Space Mono',monospace !important;
    color:#1a1a1a !important;
}

/* ── Section headers ── */
.sb-section {
    font-family:'Space Mono',monospace;
    font-size:0.58rem;
    font-weight:400;
    letter-spacing:0.20em;
    color:#999;
    text-transform:uppercase;
    margin:20px 0 8px 0;
    padding-bottom:6px;
    border-bottom:1px solid #D0CCC5;
}
.sb-section:first-child { margin-top:8px; }

/* ── Native widget labels ── */
[data-testid="stSidebar"] label {
    font-size:0.60rem !important;
    font-weight:400 !important;
    letter-spacing:0.14em !important;
    color:#777 !important;
    text-transform:uppercase !important;
}

/* ── Text input ── */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] div[data-baseweb="input"] input[type="text"] {
    background:#F0EDE6 !important;
    background-color:#F0EDE6 !important;
    color:#1a1a1a !important;
    border:none !important;
    border-bottom:1px solid #C8C4BC !important;
    border-radius:0 !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.74rem !important;
    padding:6px 4px !important;
    box-shadow:none !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] div[data-baseweb="input"] {
    background:#F0EDE6 !important;
    background-color:#F0EDE6 !important;
    border:none !important;
    border-radius:0 !important;
    box-shadow:none !important;
}
[data-testid="stSidebar"] input[type="text"]::placeholder {
    color:#AAAAAA !important;
    font-style:normal !important;
}

/* ── Number input — force light background ── */
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] input,
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
    background:#F0EDE6 !important;
    background-color:#F0EDE6 !important;
    border:none !important;
    border-bottom:1px solid #C8C4BC !important;
    border-radius:0 !important;
    color:#1a1a1a !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.78rem !important;
    font-weight:700 !important;
    padding:2px 4px !important;
    box-shadow:none !important;
    height:28px !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"],
[data-testid="stSidebar"] div[data-baseweb="base-input"],
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    background:#F0EDE6 !important;
    background-color:#F0EDE6 !important;
    border:none !important;
    border-radius:0 !important;
    box-shadow:none !important;
}
/* stepper +/- buttons */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background:#E8E4DC !important;
    color:#555 !important;
    border:none !important;
    border-radius:0 !important;
}

/* ── Ticker row columns ── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap:0px !important;
    margin-bottom:-12px !important;
    align-items:center !important;
    min-height:28px !important;
}

/* ── Benchmark preset pill buttons ── */
[data-testid="stSidebar"] [data-testid^="stButton-preset_"] > button {
    background:#EDEAE3 !important;
    border:1px solid #D0CCC5 !important;
    color:#555 !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.58rem !important;
    font-weight:700 !important;
    letter-spacing:0.06em !important;
    padding:3px 2px !important;
    border-radius:2px !important;
    height:24px !important;
    line-height:1 !important;
}
[data-testid="stSidebar"] [data-testid^="stButton-preset_"] > button:hover {
    background:#D0CCC5 !important;
    color:#1a1a1a !important;
}

/* ── Remove / delete button — truly minimal ── */
[data-testid="stSidebar"] button[kind="secondary"] {
    padding:0 4px !important;
    min-height:20px !important;
    height:20px !important;
    font-size:0.65rem !important;
    line-height:1 !important;
    border:none !important;
    border-radius:0 !important;
    background:transparent !important;
    color:#BBBBBB !important;
    box-shadow:none !important;
    letter-spacing:0 !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    color:#555 !important;
    background:transparent !important;
}

/* ── Section toggle arrows — tiny, right-aligned ── */
[data-testid="stSidebar"] button[data-testid*="toggle_sb"] {
    background:transparent !important;
    border:none !important;
    color:#AAA !important;
    font-size:0.55rem !important;
    padding:2px 0 !important;
    min-height:20px !important;
    box-shadow:none !important;
    letter-spacing:0 !important;
}
[data-testid="stSidebar"] button[data-testid*="toggle_sb"]:hover {
    color:#555 !important;
}

/* ── All sidebar buttons default: light/transparent ── */
[data-testid="stSidebar"] .stButton > button {
    font-family:'Space Mono',monospace !important;
    font-size:0.62rem !important;
    background:transparent !important;
    border:1px solid #C8C4BC !important;
    border-radius:0 !important;
    color:#555 !important;
    padding:4px 8px !important;
    letter-spacing:0.08em !important;
    transition:all 0.12s ease !important;
    box-shadow:none !important;
    min-height:28px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color:#1a1a1a !important;
    color:#1a1a1a !important;
    background:transparent !important;
}

/* ── CALCULATE RISK — only the run button, targeted by key ── */
[data-testid="stSidebar"] [data-testid="stButton-run_btn"] > button,
button[data-testid="run_btn"] {
    background:#1a1a1a !important;
    color:#F5F2EB !important;
    border:none !important;
    border-radius:0 !important;
    font-size:0.66rem !important;
    letter-spacing:0.20em !important;
    padding:11px 0 !important;
    width:100% !important;
    font-weight:400 !important;
}
[data-testid="stSidebar"] [data-testid="stButton-run_btn"] > button:hover {
    background:#333 !important;
}

/* ── Sliders — thin, muted ── */
div[data-testid="stSlider"] * {
    color:#555 !important;
    font-family:'Space Mono',monospace !important;
}
div[data-testid="stSlider"] p {
    font-size:0.60rem !important;
    font-weight:400 !important;
    letter-spacing:0.12em !important;
    color:#777 !important;
}
div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    font-size:0.72rem !important;
    font-weight:700 !important;
    color:#1a1a1a !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background:#C8C4BC !important;
    height:2px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
    background:#1a1a1a !important;
    height:2px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background:#1a1a1a !important;
    border:none !important;
    width:10px !important;
    height:10px !important;
    box-shadow:none !important;
}

/* ── Radio buttons — minimal ── */
div[data-testid="stRadio"] label {
    color:#555 !important;
    font-size:0.68rem !important;
    font-weight:400 !important;
    letter-spacing:0.08em !important;
}
div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size:0.68rem !important;
    color:#555 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap:2px !important;
}

/* ══ EXPANDERS — main content: normal boxes ════════════════════════════════ */
[data-testid="stMain"] [data-testid="stExpander"] {
    border:1px solid #D6D2CA !important;
    border-radius:4px !important;
    background:#F9F7F3 !important;
    margin-bottom:8px !important;
}
[data-testid="stMain"] [data-testid="stExpander"] summary {
    font-family:'Space Mono',monospace !important;
    font-size:0.68rem !important;
    letter-spacing:0.12em !important;
    color:#555 !important;
    padding:10px 14px !important;
    background:#F9F7F3 !important;
}
[data-testid="stMain"] [data-testid="stExpander"] summary:hover {
    background:#F0EDE6 !important;
    color:#1a1a1a !important;
}
[data-testid="stMain"] [data-testid="stExpander"] * {
    color:#1a1a1a !important;
    font-family:'Space Mono',monospace !important;
}
[data-testid="stMain"] [data-testid="stExpander"] > div > div {
    padding:14px 16px !important;
    background:#F9F7F3 !important;
}

/* ── Main content inputs — white background, readable ── */
[data-testid="stMain"] input[type="text"],
[data-testid="stMain"] input[type="number"],
[data-testid="stMain"] [data-testid="stTextInput"] input,
[data-testid="stMain"] [data-testid="stNumberInput"] input,
[data-testid="stMain"] div[data-baseweb="input"] input,
[data-testid="stMain"] div[data-baseweb="base-input"] input {
    background:#FFFFFF !important;
    background-color:#FFFFFF !important;
    color:#1a1a1a !important;
    border:1px solid #D6D2CA !important;
    border-radius:3px !important;
    font-family:'Space Mono',monospace !important;
    font-size:0.76rem !important;
    box-shadow:none !important;
}
[data-testid="stMain"] div[data-baseweb="input"],
[data-testid="stMain"] div[data-baseweb="base-input"],
[data-testid="stMain"] [data-testid="stTextInput"] > div,
[data-testid="stMain"] [data-testid="stNumberInput"] > div {
    background:#FFFFFF !important;
    background-color:#FFFFFF !important;
    border:1px solid #D6D2CA !important;
    border-radius:3px !important;
    box-shadow:none !important;
}
[data-testid="stMain"] input::placeholder {
    color:#AAAAAA !important;
    font-style:italic !important;
}
[data-testid="stMain"] [data-testid="stNumberInput"] button {
    background:#F5F2EB !important;
    color:#555 !important;
    border:none !important;
    border-left:1px solid #D6D2CA !important;
}
/* ══ EXPANDERS — sidebar: flat, no box ════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border:none !important;
    border-radius:0 !important;
    margin-bottom:0 !important;
    background:transparent !important;
    box-shadow:none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-family:'Space Mono',monospace !important;
    font-size:0.60rem !important;
    font-weight:400 !important;
    letter-spacing:0.20em !important;
    color:#888888 !important;
    padding:8px 0 !important;
    border-bottom:1px solid #D0CCC5 !important;
    background:transparent !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background:transparent !important;
    color:#1a1a1a !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] > div > div {
    padding:8px 0 0 0 !important;
}

/* ══ MAIN SECTION LABEL ═════════════════════════════════════════════════════ */
.section-label {
    font-family:'Space Mono',monospace;
    font-size:0.60rem;
    font-weight:400;
    letter-spacing:0.16em;
    color:#888;
    text-transform:uppercase;
    margin-bottom:0.3rem;
    margin-top:0.7rem;
}

/* ══ METRIC CARDS ══════════════════════════════════════════════════════════ */
.metric-card {
    background:#FFFFFF;
    border:none;
    border-radius:0;
    padding:14px 18px 12px 18px;
    font-family:'Space Mono',monospace;
    box-shadow:none;
    border-top:2px solid #E8E4DC;
}
.metric-label { font-size:0.52rem; letter-spacing:0.16em; color:#999; text-transform:uppercase; margin-bottom:10px; }
.metric-value { font-size:1.55rem; font-weight:700; color:#1a1a1a; line-height:1; letter-spacing:-0.02em; }
.metric-sub   { font-size:0.58rem; color:#BBB; margin-top:8px; letter-spacing:0.06em; }
.metric-var   { border-top:2px solid #C0392B; }
.metric-es    { border-top:2px solid #7B241C; }
.metric-vol   { border-top:2px solid #555; }
.metric-loss  { border-top:2px solid #AAAAAA; }

/* ══ TABS ═══════════════════════════════════════════════════════════════════ */
div[data-testid="stTabs"] button {
    font-family:'Space Mono',monospace !important;
    font-size:0.68rem !important;
    font-weight:400 !important;
    letter-spacing:0.12em !important;
    color:#888 !important;
    text-transform:uppercase !important;
    padding:10px 18px !important;
    border-radius:0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color:#1a1a1a !important;
    border-bottom:1px solid #1a1a1a !important;
    font-weight:700 !important;
}
div[data-testid="stTabs"] [role="tablist"] {
    border-bottom:1px solid #D4CFC4 !important;
    gap:0 !important;
}

/* ══ CHART CARD ════════════════════════════════════════════════════════════ */
.chart-card {
    background:#FFFFFF;
    border-radius:0;
    padding:4px 4px 0 4px;
    box-shadow:none;
    border:1px solid #E8E4DC;
    margin-bottom:8px;
}

/* ══ INTERPRETATION PANEL ══════════════════════════════════════════════════ */
.interp-panel {
    background:#FFFFFF;
    border-radius:0;
    padding:20px 24px;
    border:1px solid #E8E4DC;
    margin-top:8px;
}
.interp-title {
    font-family:'Space Mono',monospace;
    font-size:0.58rem;
    font-weight:400;
    letter-spacing:0.20em;
    color:#999;
    text-transform:uppercase;
    margin-bottom:14px;
}
.interp-text { font-size:0.88rem; color:#2a2a2a; line-height:1.8; font-family:'Inter',sans-serif; }
.interp-text b { color:#C0392B; }
.interp-highlight {
    background:#F9F7F3;
    border-left:2px solid #C0392B;
    padding:10px 16px;
    border-radius:0;
    margin:14px 0;
    font-size:0.82rem;
    color:#444;
    font-family:'Space Mono',sans-serif;
    line-height:1.7;
}

/* ══ COMPARISON TABLE ══════════════════════════════════════════════════════ */
.compare-table { width:100%; font-family:'Space Mono',monospace; font-size:0.76rem; border-collapse:collapse; }
.compare-table th { background:#1a1a1a; color:#F5F2EB; padding:10px 14px; text-align:left; font-size:0.58rem; letter-spacing:0.12em; text-transform:uppercase; font-weight:400; }
.compare-table td { padding:10px 14px; border-bottom:1px solid #EDEAE3; color:#1a1a1a; }
.compare-table tr:hover td { background:#F5F2EB; }

/* ══ INFO BOX ══════════════════════════════════════════════════════════════ */
.info-box { background:#F9F7F3; border-left:2px solid #888; padding:12px 16px; font-size:0.75rem; color:#555; margin:12px 0; font-family:'Inter',sans-serif; line-height:1.7; }

/* ── Collapsible Interpretation / Diagnostic expanders ───────────────────── */
[data-testid="stMain"] [data-testid="stExpander"] details {
    border: none !important;
    background: transparent !important;
    margin-bottom: 4px !important;
}
[data-testid="stMain"] [data-testid="stExpander"] details summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.60rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.16em !important;
    color: #888 !important;
    text-transform: uppercase !important;
    padding: 11px 18px !important;
    background: #F5F2EB !important;
    border: 1px solid #D6D2CA !important;
    border-radius: 0 !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stMain"] [data-testid="stExpander"] details[open] > summary {
    background: #EDEAE3 !important;
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
}
[data-testid="stMain"] [data-testid="stExpander"] details summary:hover {
    background: #EDEAE3 !important;
    color: #1a1a1a !important;
}
[data-testid="stMain"] [data-testid="stExpander"] details > div > div {
    padding: 0 !important;
}

/* ══ MISC ══════════════════════════════════════════════════════════════════ */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1rem !important; padding-bottom:1rem !important; max-width:1200px !important; }
hr { border:none; border-top:1px solid #D4CFC4; margin:1.2rem 0; }
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
def base_layout(title_text, x_title, height=360):
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8F9FA",
        font=dict(family="Space Mono, monospace", size=12, color="#1a1a1a"),
        margin=dict(l=65, r=30, t=55, b=55),
        height=height,
        title=dict(
            text=title_text,
            font=dict(size=13, color="#666666", family="Space Mono, monospace"),
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

# Session state for collapsible sections
for _k in ["sb_assets", "sb_params", "sb_risk", "show_quick_add"]:
    if _k not in st.session_state:
        st.session_state[_k] = True if _k != "show_quick_add" else False

def _section_header(label, key):
    """Render a clickable section header that toggles collapse state."""
    arrow = "▲" if st.session_state[key] else "▼"
    col_l, col_r = st.sidebar.columns([5, 1])
    col_l.markdown(
        f'''<div style="font-family:Space Mono,monospace;font-size:0.58rem;font-weight:400;
        letter-spacing:0.20em;color:#999;text-transform:uppercase;padding:6px 0 6px 0;
        border-bottom:1px solid #D0CCC5;margin-bottom:0;">{label}</div>''',
        unsafe_allow_html=True
    )
    if col_r.button(arrow, key=f"toggle_{key}"):
        st.session_state[key] = not st.session_state[key]
        st.rerun()

with st.sidebar:

    # ── Wordmark ──
    st.markdown("""
    <div style="padding-bottom:16px;border-bottom:1px solid #D0CCC5;margin-bottom:4px;">
        <div style="font-family:Space Mono,monospace;font-size:1.0rem;font-weight:700;
                    color:#1a1a1a;letter-spacing:0.02em;">KARAM.</div>
        <div style="font-family:Space Mono,monospace;font-size:0.55rem;color:#999;
                    letter-spacing:0.20em;margin-top:2px;text-transform:uppercase;">
            Tail Risk Analyzer
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══ SECTION 1: ASSET SELECTION ═══════════════════════════════════════════════
    _section_header("Asset Selection", "sb_assets")

    if st.session_state["sb_assets"]:
        # Column headers
        st.markdown("""
        <div style="display:grid;grid-template-columns:2fr 1.6fr 0.4fr;
                    padding:6px 0 4px 0;border-bottom:1px solid #C8C4BC;margin-top:6px;">
            <span style="font-family:Space Mono,monospace;font-size:0.52rem;
                         letter-spacing:0.14em;color:#AAA;text-transform:uppercase;">Ticker</span>
            <span style="font-family:Space Mono,monospace;font-size:0.52rem;
                         letter-spacing:0.14em;color:#AAA;text-transform:uppercase;">Wt %</span>
            <span></span>
        </div>
        """, unsafe_allow_html=True)

        n_tickers = len(st.session_state.tickers)
        default_w = round(100.0 / n_tickers, 1) if n_tickers > 0 else 100.0
        raw_weights = {}

        for t in list(st.session_state.tickers):
            col_name, col_w, col_x = st.columns([2, 1.6, 0.4])
            with col_name:
                st.markdown(
                    f'<div style="font-family:Space Mono,monospace;font-size:0.74rem;font-weight:700;'
                    f'color:#1a1a1a;padding:5px 0 0 0;">{t}</div>',
                    unsafe_allow_html=True
                )
            with col_w:
                raw_weights[t] = st.number_input(
                    f"w_{t}", min_value=0.0, max_value=100.0,
                    value=default_w, step=0.5, format="%.1f",
                    key=f"weight_{t}", label_visibility="collapsed",
                )
            with col_x:
                if st.button("×", key=f"remove_{t}"):
                    st.session_state.tickers.remove(t)
                    st.rerun()

        # Total allocation
        total_w = sum(raw_weights.values())
        ok = abs(total_w - 100.0) <= 0.11
        alloc_color = "#888" if ok else "#C0392B"
        alloc_sym   = "✓" if ok else "⚠"
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.58rem;color:{alloc_color};'
            f'text-align:right;padding:4px 0 2px 0;border-top:1px solid #E0DDD6;letter-spacing:0.06em;">'
            f'{alloc_sym}&nbsp;{total_w:.1f}%</div>',
            unsafe_allow_html=True
        )

        new_ticker = st.text_input(
            "Add Ticker", value="", placeholder="e.g. TSLA  →  enter",
            key="ticker_input", label_visibility="collapsed",
        )
        if new_ticker:
            t = new_ticker.strip().upper()
            if t and t not in st.session_state.tickers:
                st.session_state.tickers.append(t)
            st.rerun()

        qa_label = "QUICK ADD +" if not st.session_state.show_quick_add else "QUICK ADD −"
        if st.button(qa_label, key="toggle_qa"):
            st.session_state.show_quick_add = not st.session_state.show_quick_add
            st.rerun()
        if st.session_state.show_quick_add:
            cols = st.columns(4)
            for i, qt in enumerate(QUICK_ADD):
                if cols[i % 4].button(qt, key=f"qa_{qt}"):
                    if qt not in st.session_state.tickers:
                        st.session_state.tickers.append(qt)
                    st.rerun()
    else:
        # placeholder so variables exist
        raw_weights = {t: st.session_state.get(f"weight_{t}", round(100.0/len(st.session_state.tickers),1))
                       for t in st.session_state.tickers}
        total_w = sum(raw_weights.values())

    # ══ SECTION 2: PARAMETERS ════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    _section_header("Parameters", "sb_params")

    if st.session_state["sb_params"]:
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=1000, max_value=10_000_000,
            value=100_000, step=10_000,
        )

        # Lookback and Horizon — clean table layout
        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0;
                    padding:8px 0 4px 0;border-top:1px solid #E0DDD6;margin-top:6px;">
            <span style="font-family:Space Mono,monospace;font-size:0.52rem;
                         letter-spacing:0.14em;color:#AAA;text-transform:uppercase;">Lookback</span>
            <span style="font-family:Space Mono,monospace;font-size:0.52rem;
                         letter-spacing:0.14em;color:#AAA;text-transform:uppercase;">Horizon</span>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            lookback_years = st.select_slider(
                "Lookback", options=[1, 2, 3, 5, 7, 10], value=5,
                format_func=lambda x: f"{x}Y",
                label_visibility="collapsed",
            )
        with col_b:
            days = st.select_slider(
                "Horizon", options=list(range(1, 31)), value=5,
                format_func=lambda x: f"{x}D",
                label_visibility="collapsed",
            )
    else:
        portfolio_value = st.session_state.get("portfolio_value_input", 100_000)
        lookback_years  = 5
        days            = 5

    # ══ SECTION 3: RISK SETTINGS ═════════════════════════════════════════════════
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    _section_header("Risk Settings", "sb_risk")

    if st.session_state["sb_risk"]:
        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        confidence_pct = st.select_slider(
            "Confidence Level",
            options=[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5, 99.9],
            value=95,
            format_func=lambda x: f"{x}%",
        )
        confidence = confidence_pct / 100

        method = st.radio(
            "View Mode",
            options=["All (Compare)", "Historical", "Parametric", "Monte Carlo"],
            index=0,
            horizontal=False,
            label_visibility="collapsed",
        )
    else:
        confidence_pct = 95
        confidence     = 0.95
        method         = "All (Compare)"

    # ══ BENCHMARK PORTFOLIOS ═════════════════════════════════════════════════════
    st.markdown('<div class="sb-section" style="margin-top:22px;">BENCHMARKS</div>', unsafe_allow_html=True)

    # ── Canonical benchmark presets ───────────────────────────────────────────
    BENCH_PRESETS = [
        {"name": "S&P 500 (SPY)",   "short": "SPY",   "tickers": "SPY",     "weights": "100"},
        {"name": "Nasdaq (QQQ)",    "short": "QQQ",   "tickers": "QQQ",     "weights": "100"},
        {"name": "Dow Jones (DIA)", "short": "DIA",   "tickers": "DIA",     "weights": "100"},
        {"name": "Bonds (BND)",     "short": "BND",   "tickers": "BND",     "weights": "100"},
        {"name": "60/40 (SPY+BND)", "short": "60/40", "tickers": "SPY,BND", "weights": "60,40"},
    ]

    # Version key — bump to force-reset stale session state
    _BENCH_VERSION = "v5-indices"
    if st.session_state.get("sa_portfolios_version") != _BENCH_VERSION:
        for _si in range(2):
            for _sk in [f"sb_tickers_{_si}", f"sb_weights_{_si}",
                        f"sb_value_{_si}", f"sb_en_{_si}"]:
                st.session_state.pop(_sk, None)
        st.session_state.sa_portfolios = [
            {"name": "S&P 500 (SPY)", "tickers": "SPY", "weights": "100",
             "value": 100_000, "enabled": False},
            {"name": "Nasdaq (QQQ)",  "tickers": "QQQ", "weights": "100",
             "value": 100_000, "enabled": False},
        ]
        st.session_state.sa_portfolios_version = _BENCH_VERSION

    # Sync widget keys → session state on every run
    for _si in range(2):
        for _sf, _sk in [("tickers", f"sb_tickers_{_si}"), ("weights", f"sb_weights_{_si}"),
                          ("value",   f"sb_value_{_si}"),   ("enabled", f"sb_en_{_si}")]:
            if _sk in st.session_state:
                st.session_state.sa_portfolios[_si][_sf] = st.session_state[_sk]

    # ── Preset pill buttons: render as styled HTML links via st.markdown + st.button ──
    # Two rows of pills: row 1 = SPY QQQ DIA, row 2 = BND 60/40
    for _row_presets in [BENCH_PRESETS[:3], BENCH_PRESETS[3:]]:
        _rcols = st.sidebar.columns(len(_row_presets))
        for _pc, _bp in zip(_rcols, _row_presets):
            if _pc.button(_bp["short"], key=f"preset_{_bp['short']}",
                          use_container_width=True):
                _slot = 1 if st.session_state.sa_portfolios[0].get("enabled") else 0
                st.session_state.sa_portfolios[_slot].update({
                    "name": _bp["name"], "tickers": _bp["tickers"], "weights": _bp["weights"]
                })
                st.session_state[f"sb_tickers_{_slot}"] = _bp["tickers"]
                st.session_state[f"sb_weights_{_slot}"] = _bp["weights"]
                st.rerun()

    # ── Thin divider ─────────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="border-top:1px solid #D6D2CA;margin:10px 0 8px 0;"></div>',
        unsafe_allow_html=True)

    # ── Per-slot: compact single-row layout ───────────────────────────────────
    for _si, _sp in enumerate(st.session_state.sa_portfolios):
        _enabled = _sp["enabled"]
        _accent  = "#C0392B" if _enabled else "#AAAAAA"
        _status  = "ACTIVE" if _enabled else "OFF"

        # Name row with status badge
        st.sidebar.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin:0 0 4px 0;">'
            f'<span style="font-family:Space Mono,monospace;font-size:0.58rem;'
            f'color:#555;letter-spacing:0.10em;text-transform:uppercase;'
            f'font-weight:700;">{_sp["name"]}</span>'
            f'<span style="font-family:Space Mono,monospace;font-size:0.50rem;'
            f'color:{_accent};letter-spacing:0.12em;">{_status}</span>'
            f'</div>',
            unsafe_allow_html=True)

        # Ticker input + toggle on same row
        _c1, _c2 = st.sidebar.columns([4, 1])
        with _c1:
            st.text_input("Tickers", value=_sp["tickers"], key=f"sb_tickers_{_si}",
                          label_visibility="collapsed", placeholder="e.g. SPY")
        with _c2:
            st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
            st.toggle("", value=_enabled, key=f"sb_en_{_si}",
                      label_visibility="collapsed")

        # Weights + value only shown when enabled
        if st.session_state.get(f"sb_en_{_si}", False):
            _cw, _cv = st.sidebar.columns([2, 3])
            with _cw:
                st.text_input("Wt", value=_sp["weights"], key=f"sb_weights_{_si}",
                              label_visibility="collapsed", placeholder="wts")
            with _cv:
                st.number_input("Val", value=int(_sp["value"]), min_value=1000,
                                max_value=10_000_000, step=10_000, key=f"sb_value_{_si}",
                                label_visibility="collapsed")

        # Separator between slots
        if _si < len(st.session_state.sa_portfolios) - 1:
            st.sidebar.markdown(
                '<div style="border-top:1px dashed #E0DDD7;margin:8px 0;"></div>',
                unsafe_allow_html=True)

    # ══ RUN ══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    run = st.button("CALCULATE RISK", key="run_btn", use_container_width=True)

# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<p class="section-label">// Quantitative Risk Tool</p>
<h1 style="font-family:'Space Mono',monospace;font-size:2rem;line-height:1.1;margin:0;font-weight:700;letter-spacing:-0.02em;color:#1a1a1a;">TAIL RISK</h1>
<h1 style="font-family:'Space Mono',monospace;font-size:2rem;line-height:1.1;margin:0;font-weight:700;letter-spacing:-0.02em;color:#B5B0A4;">ANALYZER</h1>
<p style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#555;margin-top:0.8rem;letter-spacing:0.05em;">Real market data via yFinance &nbsp;·&nbsp; Historical &nbsp;·&nbsp; Parametric &nbsp;·&nbsp; Monte Carlo</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Run gate: allow re-render from cached results without re-clicking Calculate ──
_has_cached = bool(st.session_state.get("_cached_results"))

if not run and not _has_cached:
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

@st.cache_data(show_spinner=False, ttl=3600)
def load_market_data(tickers_tuple, lookback_years, days):
    """Fetch and process market data. Cached for 1 hour to avoid redundant API calls."""
    enddate   = dt.datetime.now()
    startdate = enddate - dt.timedelta(days=365 * lookback_years)
    adj_close_df = pd.DataFrame()
    for ticker in tickers_tuple:
        data = yf.download(ticker, start=startdate, end=enddate,
                           auto_adjust=False, progress=False)
        if data.empty:
            return None, f"Could not fetch data for {ticker}. Please check the ticker symbol."
        _col = data['Adj Close']
        if isinstance(_col, pd.DataFrame):
            _col = _col.iloc[:, 0]
        adj_close_df[ticker] = _col
    log_returns      = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    return log_returns, None


enddate   = dt.datetime.now()
startdate = enddate - dt.timedelta(days=365 * lookback_years)

with st.spinner("Fetching market data..."):
    try:
        log_returns, _fetch_err = load_market_data(
            tuple(tickers), lookback_years, days
        )
        if _fetch_err:
            st.error(f"❌ {_fetch_err}")
            st.stop()
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
    """True path-based Monte Carlo: simulate daily returns for `days` steps.
    Each of `sims` paths accumulates daily log-returns drawn from N(mu_daily, sigma_daily),
    capturing path dependence and compounding — unlike single-step parametric sampling.
    Returns VaR, ES, terminal P&L distribution, and the full path matrix for drawdown.
    """
    np.random.seed(42)
    mu_daily    = portfolio_mean                       # already daily log-return mean
    sigma_daily = portfolio_std_dev / np.sqrt(252)    # daily volatility

    # Shape: (sims, days) — each row is one simulated path of daily log-returns
    daily_rets  = np.random.normal(mu_daily, sigma_daily, size=(sims, days))

    # Cumulative log-return paths → price paths (relative)
    cum_log_ret = np.cumsum(daily_rets, axis=1)        # (sims, days)
    price_paths = np.exp(cum_log_ret)                  # (sims, days), starts at 1.0

    # Terminal P&L
    terminal_ret = cum_log_ret[:, -1]                  # log-return over full horizon
    scenario_pnl = portfolio_value * (np.exp(terminal_ret) - 1.0)

    VaR  = -np.percentile(scenario_pnl, 100 * (1 - confidence))
    tail = scenario_pnl[scenario_pnl <= -VaR]
    ES   = float(-tail.mean()) if len(tail) > 0 else VaR

    return float(VaR), float(ES), scenario_pnl, price_paths

hist_VaR,  hist_ES             = calc_historical(range_returns, portfolio_value, confidence)
param_VaR, param_ES            = calc_parametric(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days)
mc_VaR,    mc_ES, mc_scenarios, mc_paths = calc_montecarlo(portfolio_value, portfolio_std_dev, portfolio_mean, confidence, days)

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
        <div class="metric-sub">portfolio \u03c3 p.a.</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card metric-loss">
        <div class="metric-label">Worst Historical Loss</div>
        <div class="metric-value">${worst_loss:,.0f}</div>
        <div class="metric-sub">{days}-day window</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

# ── Portfolio allocation pill row ──────────────────────────────────────────────
alloc_pills = " &nbsp;\u00b7&nbsp; ".join(
    [f"<b>{t}</b> {weights_dict[t]*100:.1f}%" for t in tickers]
)
st.markdown(
    f'''<p style="font-family:Space Mono,monospace;font-size:0.68rem;color:#555;letter-spacing:0.06em;">'''
    f'''ALLOCATION &nbsp;\u2192&nbsp; {alloc_pills}</p>''',
    unsafe_allow_html=True
)

# ── Shared chart data ──────────────────────────────────────────────────────────
range_returns_dollar = range_returns * portfolio_value
chart_kwargs = dict(
    days=days, confidence_pct=confidence_pct,
    portfolio_mean=portfolio_mean,
    portfolio_std_dev=portfolio_std_dev,
    portfolio_value=portfolio_value,
)

# ── Comparison table + bar chart (All Compare mode) ───────────────────────────
if method == "All (Compare)":
    st.markdown('<p class="section-label">// Multi-Method Comparison</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="compare-table">
        <thead><tr>
            <th>Method</th>
            <th>VaR ({confidence_pct}%)</th>
            <th>ES ({confidence_pct}%)</th>
            <th>\u0394 vs Historical VaR</th>
        </tr></thead>
        <tbody>
            <tr><td>Historical Simulation</td><td>${hist_VaR:,.2f}</td><td>${hist_ES:,.2f}</td><td>\u2014 baseline</td></tr>
            <tr><td>Parametric (Normal)</td><td>${param_VaR:,.2f}</td><td>${param_ES:,.2f}</td><td>{((param_VaR - hist_VaR)/hist_VaR*100):+.1f}%</td></tr>
            <tr><td>Monte Carlo (10,000 sims)</td><td>${mc_VaR:,.2f}</td><td>${mc_ES:,.2f}</td><td>{((mc_VaR - hist_VaR)/hist_VaR*100):+.1f}%</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    
    methods_list = ["Historical", "Parametric", "Monte Carlo"]
    var_vals     = [hist_VaR, param_VaR, mc_VaR]
    es_vals      = [hist_ES,  param_ES,  mc_ES]
    y_min   = min(var_vals + es_vals)
    y_max   = max(var_vals + es_vals)
    y_pad   = (y_max - y_min) * 0.45
    y_floor = max(0, y_min * 0.80)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name=f"VaR ({confidence_pct}%)", x=methods_list, y=var_vals,
        marker_color=C_VAR, marker_line=dict(color="#8B1A1A", width=1),
        text=[f"<b>${v:,.0f}</b>" for v in var_vals],
        textposition="outside",
        textfont=dict(size=13, family="Space Mono, monospace", color="#1a1a1a"),
    ))
    fig_bar.add_trace(go.Bar(
        name=f"ES ({confidence_pct}%)", x=methods_list, y=es_vals,
        marker_color=C_ES, marker_line=dict(color="#4A0E0E", width=1),
        text=[f"<b>${v:,.0f}</b>" for v in es_vals],
        textposition="outside",
        textfont=dict(size=13, family="Space Mono, monospace", color="#1a1a1a"),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8F9FA",
        font=dict(family="Space Mono, monospace", size=12, color="#1a1a1a"),
        margin=dict(l=65, r=30, t=55, b=55), height=340,
        title=dict(text=f"<b>VaR & ES by Method  |  {confidence_pct}% Confidence  |  {days}-Day Horizon</b>",
                   font=dict(size=19, color="#1a1a1a"), x=0),
        xaxis=dict(showgrid=False, linecolor="#CCCCCC", linewidth=1,
                   tickfont=dict(size=13, color="#1a1a1a"),
                   title=dict(text="<b>Method</b>", font=dict(size=14, color="#1a1a1a")),
                   ticks="outside", ticklen=4),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=1,
                   tickprefix="$", tickformat=",.0f",
                   tickfont=dict(size=12, color="#333333"),
                   title=dict(text="<b>Dollar Risk ($)</b>", font=dict(size=14, color="#1a1a1a")),
                   linecolor="#CCCCCC", linewidth=1, ticks="outside", ticklen=4,
                   range=[y_floor, y_max + y_pad]),
        barmode="group", bargap=0.35, bargroupgap=0.06,
        legend=dict(orientation="h", font=dict(size=12, color="#1a1a1a"),
                    bgcolor="rgba(255,255,255,0)", borderwidth=0,
                    x=1.0, y=1.06, xanchor="right", yanchor="bottom"),
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>How to read this:</b> Historical VaR uses actual past returns \u2014 no distribution assumption.
        Parametric assumes normality (may underestimate fat tails). Monte Carlo simulates 10,000 paths
        from historical mean &amp; volatility. ES is always larger than VaR \u2014 it measures the average
        loss <i>given</i> that VaR is breached. A high ES/VaR ratio signals heavy tail risk.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Charts: Tabs for individual methods, stacked for All Compare ──────────────
if method == "All (Compare)":
    tab1, tab2, tab3 = st.tabs(["\u2460  Historical Simulation", "\u2461  Parametric (Normal)", "\u2462  Monte Carlo"])
else:
    tab1 = tab2 = tab3 = None

def render_hist(container):
    fig_h = build_risk_chart(
        title=f"Historical Simulation  ·  {days}-Day Horizon  ·  {confidence_pct}% Confidence",
        x_data=range_returns_dollar.values,
        var_val=hist_VaR, es_val=hist_ES,
        x_label=f"{days}-Day Portfolio P&L ($)",
        kde_overlay=True, normal_curve=False, **chart_kwargs,
    )
    container.markdown('<div class="chart-card">', unsafe_allow_html=True)
    container.plotly_chart(fig_h, use_container_width=True)
    container.markdown('</div>', unsafe_allow_html=True)

def render_param(container):
    fig_p = build_risk_chart(
        title=f"Parametric (Normal)  ·  {days}-Day Horizon  ·  {confidence_pct}% Confidence",
        x_data=range_returns_dollar.values,
        var_val=param_VaR, es_val=param_ES,
        x_label=f"{days}-Day Portfolio P&L ($)",
        kde_overlay=False, normal_curve=True, **chart_kwargs,
    )
    container.markdown('<div class="chart-card">', unsafe_allow_html=True)
    container.plotly_chart(fig_p, use_container_width=True)
    container.markdown('</div>', unsafe_allow_html=True)

def render_mc(container):
    fig_mc = build_risk_chart(
        title=f"Monte Carlo (10,000 sims)  ·  {days}-Day Horizon  ·  {confidence_pct}% Confidence",
        x_data=mc_scenarios,
        var_val=mc_VaR, es_val=mc_ES,
        x_label="Simulated Gain / Loss ($)",
        kde_overlay=True, normal_curve=False, **chart_kwargs,
    )
    container.markdown('<div class="chart-card">', unsafe_allow_html=True)
    container.plotly_chart(fig_mc, use_container_width=True)
    container.markdown('</div>', unsafe_allow_html=True)

if method == "Historical":
    render_hist(st)
elif method == "Parametric":
    render_param(st)
elif method == "Monte Carlo":
    render_mc(st)
elif method == "All (Compare)":
    with tab1: render_hist(tab1)
    with tab2: render_param(tab2)
    with tab3: render_mc(tab3)

# ── Interpretation variables ────────────────────────────────────────────────────
if method == "All (Compare)":
    interp_var   = hist_VaR
    interp_es    = hist_ES
    es_var_ratio = hist_ES / hist_VaR
else:
    interp_var   = display_VaR
    interp_es    = display_ES
    es_var_ratio = display_ES / display_VaR if display_VaR > 0 else 0

annualised_var = interp_var * np.sqrt(252 / days)
tail_risk_label = "elevated" if es_var_ratio > 1.5 else "moderate" if es_var_ratio > 1.2 else "contained"

# ── Interpretation — updates automatically per selected method ──────────────
with st.expander("📖  INTERPRETATION", expanded=False):

    _ann = f"~${annualised_var:,.0f}"

    _method_cfg = {
        "Historical":    ("#C0392B", "HISTORICAL SIMULATION",
                          "No distributional assumption — tail behaviour taken directly"
                          " from observed rolling returns."),
        "Parametric":    ("#2C6EAB", "PARAMETRIC  ·  NORMAL DISTRIBUTION",
                          "Assumes returns are normally distributed. Efficient but may"
                          " underestimate fat-tail losses."),
        "Monte Carlo":   ("#3D7A4F", "MONTE CARLO  ·  10,000 SIMULATED PATHS",
                          "Simulates 10,000 paths from historical μ and σ."
                          " Captures distributional shape; assumes i.i.d. daily returns."),
        "All (Compare)": ("#1a1a1a", "ALL METHODS  ·  SIDE-BY-SIDE COMPARISON",
                          "All three methods computed simultaneously."
                          " Divergence between them reveals model risk and tail sensitivity."),
    }
    _col, _lbl, _note = _method_cfg.get(method, ("#888", method.upper(), ""))

    # core paragraph — use <strong> not ** (rendered inside raw HTML)
    _para = (
        f"At <strong>{confidence_pct}% confidence</strong>, over a <strong>{days}-day horizon</strong>, "
        f"this portfolio's maximum expected loss is <strong>${interp_var:,.0f}</strong> "
        f"({interp_var / portfolio_value * 100:.2f}% of portfolio value). "
        f"When that threshold is breached, the average loss is <strong>${interp_es:,.0f}</strong> "
        f"— an ES/VaR ratio of <strong>{es_var_ratio:.2f}×</strong> ({tail_risk_label} tail risk)."
    )

    if method == "Historical":
        _para += (
            f" The {days}-day window spans <strong>{int(len(range_returns))}</strong> observed periods."
            f" Annual volatility: <strong>{annual_vol:.2f}%</strong>,"
            f" annualised return: <strong>{mean_return_ann:.2f}%</strong>,"
            " both derived from the same historical sample."
        )
    elif method == "Parametric":
        _cmp_p = "exceeds" if param_VaR > hist_VaR else "falls below"
        _why_p = ("the normal distribution inflates the tail estimate"
                  if param_VaR > hist_VaR else
                  "actual returns carry fatter tails than the normal distribution assumes")
        _para += (
            f" Parametric VaR (<strong>${param_VaR:,.0f}</strong>) {_cmp_p} Historical (<strong>${hist_VaR:,.0f}</strong>)"
            f" — {_why_p}."
        )
    elif method == "Monte Carlo":
        _cmp_mc = "exceeds" if mc_VaR > hist_VaR else "falls below"
        _why_mc = ("simulated tails are fatter than observed history"
                   if mc_VaR > hist_VaR else
                   "history contains extreme events simulations under-represent")
        _para += (
            f" Monte Carlo VaR (<strong>${mc_VaR:,.0f}</strong>) {_cmp_mc} Historical (<strong>${hist_VaR:,.0f}</strong>)"
            f" — {_why_mc}."
        )
    elif method == "All (Compare)":
        _dmc = "higher" if mc_VaR > hist_VaR else "lower"
        _wmc = ("simulated tails fatter than observed history"
                if mc_VaR > hist_VaR else "history contains extremes simulations under-represent")
        _dp  = "higher" if param_VaR > hist_VaR else "lower"
        _wp  = ("normal distribution inflates the tail"
                if param_VaR > hist_VaR else "actual returns carry fatter tails than normal assumes")
        _para += (
            f" Monte Carlo (<strong>${mc_VaR:,.0f}</strong>) is {_dmc} than"
            f" Historical (<strong>${hist_VaR:,.0f}</strong>) — {_wmc}."
            f" Parametric (<strong>${param_VaR:,.0f}</strong>) is {_dp} than Historical — {_wp}."
        )

    st.markdown(
        f'''<div style="background:#FFFFFF;border:1px solid #E0DDD6;padding:0;margin-top:2px;">

  <div style="display:flex;align-items:center;gap:10px;padding:11px 20px;
              border-bottom:1px solid #ECEAE4;background:#FAFAF8;">
    <span style="width:7px;height:7px;border-radius:50%;background:{_col};
                 display:inline-block;flex-shrink:0;"></span>
    <span style="font-family:Space Mono,monospace;font-size:0.54rem;
                 letter-spacing:0.16em;color:{_col};font-weight:700;">{_lbl}</span>
    <span style="font-family:Space Mono,monospace;font-size:0.56rem;
                 color:#BBBBBB;margin-left:auto;">
      {confidence_pct}%&nbsp;&nbsp;·&nbsp;&nbsp;{days}D&nbsp;&nbsp;·&nbsp;&nbsp;${portfolio_value:,.0f}</span>
  </div>

  <div style="padding:10px 20px 0 20px;">
    <p style="font-family:Space Mono,monospace;font-size:0.61rem;color:#999;
              line-height:1.65;margin:0;">{_note}</p>
  </div>

  <div style="padding:14px 20px 18px 20px;">
    <p style="font-family:Inter,sans-serif;font-size:0.875rem;color:#1a1a1a;
              line-height:1.9;margin:0;">{_para}</p>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);
              border-top:1px solid #ECEAE4;">
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">{days}D VaR ({confidence_pct}%)</p>
      <p style="font-family:Space Mono,monospace;font-size:1.0rem;
                font-weight:700;color:#C0392B;margin:0;">${interp_var:,.0f}</p>
    </div>
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">{days}D ES ({confidence_pct}%)</p>
      <p style="font-family:Space Mono,monospace;font-size:1.0rem;
                font-weight:700;color:#7B241C;margin:0;">${interp_es:,.0f}</p>
    </div>
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">Ann. Volatility</p>
      <p style="font-family:Space Mono,monospace;font-size:1.0rem;
                font-weight:700;color:#444;margin:0;">{annual_vol:.2f}%</p>
    </div>
    <div style="padding:12px 20px;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">Ann. VaR Equiv.</p>
      <p style="font-family:Space Mono,monospace;font-size:1.0rem;
                font-weight:700;color:#444;margin:0;">{_ann}</p>
    </div>
  </div>

</div>''',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RISK SENSITIVITY ANALYSIS & MULTI-PORTFOLIO COMPARISON ENGINE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-label">// Sensitivity Analysis</p>', unsafe_allow_html=True)
st.markdown("""
<h2 style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
           color:#1a1a1a;margin:0 0 4px 0;">RISK SENSITIVITY MATRIX</h2>
<p style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#888;margin:0 0 20px 0;
          letter-spacing:0.05em;">
  Multi-scenario stress test · 3 confidence levels · 4 horizons · 3 lookback regimes
</p>
""", unsafe_allow_html=True)

SA_CONFIDENCE = [90, 95, 99]
SA_HORIZONS   = [1, 5, 10, 20]
SA_LOOKBACKS  = [1, 3, 5]

# ── Shared compute function ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_sensitivity_matrix(tickers_tuple, weights_tuple, portfolio_value,
                                SA_CONFIDENCE, SA_HORIZONS, SA_LOOKBACKS):
    import yfinance as yf, datetime as dt
    results = {}
    max_lookback = max(SA_LOOKBACKS)
    enddate   = dt.datetime.now()
    startdate = enddate - dt.timedelta(days=365 * max_lookback + 30)
    frames = {}
    for ticker in tickers_tuple:
        data = yf.download(ticker, start=startdate, end=enddate,
                           auto_adjust=False, progress=False)
        col = data["Adj Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        frames[ticker] = col
    adj_close_df = pd.DataFrame(frames).dropna()
    log_ret_full = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    weights_arr  = np.array(weights_tuple)
    for lb in SA_LOOKBACKS:
        cutoff = enddate - dt.timedelta(days=365 * lb)
        lr     = log_ret_full[log_ret_full.index >= cutoff]
        port_r = (lr * weights_arr).sum(axis=1)
        for h in SA_HORIZONS:
            rr = port_r.rolling(window=h).sum().dropna()
            for c in SA_CONFIDENCE:
                var_v = -np.percentile(rr, 100*(1-c/100)) * portfolio_value
                losses = -rr * portfolio_value
                es_v  = float(losses[losses >= var_v].mean()) if (losses >= var_v).any() else var_v
                results[(lb, h, c)] = {"VaR": float(var_v), "ES": float(es_v)}
    return results

# ── Structural metrics per portfolio ─────────────────────────────────────────
def compute_structural_metrics(res, sa_conf, sa_hor, sa_lb):
    all_var     = [res[k]["VaR"] for k in res]
    stress_var  = res[(5, 20, 99)]["VaR"]
    base_var    = res[(1,  1, 90)]["VaR"]
    avg_var     = float(np.mean(all_var))
    std_var     = float(np.std(all_var))
    tail_amp    = res[(1, 1, 99)]["VaR"] / res[(1, 1, 90)]["VaR"] if res[(1, 1, 90)]["VaR"] > 0 else 1.0
    # Time scaling: compare 10D VaR vs sqrt(10) * 1D VaR (1Y lookback, 95%)
    var_1d      = res[(1, 1, 95)]["VaR"]
    var_10d     = res[(1, 10, 95)]["VaR"]
    expected_10 = var_1d * np.sqrt(10)
    time_dev    = abs(var_10d - expected_10) / expected_10 * 100 if expected_10 > 0 else 0.0
    # Regime sensitivity: std of VaR across 3 lookbacks at fixed (5D, 95%)
    regime_vars = [res[(lb, 5, 95)]["VaR"] for lb in sa_lb]
    regime_sens = float(np.std(regime_vars) / np.mean(regime_vars) * 100) if np.mean(regime_vars) > 0 else 0.0
    return {
        "stress_var":   stress_var,
        "base_var":     base_var,
        "avg_var":      avg_var,
        "std_var":      std_var,
        "tail_amp":     tail_amp,
        "time_dev":     time_dev,
        "regime_sens":  regime_sens,
        "stress_range": stress_var / base_var if base_var > 0 else 0,
    }

# ── Automated diagnostic commentary ──────────────────────────────────────────
def generate_diagnostics(names, metrics_list):
    lines = []
    # Tail amplification
    tamps = [(m["tail_amp"], n) for m, n in zip(metrics_list, names)]
    tamps.sort(reverse=True)
    if len(tamps) > 1 and tamps[0][0] > tamps[1][0] * 1.10:
        lines.append(f"<b>{tamps[0][1]}</b> shows significantly higher tail amplification "
                     f"({tamps[0][0]:.2f}× vs {tamps[1][0]:.2f}×) — losses worsen sharply beyond the 90th percentile.")
    # Regime sensitivity
    rsens = [(m["regime_sens"], n) for m, n in zip(metrics_list, names)]
    rsens.sort()
    if len(rsens) > 1:
        lines.append(f"<b>{rsens[0][1]}</b> is the most stable across lookback regimes "
                     f"(regime sensitivity index: {rsens[0][0]:.1f}%) — risk estimates are consistent regardless of data window used.")
        if rsens[-1][0] > rsens[0][0] * 1.5:
            lines.append(f"<b>{rsens[-1][1]}</b> is highly regime-dependent ({rsens[-1][0]:.1f}%) — "
                         f"its risk profile changes substantially across 1Y / 3Y / 5Y windows, suggesting exposure to a specific volatile period.")
    # Time scaling deviation
    tdevs = [(m["time_dev"], n) for m, n in zip(metrics_list, names)]
    tdevs.sort(reverse=True)
    if tdevs[0][0] > 15:
        lines.append(f"<b>{tdevs[0][1]}</b> deviates {tdevs[0][0]:.1f}% from √time scaling — "
                     f"10-day VaR does not scale as expected from 1-day VaR, suggesting return autocorrelation or volatility clustering.")
    else:
        lines.append(f"All portfolios scale reasonably with √time (max deviation: {tdevs[0][0]:.1f}%) — consistent with i.i.d. daily returns.")
    # Stress range
    if len(names) > 1:
        sranges = [(m["stress_range"], n) for m, n in zip(metrics_list, names)]
        sranges.sort(reverse=True)
        lines.append(f"<b>{sranges[0][1]}</b> has the widest stress range: stress VaR is "
                     f"{sranges[0][0]:.1f}× base VaR — indicating high sensitivity to both confidence level and horizon assumptions.")
    return lines


def generate_diagnostics_for_lb(names, all_sa_results_dict, lb):
    """Generate diagnostics scoped to a specific lookback window lb (1, 3, or 5)."""
    lines = []

    def _metrics_for_lb(res, lb):
        """Compute tail/time metrics using the given lookback window only."""
        var_1d  = res[(lb, 1,  95)]["VaR"]
        var_10d = res[(lb, 10, 95)]["VaR"]
        var_1d_90 = res[(lb, 1, 90)]["VaR"]
        var_1d_99 = res[(lb, 1, 99)]["VaR"]
        stress_var = res[(lb, 20, 99)]["VaR"]
        base_var   = res[(lb,  1, 90)]["VaR"]
        tail_amp = var_1d_99 / var_1d_90 if var_1d_90 > 0 else 1.0
        exp_10d  = var_1d * np.sqrt(10)
        time_dev = abs(var_10d - exp_10d) / exp_10d * 100 if exp_10d > 0 else 0.0
        stress_range = stress_var / base_var if base_var > 0 else 0
        # Regime sens for this lb: std of VaR across horizons at 95%
        var_horizons = [res[(lb, h, 95)]["VaR"] for h in [1, 5, 10, 20]]
        mn = np.mean(var_horizons)
        regime_sens = float(np.std(var_horizons) / mn * 100) if mn > 0 else 0.0
        return {
            "tail_amp":    tail_amp,
            "time_dev":    time_dev,
            "stress_range": stress_range,
            "stress_var":  stress_var,
            "base_var":    base_var,
            "regime_sens": regime_sens,
        }

    lb_metrics = {n: _metrics_for_lb(all_sa_results_dict[n], lb) for n in names}

    # Tail amplification
    tamps = sorted([(lb_metrics[n]["tail_amp"], n) for n in names], reverse=True)
    if len(tamps) > 1 and tamps[0][0] > tamps[1][0] * 1.10:
        lines.append(
            f"<b>{tamps[0][1]}</b> shows higher tail amplification over the {lb}Y window "
            f"({tamps[0][0]:.2f}× vs {tamps[1][0]:.2f}×) — losses worsen sharply beyond the 90th percentile."
        )
    elif tamps:
        lines.append(
            f"Over the {lb}Y window, tail amplification is {tamps[0][0]:.2f}× — "
            f"losses escalate meaningfully as confidence level increases."
        )

    # Time deviation
    tdevs = sorted([(lb_metrics[n]["time_dev"], n) for n in names], reverse=True)
    if tdevs[0][0] > 15:
        lines.append(
            f"<b>{tdevs[0][1]}</b> deviates {tdevs[0][0]:.1f}% from ∟time scaling in the {lb}Y window — "
            f"VaR does not scale as expected across horizons, suggesting autocorrelation or volatility clustering."
        )
    else:
        lines.append(
            f"Over the {lb}Y window, portfolios scale reasonably with ∟time "
            f"(max deviation: {tdevs[0][0]:.1f}%) — consistent with i.i.d. daily returns."
        )

    # Stress range
    sranges = sorted([(lb_metrics[n]["stress_range"], n) for n in names], reverse=True)
    lines.append(
        f"<b>{sranges[0][1]}</b> stress VaR (99%, 20D) is {sranges[0][0]:.1f}× base VaR "
        f"(90%, 1D) over the {lb}Y window — "
        + ("high sensitivity to confidence level and horizon assumptions."
           if sranges[0][0] > 5 else
           "moderate range across scenarios.")
    )

    # Regime sensitivity within this lb window (horizon spread)
    rvals = sorted([(lb_metrics[n]["regime_sens"], n) for n in names], reverse=True)
    if rvals[0][0] > 20:
        lines.append(
            f"<b>{rvals[0][1]}</b> shows high horizon-spread in the {lb}Y window "
            f"({rvals[0][0]:.1f}%) — VaR varies significantly from 1D to 20D horizons."
        )
    elif rvals[0][0] > 10:
        lines.append(
            f"Horizon spread in the {lb}Y window is moderate ({rvals[0][0]:.1f}%) — "
            f"VaR increases materially with horizon as expected."
        )
    else:
        lines.append(
            f"Horizon spread in the {lb}Y window is low ({rvals[0][0]:.1f}%) — "
            f"VaR is relatively flat across horizons, which may indicate mean-reverting returns."
        )

    return lines

# ══ PORTFOLIO DEFINITIONS ═════════════════════════════════════════════════════
st.markdown("""
<div style="background:#F9F7F3;border-left:2px solid #C0392B;padding:10px 16px;
            font-family:'Space Mono',monospace;font-size:0.68rem;color:#555;margin-bottom:18px;
            letter-spacing:0.04em;">
  <b>Portfolio A</b> uses your active sidebar portfolio.
  Define up to 2 additional portfolios below for structural comparison.
</div>
""", unsafe_allow_html=True)

# sa_portfolios already initialised and synced in sidebar block above

def parse_portfolio(p, fallback_val):
    try:
        tks  = [t.strip().upper() for t in p["tickers"].split(",") if t.strip()]
        wts  = [float(w.strip()) for w in p["weights"].split(",") if w.strip()]
        if len(tks) != len(wts) or len(tks) == 0: return None, "Ticker/weight count mismatch"
        total = sum(wts)
        if total <= 0: return None, "Weights must sum > 0"
        wts = [w/total for w in wts]
        val = p.get("value", fallback_val)
        return {"tickers": tks, "weights": wts, "value": val, "name": p["name"]}, None
    except Exception as e:
        return None, str(e)

colors_port = {
    "Portfolio A (Active)": "#C0392B",
    "S&P 500":              "#2C5F8A",
    "Nasdaq":               "#6A3FA0",
    "Dow Jones":            "#1A6B5A",
    "Bonds":                "#7A6A3A",
    "60/40":                "#5A7A3A",
    "Portfolio B":          "#2C5F8A",
    "Portfolio C":          "#5A7A3A",
}
def port_color(name):
    for k in colors_port:
        if k in name: return colors_port[k]
    return "#7A6A5A"

# ── Active benchmark summary (read-only, configured in sidebar) ──────────────
_any_active = any(p["enabled"] for p in st.session_state.sa_portfolios)
if _any_active:
    _chips = []
    for _p in st.session_state.sa_portfolios:
        if _p["enabled"]:
            _parsed, _err = parse_portfolio(_p, portfolio_value)
            if _parsed:
                _alloc = " / ".join(f"{t} {w*100:.0f}%" for t,w in zip(_parsed["tickers"], _parsed["weights"]))
                _chips.append(
                    f'<span style="display:inline-block;background:#FFFFFF;border:1px solid #D6D2CA;' +
                    f'font-family:Space Mono,monospace;font-size:0.60rem;color:#555;' +
                    f'padding:4px 10px;margin-right:8px;letter-spacing:0.06em;">' +
                    f'<b style="color:{port_color(_p["name"])}">{_p["name"]}</b>' +
                    f' &nbsp;·&nbsp; {_alloc} &nbsp;·&nbsp; ${_parsed["value"]:,.0f}</span>'
                )
    st.markdown(
        '<div style="margin-bottom:14px;">' + ''.join(_chips) + '</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#AAA;' +
        'letter-spacing:0.08em;margin-bottom:14px;">' +
        'Enable benchmarks in the sidebar under <b style="color:#555;">BENCHMARKS</b> to compare.' +
        '</div>',
        unsafe_allow_html=True)

# ── Collect active portfolios ─────────────────────────────────────────────────# ── Collect active portfolios ─────────────────────────────────────────────────
active_portfolios = [{"name": "Portfolio A (Active)", "tickers": tickers,
                       "weights": weights.tolist(), "value": portfolio_value}]
for p in st.session_state.sa_portfolios:
    if p["enabled"]:
        parsed, err = parse_portfolio(p, portfolio_value)
        if parsed:
            active_portfolios.append(parsed)

# ── Compute matrices for all active portfolios ───────────────────────────────
all_sa_results = {}
all_sa_metrics = {}

if run:
    # Full calculation triggered by user — compute and cache
    with st.spinner(f"Computing sensitivity matrix for {len(active_portfolios)} portfolio(s)..."):
        for p in active_portfolios:
            res = compute_sensitivity_matrix(
                tuple(p["tickers"]), tuple(p["weights"]), p["value"],
                SA_CONFIDENCE, SA_HORIZONS, SA_LOOKBACKS
            )
            all_sa_results[p["name"]] = res
            all_sa_metrics[p["name"]] = compute_structural_metrics(res, SA_CONFIDENCE, SA_HORIZONS, SA_LOOKBACKS)
    # Persist to session state so slider reruns don't lose the data
    st.session_state["_cached_results"] = {
        "all_sa_results": all_sa_results,
        "all_sa_metrics": all_sa_metrics,
        "active_portfolios": active_portfolios,
    }
elif _has_cached:
    # Restore from cache (triggered by slider or tab interaction)
    _cache = st.session_state["_cached_results"]
    all_sa_results   = _cache["all_sa_results"]
    all_sa_metrics   = _cache["all_sa_metrics"]
    active_portfolios = _cache["active_portfolios"]

port_names = list(all_sa_results.keys())

# ══ SUMMARY CARDS — Portfolio A ═══════════════════════════════════════════════
p_a      = all_sa_metrics[port_names[0]]
all_var  = [all_sa_results[port_names[0]][k]["VaR"] for k in all_sa_results[port_names[0]]]
all_es   = [all_sa_results[port_names[0]][k]["ES"]  for k in all_sa_results[port_names[0]]]
avg_ratio= np.mean([all_sa_results[port_names[0]][k]["ES"] / all_sa_results[port_names[0]][k]["VaR"]
                    for k in all_sa_results[port_names[0]]])

c1,c2,c3,c4 = st.columns(4)
cards = [
    (c1,"metric-var",  "Stress VaR (99%,20D,5Y)", f"${p_a['stress_var']:,.0f}", "worst-case scenario"),
    (c2,"metric-es",   "Avg ES/VaR Ratio",          f"{avg_ratio:.2f}×",         "tail severity"),
    (c3,"metric-vol",  "Tail Amplification",         f"{p_a['tail_amp']:.2f}×",   "99% vs 90% VaR"),
    (c4,"metric-loss", "Regime Sensitivity",         f"{p_a['regime_sens']:.1f}%","std / mean across lookbacks"),
]
for col,cls,lbl,val,sub in cards:
    col.markdown(f"""<div class="metric-card {cls}">
        <div class="metric-label">{lbl}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══ MAIN TABS ═════════════════════════════════════════════════════════════════
tab_labels = ["  1Y Lookback  ", "  3Y Lookback  ", "  5Y Lookback  ", "  Portfolio Comparison  "]
all_tabs   = st.tabs(tab_labels)
lb_tabs    = all_tabs[:3]
comp_tab   = all_tabs[3]

# ── Lookback matrix tabs ──────────────────────────────────────────────────────
for tab, lb in zip(lb_tabs, SA_LOOKBACKS):
    with tab:
        pname   = port_names[0]
        res     = all_sa_results[pname]
        gmin    = min(all_var); gmax = max(all_var)
        es_min  = min(all_es);  es_max = max(all_es)

        col_var, col_es = st.columns(2)

        def heatmap_table(res_dict, lb, metric, conf_list, hor_list, vmin, vmax, rgba):
            html = f'<table class="compare-table" style="font-size:0.72rem;width:100%;">'
            html += f'<thead><tr><th>↓ Horizon  /  Confidence →</th>'
            for c in conf_list: html += f'<th>{metric} {c}%</th>'
            html += '</tr></thead><tbody>'
            for h in hor_list:
                html += f'<tr><td><b>{h}D</b></td>'
                for c in conf_list:
                    v   = res_dict[(lb, h, c)][metric]
                    pct = (v - vmin)/(vmax - vmin) if vmax != vmin else 0
                    a   = 0.05 + pct * 0.22
                    html += f'<td style="background:{rgba.format(a=a)}">${v:,.0f}</td>'
                html += '</tr>'
            html += '</tbody></table>'
            return html

        with col_var:
            st.markdown(f'<p style="font-family:Space Mono,monospace;font-size:0.58rem;'
                        f'letter-spacing:0.16em;color:#888;text-transform:uppercase;'
                        f'margin:0 0 6px 0;">Value at Risk — {lb}Y window</p>', unsafe_allow_html=True)
            st.markdown(heatmap_table(res, lb, "VaR", SA_CONFIDENCE, SA_HORIZONS, gmin, gmax,
                                      "rgba(192,57,43,{a:.2f})"), unsafe_allow_html=True)
        with col_es:
            st.markdown(f'<p style="font-family:Space Mono,monospace;font-size:0.58rem;'
                        f'letter-spacing:0.16em;color:#888;text-transform:uppercase;'
                        f'margin:0 0 6px 0;">Expected Shortfall — {lb}Y window</p>', unsafe_allow_html=True)
            st.markdown(heatmap_table(res, lb, "ES", SA_CONFIDENCE, SA_HORIZONS, es_min, es_max,
                                      "rgba(123,36,28,{a:.2f})"), unsafe_allow_html=True)

        stress_var  = res[(lb, 20, 99)]["VaR"]
        base_var    = res[(lb,  1, 90)]["VaR"]
        scale_x     = stress_var / base_var if base_var > 0 else 0
        td          = p_a["time_dev"]
        ta          = p_a["tail_amp"]
        rs          = p_a["regime_sens"]
        _v1  = res[(lb,  1, 95)]["VaR"]
        _v5  = res[(lb,  5, 95)]["VaR"]
        _v10 = res[(lb, 10, 95)]["VaR"]
        _v20 = res[(lb, 20, 95)]["VaR"]
        _ta_l = ("heavy fat tails" if ta > 3 else
                 "moderate tail amplification" if ta > 2 else "mild tail amplification")
        _rs_l = "high" if rs > 20 else "moderate" if rs > 10 else "low"
        _rs_n = ("risk estimates shift materially across lookback windows"
                 if _rs_l != "low" else "risk estimates are stable across lookback windows")
        _td_n = ("suggests autocorrelation or volatility clustering"
                 if td > 20 else "broadly consistent with i.i.d. daily returns")

        tab.markdown("<br>", unsafe_allow_html=True)
        with tab.expander(
            f"📊  SENSITIVITY INTERPRETATION  —  {lb}Y LOOKBACK",
            expanded=False,
        ):
            tab.markdown(
                f'''<div style="background:#FFFFFF;border:1px solid #E0DDD6;padding:0;margin-top:2px;">

  <div style="display:flex;align-items:center;gap:10px;padding:11px 20px;
              border-bottom:1px solid #ECEAE4;background:#FAFAF8;">
    <span style="width:7px;height:7px;border-radius:50%;background:#C0392B;
                 display:inline-block;flex-shrink:0;"></span>
    <span style="font-family:Space Mono,monospace;font-size:0.54rem;
                 letter-spacing:0.16em;color:#C0392B;font-weight:700;">{lb}Y LOOKBACK WINDOW</span>
    <span style="font-family:Space Mono,monospace;font-size:0.56rem;
                 color:#BBBBBB;margin-left:auto;">{len(SA_CONFIDENCE)} confidence levels&nbsp;·&nbsp;{len(SA_HORIZONS)} horizons</span>
  </div>

  <div style="padding:16px 20px 18px 20px;">
    <p style="font-family:Inter,sans-serif;font-size:0.875rem;color:#1a1a1a;line-height:1.9;margin:0;">
      Over the <strong>{lb}Y window</strong>, worst-case stress VaR
      (99% confidence, 20-day) is <strong>${stress_var:,.0f}</strong>
      — <strong>{scale_x:.1f}×</strong> the base VaR (90%, 1-day, ${base_var:,.0f}).
      Tail amplification of <strong>{ta:.2f}×</strong> signals <em>{_ta_l}</em>:
      losses deteriorate sharply as confidence increases.
      Regime sensitivity of <strong>{rs:.1f}%</strong> is {_rs_l} — {_rs_n}.
      A √time deviation of <strong>{td:.1f}%</strong> {_td_n}.
    </p>
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #ECEAE4;">
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">Stress VaR (99%,20D)</p>
      <p style="font-family:Space Mono,monospace;font-size:0.95rem;
                font-weight:700;color:#C0392B;margin:0;">${stress_var:,.0f}</p>
    </div>
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">Stress / Base</p>
      <p style="font-family:Space Mono,monospace;font-size:0.95rem;
                font-weight:700;color:#444;margin:0;">{scale_x:.1f}×</p>
    </div>
    <div style="padding:12px 20px;border-right:1px solid #ECEAE4;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">√Time Deviation</p>
      <p style="font-family:Space Mono,monospace;font-size:0.95rem;
                font-weight:700;color:#444;margin:0;">{td:.1f}%</p>
    </div>
    <div style="padding:12px 20px;">
      <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
                color:#BBBBBB;text-transform:uppercase;margin:0 0 5px 0;">Tail Amplification</p>
      <p style="font-family:Space Mono,monospace;font-size:0.95rem;
                font-weight:700;color:#444;margin:0;">{ta:.2f}×</p>
    </div>
  </div>

  <div style="padding:12px 20px;border-top:1px solid #ECEAE4;">
    <p style="font-family:Space Mono,monospace;font-size:0.47rem;letter-spacing:0.12em;
              color:#BBBBBB;text-transform:uppercase;margin:0 0 8px 0;">
      VaR at 95% across horizons — {lb}Y window</p>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);">
      <div><p style="font-family:Space Mono,monospace;font-size:0.57rem;color:#999;margin:0 0 3px 0;">1D</p>
           <p style="font-family:Space Mono,monospace;font-size:0.82rem;
                     font-weight:700;color:#1a1a1a;margin:0;">${_v1:,.0f}</p></div>
      <div><p style="font-family:Space Mono,monospace;font-size:0.57rem;color:#999;margin:0 0 3px 0;">5D</p>
           <p style="font-family:Space Mono,monospace;font-size:0.82rem;
                     font-weight:700;color:#1a1a1a;margin:0;">${_v5:,.0f}</p></div>
      <div><p style="font-family:Space Mono,monospace;font-size:0.57rem;color:#999;margin:0 0 3px 0;">10D</p>
           <p style="font-family:Space Mono,monospace;font-size:0.82rem;
                     font-weight:700;color:#1a1a1a;margin:0;">${_v10:,.0f}</p></div>
      <div><p style="font-family:Space Mono,monospace;font-size:0.57rem;color:#999;margin:0 0 3px 0;">20D</p>
           <p style="font-family:Space Mono,monospace;font-size:0.82rem;
                     font-weight:700;color:#1a1a1a;margin:0;">${_v20:,.0f}</p></div>
    </div>
  </div>

</div>''',
                unsafe_allow_html=True,
            )

# ── Portfolio Comparison tab ──────────────────────────────────────────────────
with comp_tab:

    if len(active_portfolios) < 2:
        st.markdown(
            '''<div style="border:1px solid #D6D2CA;background:#F9F7F3;padding:32px 28px;
            font-family:Space Mono,monospace;font-size:0.70rem;color:#888;
            letter-spacing:0.08em;text-align:center;margin-top:16px;">
            Enable <b style="color:#1a1a1a;">S&P 500 (SPY)</b> or
            <b style="color:#1a1a1a;">60/40 Benchmark</b> in the expanders above
            to run benchmark comparison.
            <br><br><span style="font-size:0.60rem;color:#BBB;">
            Toggle Enable on a benchmark, then hit Calculate Risk.
            </span></div>''',
            unsafe_allow_html=True)
    else:
        # ── Helper: normalise a dollar value to % of that portfolio's value ──────
        def to_pct(val, pval): return val / pval * 100 if pval > 0 else 0

        # Build normalised metrics dict: VaR as % of portfolio value
        norm_metrics = {}
        for pname in port_names:
            m   = all_sa_metrics[pname]
            pv  = next(p["value"] for p in active_portfolios if p["name"] == pname)
            norm_metrics[pname] = {
                "stress_var_pct": to_pct(m["stress_var"], pv),
                "avg_var_pct":    to_pct(m["avg_var"],    pv),
                "base_var_pct":   to_pct(m["base_var"],   pv),
                "std_var_pct":    to_pct(m["std_var"],     pv),
                "tail_amp":       m["tail_amp"],
                "time_dev":       m["time_dev"],
                "regime_sens":    m["regime_sens"],
                "stress_range":   m["stress_range"],
                "pval":           pv,
            }

        # ── Section header ───────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:16px;
                    border-bottom:1px solid #E8E4DC;padding-bottom:12px;">
            <span style="font-family:Space Mono,monospace;font-size:0.58rem;
                         letter-spacing:0.18em;color:#999;text-transform:uppercase;">
                Metrics normalised to % of portfolio value · Portfolios are fairly comparable
            </span>
        </div>""", unsafe_allow_html=True)

        # ── Summary comparison table (normalised) ────────────────────────────────
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.58rem;' +
                    'letter-spacing:0.16em;color:#888;text-transform:uppercase;' +
                    'margin:0 0 8px 0;">Structural Risk Comparison  ·  VaR as % of portfolio value</p>',
                    unsafe_allow_html=True)

        hdr = ["Portfolio", "Value", "Stress VaR %", "Avg VaR %", "Std Dev %",
               "Tail Amp", "√Time Dev", "Regime Sens"]
        tbl  = '<table class="compare-table" style="font-size:0.72rem;width:100%;">' 
        tbl += '<thead><tr>' + ''.join(f'<th>{h}</th>' for h in hdr) + '</tr></thead><tbody>'

        for pname in port_names:
            m   = norm_metrics[pname]
            raw = all_sa_metrics[pname]
            clr = port_color(pname)

            # Delta vs Portfolio A (first row is always Portfolio A)
            if pname == port_names[0]:
                delta_stress = ""
                delta_avg    = ""
            else:
                ref_stress = norm_metrics[port_names[0]]["stress_var_pct"]
                ref_avg    = norm_metrics[port_names[0]]["avg_var_pct"]
                ds = m["stress_var_pct"] - ref_stress
                da = m["avg_var_pct"]    - ref_avg
                delta_stress = (f'<span style="color:#C0392B;font-size:0.60rem"> ↑{ds:+.2f}%</span>'
                                if ds > 0 else
                                f'<span style="color:#5A7A3A;font-size:0.60rem"> ↓{ds:+.2f}%</span>')
                delta_avg    = (f'<span style="color:#C0392B;font-size:0.60rem"> ↑{da:+.2f}%</span>'
                                if da > 0 else
                                f'<span style="color:#5A7A3A;font-size:0.60rem"> ↓{da:+.2f}%</span>')

            tbl += (f'<tr>'
                    f'<td><b style="color:{clr}">{pname}</b></td>'
                    f'<td style="color:#888">${m["pval"]:,.0f}</td>'
                    f'<td>{m["stress_var_pct"]:.2f}%{delta_stress}</td>'
                    f'<td>{m["avg_var_pct"]:.2f}%{delta_avg}</td>'
                    f'<td>{m["std_var_pct"]:.2f}%</td>'
                    f'<td>{raw["tail_amp"]:.2f}×</td>'
                    f'<td>{raw["time_dev"]:.1f}%</td>'
                    f'<td>{raw["regime_sens"]:.1f}%</td>'
                    f'</tr>')
        tbl += '</tbody></table>'
        st.markdown(tbl, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:4px;font-family:Space Mono,monospace;font-size:0.56rem;color:#BBB;'>" +
                    "↑↓ delta vs Portfolio A (Active)</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bar chart (normalised %) + Radar ─────────────────────────────────────
        col_bar, col_radar = st.columns([3, 2])

        with col_bar:
            st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.58rem;' +
                        'letter-spacing:0.16em;color:#888;text-transform:uppercase;' +
                        'margin:0 0 6px 0;">VaR as % of Portfolio Value</p>', unsafe_allow_html=True)
            bar_labels = ["Stress VaR % (99%,20D,5Y)", "Avg VaR % (36 scenarios)", "Base VaR % (90%,1D,1Y)"]
            fig_bar = go.Figure()
            for pname in port_names:
                m   = norm_metrics[pname]
                clr = port_color(pname)
                vals = [m["stress_var_pct"], m["avg_var_pct"], m["base_var_pct"]]
                fig_bar.add_trace(go.Bar(
                    name=pname, x=bar_labels, y=vals,
                    marker_color=clr, marker_line_width=0,
                    text=[f'{v:.2f}%' for v in vals],
                    textposition="outside",
                    textfont=dict(size=10, family="Space Mono, monospace"),
                ))
            fig_bar.update_layout(
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAF8",
                font=dict(family="Space Mono, monospace", size=11, color="#1a1a1a"),
                margin=dict(l=55, r=20, t=44, b=90), height=350,
                barmode="group", bargap=0.28, bargroupgap=0.05,
                xaxis=dict(showgrid=False, linecolor="#D6D2CA",
                           tickfont=dict(size=10, family="Space Mono, monospace", color="#444"),
                           tickangle=0),
                yaxis=dict(showgrid=True, gridcolor="#EEEBE4", ticksuffix="%",
                           tickfont=dict(size=10, family="Space Mono, monospace"),
                           title=dict(text="% of Portfolio Value",
                           font=dict(size=10, color="#888", family="Space Mono, monospace"))),
                legend=dict(orientation="h", y=-0.32, x=0.5, xanchor="center",
                            font=dict(size=11, family="Space Mono, monospace", color="#333"),
                            bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant"),
                title=dict(text=f"Stress / Avg / Base VaR · Normalised to % of portfolio value",
                           font=dict(size=12, color="#666", family="Space Mono, monospace"),
                           x=0, xanchor="left"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_radar:
            st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.58rem;' +
                        'letter-spacing:0.16em;color:#888;text-transform:uppercase;' +
                        'margin:0 0 6px 0;">Risk Structure Fingerprint</p>', unsafe_allow_html=True)
            radar_dims = ["Tail Amp", "Time Dev", "Regime Sens", "Stress Range", "VaR Conc."]

            # ── Absolute scale anchors (domain-knowledge floors/ceilings)
            # Each dimension is scored 0–100 on an absolute scale so that
            # no portfolio collapses to zero just because it is the minimum.
            #   tail_amp:    1.0 = no amplification (floor=1.0, ceil=5.0)
            #   time_dev:    0 % = perfect √t scaling  (floor=0,  ceil=60)
            #   regime_sens: 0 % = perfectly stable     (floor=0,  ceil=40)
            #   stress_range:1.0 = flat                 (floor=1,  ceil=15)
            #   stress_var_pct: 0 % = no risk           (floor=0,  ceil=10)
            _FLOORS  = [1.0,  0.0,  0.0,  1.0,  0.0]
            _CEILS   = [5.0, 60.0, 40.0, 15.0, 10.0]

            def _abs_score(val, floor, ceil):
                """Map val to 0-100 using absolute floor/ceil. Clamp to [0,100]."""
                if ceil == floor:
                    return 50.0
                return max(0.0, min(100.0, (val - floor) / (ceil - floor) * 100))

            fig_rad = go.Figure()
            _hover_texts = {}   # store raw values for hover

            for pname in port_names:
                m   = all_sa_metrics[pname]
                nm  = norm_metrics[pname]
                clr = port_color(pname)
                vals_raw = [
                    m["tail_amp"],
                    m["time_dev"],
                    m["regime_sens"],
                    m["stress_range"],
                    nm["stress_var_pct"],
                ]
                scores = [_abs_score(v, f, c)
                          for v, f, c in zip(vals_raw, _FLOORS, _CEILS)]
                r_ch = int(clr[1:3],16); g_ch = int(clr[3:5],16); b_ch = int(clr[5:7],16)
                # Build hover text with actual raw values
                _hover = [
                    f"Tail Amp: {vals_raw[0]:.2f}×",
                    f"Time Dev: {vals_raw[1]:.1f}%",
                    f"Regime Sens: {vals_raw[2]:.1f}%",
                    f"Stress Range: {vals_raw[3]:.1f}×",
                    f"VaR Conc.: {vals_raw[4]:.2f}%",
                ]
                fig_rad.add_trace(go.Scatterpolar(
                    r=scores+[scores[0]],
                    theta=radar_dims+[radar_dims[0]],
                    name=pname,
                    fill="toself",
                    line=dict(color=clr, width=2),
                    fillcolor=f"rgba({r_ch},{g_ch},{b_ch},0.12)",
                    opacity=0.95,
                    hovertext=_hover+[_hover[0]],
                    hovertemplate="%{hovertext}<extra>%{fullData.name}</extra>",
                ))

            fig_rad.update_layout(
                polar=dict(
                    bgcolor="#FAFAF8",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[0, 25, 50, 75, 100],
                        ticktext=["0", "25", "50", "75", "100"],
                        showticklabels=True,
                        tickfont=dict(size=9, family="Space Mono, monospace", color="#AAA"),
                        gridcolor="#E0DDD6",
                        linecolor="#CCCCCC",
                        tickangle=45,
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=11, family="Space Mono, monospace", color="#444"),
                        linecolor="#CCCCCC",
                        gridcolor="#EEEBE4",
                    ),
                ),
                paper_bgcolor="#FFFFFF",
                font=dict(family="Space Mono, monospace", size=11, color="#333"),
                margin=dict(l=50, r=50, t=60, b=80),
                height=380,
                legend=dict(
                    orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=11, family="Space Mono, monospace", color="#333"),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant",
                ),
                title=dict(
                    text="Risk Structure Fingerprint  ·  score 0–100 (higher = more risk)",
                    font=dict(size=11, color="#888", family="Space Mono, monospace"),
                    x=0.5, xanchor="center",
                ),
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # ── Side-by-side heatmaps (normalised %) ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.58rem;' +
                    'letter-spacing:0.16em;color:#888;text-transform:uppercase;' +
                    'margin:0 0 8px 0;">VaR Heatmaps  ·  % of Portfolio Value</p>',
                    unsafe_allow_html=True)

        lb_sel = st.select_slider("Lookback window",
                                  options=[1, 3, 5], value=5,
                                  format_func=lambda x: f"{x}Y",
                                  key="comp_lb_sel")

        hmap_cols = st.columns(len(port_names))

        # Global min/max in % terms for shared colour scale
        all_pct_vals = [
            all_sa_results[pn][(lb_sel, h, c)]["VaR"] /
            next(p["value"] for p in active_portfolios if p["name"] == pn) * 100
            for pn in port_names for h in SA_HORIZONS for c in SA_CONFIDENCE
        ]
        g_min_pct, g_max_pct = min(all_pct_vals), max(all_pct_vals)

        for col_h, pname in zip(hmap_cols, port_names):
            with col_h:
                clr     = port_color(pname)
                pv      = next(p["value"] for p in active_portfolios if p["name"] == pname)
                r_c     = int(clr[1:3],16); g_c = int(clr[3:5],16); b_c = int(clr[5:7],16)
                col_h.markdown(
                    f'<p style="font-family:Space Mono,monospace;font-size:0.62rem;' +
                    f'font-weight:700;color:{clr};margin:0 0 4px 0;">{pname}</p>' +
                    f'<p style="font-family:Space Mono,monospace;font-size:0.56rem;' +
                    f'color:#AAA;margin:0 0 6px 0;">${pv:,.0f} portfolio</p>',
                    unsafe_allow_html=True)
                html  = '<table class="compare-table" style="font-size:0.66rem;width:100%;">'
                html += '<thead><tr><th>H↓/C→</th>'
                for c in SA_CONFIDENCE: html += f'<th>{c}%</th>'
                html += '</tr></thead><tbody>'
                for h in SA_HORIZONS:
                    html += f'<tr><td><b>{h}D</b></td>'
                    for c in SA_CONFIDENCE:
                        v_abs = all_sa_results[pname][(lb_sel, h, c)]["VaR"]
                        v_pct = v_abs / pv * 100
                        pct_norm = (v_pct-g_min_pct)/(g_max_pct-g_min_pct) if g_max_pct != g_min_pct else 0
                        a = 0.05 + pct_norm * 0.25
                        html += (f'<td style="background:rgba({r_c},{g_c},{b_c},{a:.2f})">' +
                                 f'<span style="font-size:0.64rem">{v_pct:.2f}%</span>' +
                                 f'<br><span style="font-size:0.56rem;color:#888">${v_abs:,.0f}</span></td>')
                    html += '</tr>'
                html += '</tbody></table>'
                col_h.markdown(html, unsafe_allow_html=True)

        # ── Automated Diagnostics — lookback-aware, collapsible ─────────────────
        _diag_lines = generate_diagnostics_for_lb(port_names, all_sa_results, lb_sel)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(
            f"📋  AUTOMATED DIAGNOSTIC INSIGHTS  —  {lb_sel}Y LOOKBACK",
            expanded=False,
        ):
            st.markdown(
                f'''<div style="background:#FAFAF8;border:1px solid #E0DDD6;
                padding:4px 20px 4px 20px;">''',
                unsafe_allow_html=True,
            )
            for idx_d, dline in enumerate(_diag_lines):
                num = f"0{idx_d+1}" if idx_d < 9 else str(idx_d+1)
                st.markdown(
                    f'''<div style="display:flex;gap:16px;padding:12px 0;
                    border-bottom:1px solid #F0EDE6;align-items:flex-start;">
                    <span style="font-family:Space Mono,monospace;font-size:0.52rem;
                    color:#CCCCCC;flex-shrink:0;padding-top:3px;">{num}</span>
                    <span style="font-family:Inter,sans-serif;font-size:0.875rem;
                    color:#1a1a1a;line-height:1.75;">{dline}</span>
                    </div>''',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DRAWDOWN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-label">// Drawdown Analysis</p>', unsafe_allow_html=True)
st.markdown('''
<h2 style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
           color:#1a1a1a;margin:0 0 4px 0;">DRAWDOWN SIMULATION</h2>
<p style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#888;margin:0 0 20px 0;
          letter-spacing:0.05em;">
  Max drawdown distribution across 10,000 simulated paths · funds care more about drawdowns than VaR
</p>
''', unsafe_allow_html=True)

# ── Compute max drawdown per path from mc_paths ───────────────────────────────
# mc_paths shape: (sims, days) — each row is a price path (relative, starts at 1.0)
# Prepend 1.0 as the starting price for each path
_paths_full   = np.hstack([np.ones((mc_paths.shape[0], 1)), mc_paths])  # (sims, days+1)
_running_max  = np.maximum.accumulate(_paths_full, axis=1)              # (sims, days+1)
_drawdowns    = (_paths_full - _running_max) / _running_max             # relative drawdown (<=0)
_max_dd_rel   = _drawdowns.min(axis=1)                                  # worst DD per path (<= 0)
_max_dd_dollar= -_max_dd_rel * portfolio_value                          # in $ (positive = loss)

_pct5_dd  = float(np.percentile(_max_dd_dollar, 5))
_pct50_dd = float(np.percentile(_max_dd_dollar, 50))
_pct95_dd = float(np.percentile(_max_dd_dollar, 95))
_mean_dd  = float(_max_dd_dollar.mean())
_worst_dd = float(_max_dd_dollar.max())

# ── Summary cards ────────────────────────────────────────────────────────────
_dd_c1, _dd_c2, _dd_c3, _dd_c4 = st.columns(4)
for _col_dd, _lbl_dd, _val_dd, _sub_dd, _cls_dd in [
    (_dd_c1, "Median Max Drawdown",   f"${_pct50_dd:,.0f}", "50th percentile",      "metric-var"),
    (_dd_c2, "Expected Max Drawdown", f"${_mean_dd:,.0f}",  "mean across 10,000 paths","metric-es"),
    (_dd_c3, "95th Pctl Drawdown",    f"${_pct95_dd:,.0f}", "severe scenario",       "metric-vol"),
    (_dd_c4, "Worst Simulated DD",    f"${_worst_dd:,.0f}", "single worst path",     "metric-loss"),
]:
    with _col_dd:
        st.markdown(f'''
        <div class="metric-card {_cls_dd}">
            <div class="metric-label">{_lbl_dd}</div>
            <div class="metric-value">{_val_dd}</div>
            <div class="metric-sub">{_sub_dd}</div>
        </div>''', unsafe_allow_html=True)

# ── Drawdown distribution chart ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_dd_col_chart, _dd_col_paths = st.columns([3, 2])

with _dd_col_chart:
    st.markdown(
        '<p style="font-family:Space Mono,monospace;font-size:0.58rem;'
        'letter-spacing:0.16em;color:#888;text-transform:uppercase;margin:0 0 8px 0;">'
        'Max Drawdown Distribution  ·  10,000 paths</p>',
        unsafe_allow_html=True)
    _dd_fig = go.Figure()

    # Histogram of max drawdown in $
    _dd_fig.add_trace(go.Histogram(
        x=-_max_dd_dollar,   # show as negative (loss side)
        nbinsx=80,
        name="Max Drawdown",
        marker_color="rgba(192,57,43,0.55)",
        marker_line_color="rgba(192,57,43,0.9)",
        marker_line_width=0.4,
        hovertemplate="Drawdown: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))

    # Percentile lines
    for _pv, _pl, _pc in [
        (-_pct50_dd, "Median",  "#888888"),
        (-_pct95_dd, "95th pctl", "#C0392B"),
    ]:
        _dd_fig.add_vline(
            x=_pv, line_dash="dash", line_color=_pc, line_width=1.5,
            annotation_text=_pl,
            annotation_font=dict(size=10, family="Space Mono, monospace", color=_pc),
            annotation_position="top right",
        )

    _dd_fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAF8",
        font=dict(family="Space Mono, monospace", size=11, color="#1a1a1a"),
        margin=dict(l=55, r=20, t=20, b=55), height=300,
        showlegend=False,
        xaxis=dict(
            title=dict(text=f"Max Drawdown ($) over {days}-Day Path",
                       font=dict(size=11, color="#888", family="Space Mono, monospace")),
            showgrid=False, linecolor="#D6D2CA",
            tickformat="$,.0f",
            tickfont=dict(size=10, family="Space Mono, monospace", color="#666"),
        ),
        yaxis=dict(
            title=dict(text="Number of Paths",
                       font=dict(size=11, color="#888", family="Space Mono, monospace")),
            showgrid=True, gridcolor="#EEEBE4",
            tickfont=dict(size=10, family="Space Mono, monospace"),
        ),
    )
    st.plotly_chart(_dd_fig, use_container_width=True)

with _dd_col_paths:
    st.markdown(
        '<p style="font-family:Space Mono,monospace;font-size:0.58rem;'
        'letter-spacing:0.16em;color:#888;text-transform:uppercase;margin:0 0 8px 0;">'
        'Simulated Paths  ·  200 trajectories</p>',
        unsafe_allow_html=True)

    # ── 200 individual paths drawn as lines ───────────────────────────────────
    # Thin evenly-spaced sample from all 10,000 paths
    _sample_idx   = np.linspace(0, mc_paths.shape[0] - 1, 200, dtype=int)
    _sample_paths = mc_paths[_sample_idx]          # (200, days)
    _x_days       = np.arange(1, days + 1)
    _pnl_sample   = (_sample_paths - 1.0) * portfolio_value  # (200, days) in $

    _path_fig = go.Figure()

    # Draw loss paths (terminal value < 0) in red, gain paths in green
    # Batch by colour to keep trace count at 2 groups via scattergl for performance
    _loss_x, _loss_y = [], []
    _gain_x, _gain_y = [], []
    for _path_pnl in _pnl_sample:
        _is_loss = _path_pnl[-1] < 0
        _xs = list(_x_days) + [None]          # None creates a gap between paths
        _ys = list(_path_pnl) + [None]
        if _is_loss:
            _loss_x.extend(_xs); _loss_y.extend(_ys)
        else:
            _gain_x.extend(_xs); _gain_y.extend(_ys)

    # Loss paths — red
    if _loss_x:
        _path_fig.add_trace(go.Scatter(
            x=_loss_x, y=_loss_y,
            mode="lines",
            name="Loss paths",
            line=dict(color="rgba(192,57,43,0.28)", width=0.9),
            showlegend=True,
            hoverinfo="skip",
        ))

    # Gain paths — green
    if _gain_x:
        _path_fig.add_trace(go.Scatter(
            x=_gain_x, y=_gain_y,
            mode="lines",
            name="Gain paths",
            line=dict(color="rgba(61,122,79,0.28)", width=0.9),
            showlegend=True,
            hoverinfo="skip",
        ))

    # Median path — thick black, always on top
    _med_pnl = np.median(_pnl_sample, axis=0)
    _path_fig.add_trace(go.Scatter(
        x=_x_days, y=_med_pnl,
        mode="lines",
        name="Median",
        line=dict(color="#1a1a1a", width=2.5),
        hovertemplate="Day %{x}  |  $%{y:,.0f}<extra>Median</extra>",
    ))

    # VaR threshold line — dashed red
    _path_fig.add_hline(
        y=-mc_VaR,
        line_color="#C0392B", line_width=1.5, line_dash="dash",
        annotation_text=f"VaR ({confidence_pct}%)  −${mc_VaR:,.0f}",
        annotation_font=dict(size=9, family="Space Mono, monospace", color="#C0392B"),
        annotation_position="bottom right",
    )

    # Break-even line
    _path_fig.add_hline(
        y=0,
        line_color="#999999", line_width=1, line_dash="solid",
        annotation_text="Break-even",
        annotation_font=dict(size=9, family="Space Mono, monospace", color="#999"),
        annotation_position="top right",
    )

    _path_fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAF8",
        font=dict(family="Space Mono, monospace", size=11, color="#1a1a1a"),
        margin=dict(l=65, r=90, t=16, b=60),
        height=380,
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=10, family="Space Mono, monospace", color="#444"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant",
        ),
        xaxis=dict(
            title=dict(
                text="Trading Day",
                font=dict(size=11, color="#888", family="Space Mono, monospace"),
            ),
            showgrid=True, gridcolor="#EEEBE4",
            linecolor="#D6D2CA",
            tickmode="linear", tick0=1,
            dtick=max(1, days // 5),
            tickfont=dict(size=10, family="Space Mono, monospace", color="#555"),
        ),
        yaxis=dict(
            title=dict(
                text="Portfolio P&L ($)",
                font=dict(size=11, color="#888", family="Space Mono, monospace"),
            ),
            showgrid=True, gridcolor="#EEEBE4",
            tickformat="$,.0f",
            tickfont=dict(size=10, family="Space Mono, monospace"),
            zeroline=False,
        ),
    )
    st.plotly_chart(_path_fig, use_container_width=True)

# ── Drawdown interpretation ───────────────────────────────────────────────────
with st.expander("📉  DRAWDOWN INTERPRETATION", expanded=False):
    _dd_ratio = _pct95_dd / hist_VaR if hist_VaR > 0 else 0
    _dd_severity = ("severe" if _dd_ratio > 3 else "moderate" if _dd_ratio > 1.5 else "contained")
    st.markdown(f'''
<div style="background:#FFFFFF;border:1px solid #E0DDD6;padding:0;">
  <div style="padding:16px 20px 18px 20px;">
    <p style="font-family:Inter,sans-serif;font-size:0.875rem;color:#1a1a1a;line-height:1.9;margin:0;">
      Across 10,000 simulated <strong>{days}-day paths</strong>, the median maximum drawdown is
      <strong>${_pct50_dd:,.0f}</strong> and the 95th-percentile drawdown reaches
      <strong>${_pct95_dd:,.0f}</strong> — <strong>{_dd_ratio:.1f}×</strong> the {days}-day
      Historical VaR (${hist_VaR:,.0f}), indicating <strong>{_dd_severity}</strong> drawdown risk.
      The worst single simulated path lost <strong>${_worst_dd:,.0f}</strong>.
      Drawdown measures peak-to-trough loss within the path, capturing risk that VaR (a single-point
      loss estimate) cannot: a portfolio can stay within its VaR on any given day yet suffer
      a sustained drawdown over the holding period.
    </p>
  </div>
</div>
''', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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
