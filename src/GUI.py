"""
GUI.py — Streamlit dashboard for the Portfolio Tracker.

Run with:
    streamlit run src/GUI.py

Wires directly to Model.py for data and View.py for matplotlib charts.
View.show_* (rich terminal methods) are not used here; tables are rendered
natively via st.dataframe. All View.plot_* methods work unchanged via
st.pyplot(fig).

Controller.py is unaffected — both can run independently.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from Model import Asset, Portfolio, PortfolioAnalytics
from View  import View


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme  (IB TWS inspired) ─────────────────────────────────────────────

st.markdown("""
<style>
/* Core background */
[data-testid="stApp"]            { background: #0F1117; color: #DCE0F0; }
[data-testid="stSidebar"]        { background: #1A1D2E; border-right: 1px solid #252839; }
[data-testid="stSidebarContent"] { padding-top: 1.2rem; }

/* Headings */
h1, h2, h3 { color: #4C9BE8 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1A1D2E;
    border: 1px solid #252839;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"]  { color: #8B8FA8 !important; font-size: 0.78rem; }
[data-testid="stMetricValue"]  { color: #DCE0F0 !important; font-size: 1.25rem; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem; }

/* Dataframe */
[data-testid="stDataFrame"] { background: #1A1D2E; }

/* Expander */
[data-testid="stExpander"] details {
    background: #1A1D2E;
    border: 1px solid #252839;
    border-radius: 8px;
}

/* Select / input backgrounds */
[data-baseweb="select"] > div  { background: #1A1D2E !important; border-color: #252839 !important; }
[data-baseweb="input"]  > div  { background: #1A1D2E !important; border-color: #252839 !important; }
input, textarea { color: #DCE0F0 !important; background: #1A1D2E !important; }

/* All text inside selects/inputs → white */
[data-baseweb="select"] * { color: #DCE0F0 !important; }
[data-baseweb="select"] div[role="button"] { color: #DCE0F0 !important; }
[data-baseweb="select"] [data-testid="stSelectbox"] { color: #DCE0F0 !important; }

/* Placeholder text inside selects */
[data-baseweb="select"] [aria-selected] { color: #DCE0F0 !important; }
[data-baseweb="select"] input::placeholder { color: #8B8FA8 !important; }

/* Dropdown option list */
[role="option"] { color: #DCE0F0 !important; background: #1A1D2E !important; }
[role="listbox"] { background: #1A1D2E !important; }

/* All widget labels (selectbox, text_input, number_input, multiselect, radio…) */
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
.stRadio label, .stCheckbox label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label { color: #DCE0F0 !important; }

/* Caption text (sidebar subtitle, table captions) */
[data-testid="stCaptionContainer"] p,
.stCaption, small, caption { color: #DCE0F0 !important; }

/* Expander headers */
[data-testid="stExpander"] details > summary { color: #DCE0F0 !important; }
[data-testid="stExpander"] details > summary:hover { background: rgba(76, 155, 232, 0.1); }
[data-testid="stExpander"] details > summary span { color: #DCE0F0 !important; }

/* Multiselect tags/chips — fix clipping on first chip */
[data-baseweb="tag"] {
    background: #4C9BE8 !important;
    color: #0F1117 !important;
    max-width: none !important;
    overflow: visible !important;
}
[data-baseweb="tag"] span {
    color: #0F1117 !important;
    overflow: visible !important;
    white-space: nowrap !important;
    max-width: none !important;
    text-overflow: unset !important;
}
[data-baseweb="tag"] > div {
    overflow: visible !important;
    max-width: none !important;
}

/* Buttons */
[data-testid="stFormSubmitButton"] button,
[data-testid="stButton"] button {
    background: #4C9BE8; color: #0F1117; font-weight: 700;
    border: none; border-radius: 6px;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stButton"] button:hover { background: #3a85d0; }

/* Divider */
hr { border-color: #252839; }

/* Caption / info */
.stCaption, small { color: #DCE0F0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

CHART_TYPES = [
    "Price History",
    "Weights Pie",
    "Sharpe Bars",
    "Correlation Heatmap",
    "Optimal Weights",
    "Efficient Frontier",
    "Monte Carlo Simulation",
    "Benchmark Comparison",
    "Risk Metrics Bars",
]

# ── Session state initialisation ───────────────────────────────────────────────

if "portfolio" not in st.session_state:
    st.session_state.portfolio = None

if "analytics" not in st.session_state:
    st.session_state.analytics = None

if "view" not in st.session_state:
    st.session_state.view = View()

view: View = st.session_state.view


def _refresh() -> None:
    """Rebuild PortfolioAnalytics after any portfolio mutation."""
    st.session_state.analytics = PortfolioAnalytics(st.session_state.portfolio)


# ── Setup screen (first launch) ────────────────────────────────────────────────

if st.session_state.portfolio is None:
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("# 📈 Portfolio Tracker")
        st.markdown("---")
        st.subheader("Create your portfolio")
        with st.form("setup_form"):
            name = st.text_input("Portfolio name", "My Portfolio")
            cash = st.number_input("Starting cash balance",
                                   min_value=0.0, value=0.0, step=100.0,
                                   format="%.2f")
            if st.form_submit_button("Create Portfolio", use_container_width=True):
                if name.strip():
                    st.session_state.portfolio = Portfolio(
                        name.strip(), currency="USD", cash_balance=float(cash)
                    )
                    _refresh()
                    st.rerun()
                else:
                    st.warning("Please enter a portfolio name.")
    st.stop()


portfolio: Portfolio          = st.session_state.portfolio
analytics: PortfolioAnalytics = st.session_state.analytics
c: str = portfolio.currency


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown(f"### {portfolio.name}")
    st.caption(f"Base currency: {c}  ·  {len(portfolio.assets)} asset(s)")
    st.markdown("---")

    # ── KPI metrics ───────────────────────────────────────────────────────────
    try:
        total_invested = analytics.get_total_invested_value()
        total_current  = analytics.get_total_current_value()
        total_pnl      = analytics.get_total_profit_loss()
        pnl_pct        = (total_pnl / total_invested * 100) if total_invested else 0.0
    except Exception:
        total_invested = total_current = total_pnl = pnl_pct = 0.0

    pnl_sign = "+" if total_pnl >= 0 else ""
    st.metric("Portfolio Value",  f"{c} {total_current:,.2f}")
    st.metric("Total Invested",   f"{c} {total_invested:,.2f}")
    st.metric("Unrealised P&L",   f"{c} {total_pnl:,.2f}",
              delta=f"{pnl_sign}{pnl_pct:.2f}%")
    st.metric("Cash Balance",     f"{c} {portfolio.cash_balance:,.2f}")

    st.markdown("---")

    # ── Add Asset ─────────────────────────────────────────────────────────────
    with st.expander("➕  Add Asset", expanded=not bool(portfolio.assets)):
        with st.form("add_asset_form", clear_on_submit=True):
            ticker    = st.text_input("Ticker symbol (e.g. AAPL)").strip().upper()
            quantity  = st.number_input("Quantity", min_value=1, value=1, step=1)
            buy_price = st.number_input("Purchase price per unit",
                                        min_value=0.01, value=100.0, step=0.01,
                                        format="%.2f")
            sector    = st.text_input("Sector  (blank = auto-detect)", "")
            a_class   = st.text_input("Asset class  (blank = auto-detect)", "")
            submitted = st.form_submit_button("Add Asset", use_container_width=True)

        if submitted:
            if not ticker:
                st.warning("Please enter a ticker symbol.")
            else:
                with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
                    try:
                        asset = Asset(
                            ticker, int(quantity), float(buy_price),
                            sector      = sector.strip()  or None,
                            asset_class = a_class.strip() or None,
                        )
                        portfolio.add_asset(asset)
                        _refresh()
                        st.success(
                            f"Added {ticker}  ·  {asset.sector}  ·  {asset.asset_class}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not add {ticker}: {exc}")

    # ── Remove Asset ──────────────────────────────────────────────────────────
    with st.expander("➖  Remove Asset"):
        if portfolio.assets:
            tickers_list = [a.ticker for a in portfolio.assets]
            to_remove    = st.selectbox("Select asset", tickers_list, key="rm_sel")
            if st.button("Remove", key="rm_btn", use_container_width=True):
                portfolio.remove_asset(to_remove)
                _refresh()
                st.success(f"Removed {to_remove}")
                st.rerun()
        else:
            st.caption("No assets in portfolio.")

    # ── Manage Cash ───────────────────────────────────────────────────────────
    with st.expander("💵  Manage Cash"):
        st.caption(f"Current balance: {c} {portfolio.cash_balance:,.2f}")
        with st.form("cash_form", clear_on_submit=True):
            action = st.radio("Action", ["Deposit", "Withdraw"], horizontal=True)
            amount = st.number_input("Amount", min_value=0.01, value=1000.0,
                                     step=100.0, format="%.2f")
            if st.form_submit_button("Confirm", use_container_width=True):
                try:
                    if action == "Deposit":
                        portfolio.deposit_cash(float(amount))
                    else:
                        portfolio.withdraw_cash(float(amount))
                    _refresh()
                    st.success(
                        f"{action}ed {c} {amount:,.2f}  ·  "
                        f"New balance: {c} {portfolio.cash_balance:,.2f}"
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


# ── MAIN AREA ──────────────────────────────────────────────────────────────────

if not portfolio.assets:
    st.info("Your portfolio is empty — add assets using the sidebar to get started.")
    st.stop()


# ── Holdings table ─────────────────────────────────────────────────────────────

st.markdown(f"## {portfolio.name}")
st.markdown("---")

with st.spinner("Fetching live prices…"):
    rows = []
    for a in portfolio.assets:
        try:
            cp  = a.get_current_price()
            cv  = a.get_current_value()
            tv  = a.get_transaction_value()
            pnl = a.get_profit_loss()
            pct = (pnl / tv * 100) if tv else 0.0
            rows.append({
                "Ticker":        a.ticker,
                "Sector":        a.sector,
                "Class":         a.asset_class,
                "Qty":           a.quantity,
                "Buy Price": round(a.purchase_price, 3),
                "Current Price": round(cp, 3),
                "Invested":      round(tv, 3),
                "Value":         round(cv, 3),
                "P&L":           round(pnl, 3),
                "P&L %":         round(pct, 3),
            })
        except Exception:
            pass

if rows:
    df_pos = pd.DataFrame(rows)

    def _pnl_color(val):
        if isinstance(val, (int, float)):
            if val > 0: return "color: #2ECC71"
            if val < 0: return "color: #E74C3C"
        return ""

    styled = df_pos.style.map(_pnl_color, subset=["P&L", "P&L %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


st.markdown("---")


# ── TABLES ROW  (Weights  |  Sharpe Ratios) ────────────────────────────────────

tbl_left, tbl_right = st.columns(2)

# Weights table ────────────────────────────────────────────────────────────────
with tbl_left:
    st.subheader("Weights")
    w_mode = st.selectbox("Group by", ["Asset", "Sector", "Asset class"],
                          key="w_mode")
    if w_mode == "Asset":
        weights = analytics.get_asset_weights()
    elif w_mode == "Sector":
        weights = analytics.get_weights_by_sector()
    else:
        weights = analytics.get_weights_by_asset_class()

    df_w = pd.DataFrame([
        {
            "Name":   name,
            "Weight": f"{v * 100:.1f}%",
            "Value":  f"{c} {v * total_current:,.2f}",
        }
        for name, v in sorted(weights.items(), key=lambda x: -x[1])
    ])
    st.dataframe(df_w, use_container_width=True, hide_index=True)

# Sharpe ratios table ──────────────────────────────────────────────────────────
with tbl_right:
    st.subheader("Sharpe Ratios")
    s_mode   = st.selectbox("Group by",
                            ["Per asset", "By sector",
                             "By asset class", "Portfolio (overall)"],
                            key="s_mode")
    s_period = st.selectbox("Period", PERIODS, index=3, key="s_period")

    with st.spinner("Computing Sharpe ratios…"):
        try:
            rfr = analytics.get_risk_free_rate()
            if s_mode == "Per asset":
                s_data = analytics.get_sharpe_ratio_per_asset(period=s_period)
            elif s_mode == "By sector":
                s_data = analytics.get_sharpe_ratio_by_sector(period=s_period)
            elif s_mode == "By asset class":
                s_data = analytics.get_sharpe_ratio_by_asset_class(period=s_period)
            else:
                s_data = {
                    "Portfolio": analytics.get_portfolio_sharpe_ratio(period=s_period)
                }

            def _rating(s: float) -> str:
                if s >= 3:  return "Excellent"
                if s >= 2:  return "Very Good"
                if s >= 1:  return "Good"
                if s >= 0:  return "Subpar"
                return "Negative"

            df_s = pd.DataFrame([
                {"Name": k, "Sharpe Ratio": round(v, 3), "Rating": _rating(v)}
                for k, v in s_data.items()
                if v is not None and np.isfinite(v)
            ])
            st.dataframe(df_s, use_container_width=True, hide_index=True)
            st.caption(f"Risk-free rate: {rfr * 100:.2f}%  ·  Period: {s_period}")
        except Exception as exc:
            st.error(str(exc))


st.markdown("---")


# ── RISK METRICS TABLE  (full width) ──────────────────────────────────────────

st.subheader("Risk Metrics  —  GARCH(1,1) · Student-t · 1-Month Horizon")

rm_ctrl_left, rm_ctrl_right, _ = st.columns([1, 1, 3])
with rm_ctrl_left:
    rm_mode = st.selectbox("Group by",
                           ["Per asset", "By sector",
                            "By asset class", "Portfolio (overall)"],
                           key="rm_mode")
with rm_ctrl_right:
    rm_period = st.selectbox("Period", PERIODS, index=3, key="rm_period")

with st.spinner("Fitting GARCH(1,1) models — may take a moment…"):
    try:
        if rm_mode == "Per asset":
            rm_data = analytics.get_risk_metrics_per_asset(period=rm_period)
        elif rm_mode == "By sector":
            rm_data = analytics.get_risk_metrics_by_sector(period=rm_period)
        elif rm_mode == "By asset class":
            rm_data = analytics.get_risk_metrics_by_asset_class(period=rm_period)
        else:
            m       = analytics.get_portfolio_risk_metrics(period=rm_period)
            rm_data = {"Portfolio": m}

        rm_rows = []
        for name, m in rm_data.items():
            if not m:
                continue
            rm_rows.append({
                "Name":           name,
                "Hist Vol (mo)":  f"{m.get('hist_monthly_vol',    0) * 100:.2f}%",
                "GARCH Vol (mo)": f"{m.get('garch_predicted_vol', 0) * 100:.2f}%",
                "VaR 95%":        f"{m.get('var_95', 0) * 100:.2f}%",
                "VaR 99%":        f"{m.get('var_99', 0) * 100:.2f}%",
                "ES 95%":         f"{m.get('es_95',  0) * 100:.2f}%",
                "ES 99%":         f"{m.get('es_99',  0) * 100:.2f}%",
            })

        if rm_rows:
            st.dataframe(pd.DataFrame(rm_rows), use_container_width=True, hide_index=True)
            st.caption(
                "VaR: maximum expected loss at the given confidence level.  "
                "ES (CVaR): average loss beyond the VaR threshold.  "
                "Volatility scaled to 21 trading days via √t rule."
            )
        else:
            st.warning(
                "Not enough data for risk metrics. Try a longer period (e.g. 1y or 2y)."
            )
    except Exception as exc:
        st.error(str(exc))


st.markdown("---")


# ── CHARTS ROW ────────────────────────────────────────────────────────────────

st.subheader("Charts")

chart_left, chart_right = st.columns(2)


def _render_chart(col, key_prefix: str) -> None:
    """Render one interactive chart panel with full dropdown control."""
    with col:
        chart_type = st.selectbox("Chart type", CHART_TYPES,
                                   key=f"{key_prefix}_type")
        fig = None

        try:
            # ── Price History ──────────────────────────────────────────────────
            if chart_type == "Price History":
                tickers_all = [a.ticker for a in portfolio.assets]
                selected    = st.multiselect("Tickers", tickers_all,
                                             default=tickers_all[:1],
                                             key=f"{key_prefix}_ph_tickers")
                period      = st.selectbox("Period", PERIODS, index=3,
                                           key=f"{key_prefix}_ph_period")
                show_vol    = (
                    len(selected) == 1
                    and st.checkbox("Show volume", False, key=f"{key_prefix}_ph_vol")
                )
                if selected:
                    with st.spinner("Fetching price history…"):
                        price_data = {}
                        for t in selected:
                            asset = portfolio.get_asset(t)
                            if asset:
                                price_data[t] = asset.get_historical_prices(period=period)
                        if price_data:
                            fig = view.plot_price_history(price_data, period=period,
                                                          show_volume=show_vol)

            # ── Weights Pie ───────────────────────────────────────────────────
            elif chart_type == "Weights Pie":
                mode = st.selectbox("Group by", ["Asset", "Sector", "Asset class"],
                                    key=f"{key_prefix}_wp_mode")
                if mode == "Asset":
                    data = analytics.get_asset_weights()
                elif mode == "Sector":
                    data = analytics.get_weights_by_sector()
                else:
                    data = analytics.get_weights_by_asset_class()
                fig = view.plot_weights_pie(data, title=f"Weights by {mode}")

            # ── Sharpe Bars ───────────────────────────────────────────────────
            elif chart_type == "Sharpe Bars":
                mode   = st.selectbox("Group by",
                                      ["Per asset", "By sector", "By asset class"],
                                      key=f"{key_prefix}_sb_mode")
                period = st.selectbox("Period", PERIODS, index=3,
                                      key=f"{key_prefix}_sb_period")
                with st.spinner("Computing Sharpe ratios…"):
                    rfr = analytics.get_risk_free_rate()
                    if mode == "Per asset":
                        data  = analytics.get_sharpe_ratio_per_asset(period=period)
                        label = "Asset"
                    elif mode == "By sector":
                        data  = analytics.get_sharpe_ratio_by_sector(period=period)
                        label = "Sector"
                    else:
                        data  = analytics.get_sharpe_ratio_by_asset_class(period=period)
                        label = "Asset class"
                    fig = view.plot_sharpe_bars(
                        data, title=f"Sharpe — {label}", risk_free_rate=rfr
                    )

            # ── Correlation Heatmap ───────────────────────────────────────────
            elif chart_type == "Correlation Heatmap":
                if len(portfolio.assets) < 2:
                    st.info("Need at least 2 assets.")
                else:
                    period = st.selectbox("Period", PERIODS, index=3,
                                          key=f"{key_prefix}_ch_period")
                    with st.spinner("Computing correlation…"):
                        corr = analytics.get_correlation_matrix(period=period)
                        if not corr.empty:
                            fig = view.plot_correlation_heatmap(
                                corr, title=f"Correlation  ·  {period}"
                            )
                        else:
                            st.warning("Not enough data.")

            # ── Optimal Weights ───────────────────────────────────────────────
            elif chart_type == "Optimal Weights":
                if len(portfolio.assets) < 2:
                    st.info("Need at least 2 assets for Markowitz optimisation.")
                else:
                    period = st.selectbox("Period", PERIODS, index=3,
                                          key=f"{key_prefix}_ow_period")
                    with st.spinner("Running Markowitz optimisation…"):
                        result = analytics.get_optimal_weights(period=period)
                        if result:
                            fig = view.plot_optimal_weights_comparison(result)
                        else:
                            st.warning("Not enough data.")

            # ── Efficient Frontier ────────────────────────────────────────────
            elif chart_type == "Efficient Frontier":
                if len(portfolio.assets) < 2:
                    st.info("Need at least 2 assets.")
                else:
                    period   = st.selectbox("Period", PERIODS, index=3,
                                            key=f"{key_prefix}_ef_period")
                    n_points = st.slider("Frontier points", 20, 200, 100,
                                         key=f"{key_prefix}_ef_pts")
                    with st.spinner("Computing efficient frontier…"):
                        result = analytics.get_efficient_frontier(
                            num_points=n_points, period=period
                        )
                        opt_w  = analytics.get_optimal_weights(period=period)
                        fig = view.plot_efficient_frontier(
                            result,
                            current_return     = opt_w.get("current_return"),
                            current_volatility = opt_w.get("current_volatility"),
                        )

            # ── Monte Carlo ───────────────────────────────────────────────────
            elif chart_type == "Monte Carlo Simulation":
                years  = st.slider("Horizon (years)", 1, 30, 15,
                                   key=f"{key_prefix}_mc_years")
                n_sims = st.select_slider(
                    "Simulations",
                    options=[1_000, 5_000, 10_000, 50_000, 100_000],
                    value=10_000,
                    key=f"{key_prefix}_mc_sims",
                )
                with st.spinner(f"Running {n_sims:,} paths…"):
                    result = analytics.simulate_portfolio(
                        years=years, num_simulations=n_sims
                    )
                    fig = view.plot_simulation(
                        result,
                        currency      = portfolio.currency,
                        years         = years,
                        initial_value = analytics.get_total_current_value(),
                    )

            # ── Benchmark Comparison ──────────────────────────────────────────
            elif chart_type == "Benchmark Comparison":
                bench  = st.text_input("Benchmark ticker", "ACWI",
                                       key=f"{key_prefix}_bc_bench").strip().upper()
                period = st.selectbox("Period", PERIODS, index=3,
                                      key=f"{key_prefix}_bc_period")
                if bench:
                    with st.spinner(f"Comparing vs {bench}…"):
                        result = analytics.get_benchmark_comparison(
                            benchmark_ticker=bench, period=period
                        )
                        if result:
                            fig = view.plot_benchmark_comparison(
                                result, benchmark_label=bench
                            )
                        else:
                            st.warning(f"No data for '{bench}'. Check the ticker.")

            # ── Risk Metrics Bars ─────────────────────────────────────────────
            elif chart_type == "Risk Metrics Bars":
                mode   = st.selectbox(
                    "Group by",
                    ["Per asset", "By sector", "By asset class", "Portfolio (overall)"],
                    key=f"{key_prefix}_rm_mode",
                )
                period = st.selectbox("Period", PERIODS, index=3,
                                      key=f"{key_prefix}_rm_period")
                with st.spinner("Fitting GARCH(1,1)…"):
                    if mode == "Per asset":
                        data  = analytics.get_risk_metrics_per_asset(period=period)
                        label = "Asset"
                    elif mode == "By sector":
                        data  = analytics.get_risk_metrics_by_sector(period=period)
                        label = "Sector"
                    elif mode == "By asset class":
                        data  = analytics.get_risk_metrics_by_asset_class(period=period)
                        label = "Asset class"
                    else:
                        m     = analytics.get_portfolio_risk_metrics(period=period)
                        data  = {"Portfolio": m}
                        label = "Portfolio"
                    fig = view.plot_risk_metrics_bars(
                        data, title=f"Risk Metrics — {label}"
                    )

        except Exception as exc:
            st.error(str(exc))

        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


_render_chart(chart_left,  "cl")
_render_chart(chart_right, "cr")


