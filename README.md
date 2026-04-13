# Portfolio Tracker

A Python portfolio analytics tool with two interfaces: a terminal CLI and a Streamlit web dashboard. Built on an MVC architecture using Yahoo Finance as the market data source.

---

## Features

- **Adding and removing assets** — by giving a ticker as input, the quantity and the buying price
- **Price history chart** — multiple assets can be plotted together in one chart
- **Holdings overview** — live prices, P&L, transaction vs. current value
- **Weight breakdown** — by asset, sector, or asset class
- **Sharpe ratios** — per asset, sector, asset class, or portfolio level
- **Correlation matrix** — Pearson heatmap across all positions
- **Markowitz optimisation** — mean-variance optimal weights via SLSQP
- **Optimal weights** — bar chart and table with mean-variance optimal weights
- **Efficient frontier** — risk/return frontier coloured by Sharpe ratio
- **Monte Carlo simulation** — correlated GBM with Cholesky decomposition
- **Benchmark comparison** — alpha, tracking error, cumulative return chart
- **Risk metrics (GARCH)** — 1-month VaR and Expected Shortfall via GARCH(1,1) with Student-t innovations, per asset / sector / asset class / portfolio
- **Cash management** — deposit and withdraw cash balance

---

## Architecture

```
src/
├── Model.py       # Data layer — yfinance, calculations, analytics
├── View.py        # Presentation layer — rich tables, matplotlib figures
├── Controller.py  # CLI entry point — user input loop, wires Model + View
└── GUI.py         # Streamlit dashboard — alternative to the CLI
tests/
├── test_model.py      # Unit tests for Model.py (Asset, Portfolio, PortfolioAnalytics)
├── test_view.py       # Unit tests for View.py rendering methods
└── test_controller.py # Unit tests for Controller.py input/dispatch logic
```

**Design contract:**
- `Model.py` never imports from `View` or `Controller`
- `View.plot_*` methods return a `matplotlib.Figure` and never call `plt.show()` — the caller decides how to render it (CLI via `view.show_figure(fig)`, Streamlit via `st.pyplot(fig)`)
- `Controller.py` and `GUI.py` are fully independent — running one has no effect on the other

---

## Setup

**Requirements:** Python 3.12+

```bash
# 1. Clone the repo
git clone <repo-url>
cd Command_Line_Portfolio

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running

### CLI
```bash
python src/Controller.py
```
Interactive menu driven by numbered input. All prompts include defaults; press Enter to accept.

### Web dashboard

The dashboard is deployed and accessible at:

**https://commandlineportfolio-k7ksnyxbywvbumcj7jm2wi.streamlit.app/**


## Testing

```bash
pytest
```

The `pytest.ini` at the project root configures `pythonpath = src` so no manual path manipulation is needed.

| File | Covers |
|---|---|
| `tests/test_model.py` | `Asset`, `Portfolio`, `PortfolioAnalytics` |
| `tests/test_view.py` | `View` rendering methods |
| `tests/test_controller.py` | `Controller` input/dispatch logic |

To run a specific test file or class:
```bash
pytest tests/test_model.py::TestPortfolioAnalytics -v
```

---

## Key Implementation Notes

### Data sourcing
All market data is fetched live from Yahoo Finance via `yfinance`. Tickers follow Yahoo's convention (e.g. `AAPL`, `BTC-USD`, `IUSQ.AS`). Sector and asset class are auto-detected from Yahoo's `info` payload but can be overridden manually.

### Timezone normalisation
Yahoo Finance returns tz-aware timestamps that vary by exchange. All historical series are normalised to tz-naive date-only indexes before any `pd.concat` to prevent silent NaN propagation across assets trading on different exchanges.

### GARCH(1,1) risk metrics
Risk metrics use the `arch` library to fit a GARCH(1,1) model with Student-t innovations to each asset's daily return series. The 1-day-ahead conditional volatility forecast is scaled to a 1-month (21 trading day) horizon using the square-root-of-time rule. VaR and ES are computed parametrically from the t-distribution quantiles:

```
VaR(1-α, h) = −(μ·h + z_α · σ_h)
ES(1-α, h)  = −(μ·h + σ_h · E[Z | Z < z_α])
```

where `E[Z | Z < z_α]` uses the closed-form Student-t conditional expectation. Falls back to parametric-normal if the GARCH fit fails (e.g. too few observations).

### Markowitz optimisation
Optimal weights are found by maximising the Sharpe ratio subject to long-only constraints (weights ∈ [0,1], sum = 1) using `scipy.optimize.minimize` with the SLSQP method.

### Monte Carlo simulation
Uses correlated Geometric Brownian Motion. Asset correlations are enforced via Cholesky decomposition of the historical covariance matrix. Parameters (drift, volatility) are estimated from historical log-returns.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Market data |
| `pandas` / `numpy` | Data manipulation |
| `scipy` | Markowitz optimisation |
| `arch` | GARCH(1,1) model fitting |
| `matplotlib` | Charts and figures |
| `rich` | Terminal formatting (CLI) |
| `streamlit` | Web dashboard (GUI) |
| `pytest` | Unit testing |

---

## Notes

- A minimum of ~60 trading days of history is required for GARCH fitting. Use periods of `6mo` or longer for risk metrics.
- The efficient frontier and Markowitz optimisation require at least 2 assets.
- Risk-free rate is fetched live from the 13-week US T-Bill yield (`^IRX`) and falls back to 2% if unavailable.
- The Streamlit dashboard holds portfolio state in `st.session_state` for the duration of the browser session; it is not persisted to disk.
