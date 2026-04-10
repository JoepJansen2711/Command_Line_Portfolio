"""
View module for the CLI Portfolio Tracker.

Responsible for all output: rich terminal tables and matplotlib figures.

Design contract:
  - show_*  methods print rich-formatted output to the terminal and return
            the rich Table/Panel so a caller can inspect or re-use it.
  - plot_*  methods return a matplotlib Figure and never call plt.show().
            The caller (Controller or Streamlit) decides what to do with it:
              CLI        → view.show_figure(fig)
              Streamlit  → st.pyplot(fig)

Every method that renders a subset of data (by sector, asset class, period…)
accepts plain Python structures (dicts, DataFrames) rather than Model objects,
so the Controller can pre-filter before passing data in.  This makes every
method drop-in compatible with future Streamlit dropdown widgets.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


# ── Colour palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0F1117",
    "surface":  "#1A1D2E",
    "primary":  "#4C9BE8",
    "accent":   "#F0A500",
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral":  "#8B8FA8",
    "text":     "#DCE0F0",
    "grid":     "#252839",
}

# Distinct colours for up to 12 assets / slices
SERIES_COLORS = [
    "#4C9BE8", "#F0A500", "#2ECC71", "#E74C3C",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    "#F39C12", "#27AE60", "#E91E63", "#00BCD4",
]


# ── Shared style helper ─────────────────────────────────────────────────────────

def _style(fig: plt.Figure, axes) -> None:
    """Apply the dark theme to a Figure and one or more Axes."""
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["text"], labelsize=9)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.5, alpha=0.7)


# ── View ───────────────────────────────────────────────────────────────────────

class View:
    """
    All display logic for the Portfolio application.

    Instantiate once and reuse across the session:
        view = View()
        view.show_portfolio_summary(...)
        fig  = view.plot_price_history(...)
        view.show_figure(fig)          # CLI
        # — or —
        st.pyplot(fig)                 # Streamlit
    """

    def __init__(self):
        self.console = Console()

    # ═══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════

    def show_portfolio_summary(
        self,
        name: str,
        currency: str,
        total_invested: float,
        total_current: float,
        total_pnl: float,
        cash_balance: float,
    ) -> None:
        """
        Print a top-level portfolio summary panel.

        Args:
            name:            Portfolio name.
            currency:        Base currency string (e.g. 'EUR').
            total_invested:  Sum of all purchase costs.
            total_current:   Sum of all current market values + cash.
            total_pnl:       Unrealised profit / loss (excl. cash).
            cash_balance:    Available cash.
        """
        pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0
        sign     = "+" if total_pnl >= 0 else ""
        color    = "green" if total_pnl >= 0 else "red"

        body = (
            f"[bold]{name}[/bold]  ·  {currency}\n\n"
            f"  Invested       [bold]{currency} {total_invested:>13,.2f}[/bold]\n"
            f"  Current Value  [bold]{currency} {total_current:>13,.2f}[/bold]\n"
            f"  Cash           [bold]{currency} {cash_balance:>13,.2f}[/bold]\n"
            f"  P&L            [{color}][bold]{sign}{currency} {total_pnl:>11,.2f}"
            f"  ({sign}{pnl_pct:.2f}%)[/bold][/{color}]"
        )
        self.console.print(
            Panel(body, title="[bold cyan]Portfolio Summary[/bold cyan]",
                  border_style="cyan", expand=False)
        )

    def show_portfolio_table(
        self,
        assets_data: list[dict],
        currency: str = "EUR",
    ) -> Table:
        """
        Render a detailed holdings table and print it.

        Args:
            assets_data: List of dicts, one per asset, with keys:
                ticker, sector, asset_class, quantity, purchase_price,
                transaction_value, current_price, current_value, pnl, pnl_pct.
            currency: Label shown in monetary column headers.

        Returns:
            The rich Table object.
        """
        table = Table(
            title="Holdings",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
            border_style="cyan",
        )
        c = currency
        table.add_column("Ticker",        style="bold white", min_width=8)
        table.add_column("Sector",        style="dim white",  min_width=14)
        table.add_column("Class",         style="dim white",  min_width=8)
        table.add_column("Qty",           justify="right",    min_width=6)
        table.add_column(f"Buy ({c})",    justify="right",    min_width=12)
        table.add_column(f"Now ({c})",    justify="right",    min_width=12)
        table.add_column(f"Cost ({c})",   justify="right",    min_width=13)
        table.add_column(f"Value ({c})",  justify="right",    min_width=13)
        table.add_column("P&L",           justify="right",    min_width=18)

        for a in assets_data:
            pnl     = a["pnl"]
            pnl_pct = a.get("pnl_pct", 0.0)
            sign    = "+" if pnl >= 0 else ""
            pnl_str = f"{sign}{pnl:,.2f}  ({sign}{pnl_pct:.1f}%)"
            table.add_row(
                a["ticker"],
                a.get("sector", "—"),
                a.get("asset_class", "—"),
                str(a["quantity"]),
                f"{a['purchase_price']:,.2f}",
                f"{a['current_price']:,.2f}",
                f"{a['transaction_value']:,.2f}",
                f"{a['current_value']:,.2f}",
                Text(pnl_str, style="green" if pnl >= 0 else "red"),
            )

        self.console.print(table)
        return table

    # ═══════════════════════════════════════════════════════════════════════════
    # WEIGHTS
    # ═══════════════════════════════════════════════════════════════════════════

    def show_weights_table(
        self,
        weights: dict,
        label: str = "Asset",
        total_value: float = None,
        currency: str = "EUR",
    ) -> Table:
        """
        Print a weights breakdown table with an inline bar.

        Args:
            weights:     dict mapping name -> weight (0–1).
            label:       Header for the name column ('Asset', 'Sector', …).
            total_value: If given, adds an absolute value column.
            currency:    Used in the value column header.

        Returns:
            The rich Table object.
        """
        table = Table(
            title=f"Weights by {label}",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column(label,   style="bold white", min_width=18)
        table.add_column("Weight", justify="right",   min_width=9)
        table.add_column("",       min_width=22)
        if total_value is not None:
            table.add_column(f"Value ({currency})", justify="right", min_width=14)

        for name, w in sorted(weights.items(), key=lambda x: -x[1]):
            filled = int(w * 20)
            bar    = "[cyan]" + "█" * filled + "[/cyan]" + "░" * (20 - filled)
            row    = [name, f"{w * 100:.2f}%", bar]
            if total_value is not None:
                row.append(f"{w * total_value:,.2f}")
            table.add_row(*row)

        self.console.print(table)
        return table

    def plot_weights_pie(
        self,
        weights: dict,
        title: str = "Portfolio Weights",
        subtitle: str = None,
    ) -> plt.Figure:
        """
        Donut-style pie chart of portfolio weights.

        Args:
            weights:  dict mapping name -> weight (0–1).
            title:    Chart title.
            subtitle: Optional second line (e.g. 'Filtered by: Technology').

        Returns:
            matplotlib Figure.
        """
        labels = list(weights.keys())
        sizes  = list(weights.values())
        colors = SERIES_COLORS[: len(labels)]

        fig, ax = plt.subplots(figsize=(7, 7))
        _style(fig, ax)

        wedges, _, autotexts = ax.pie(
            sizes,
            colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90,
            wedgeprops=dict(edgecolor=PALETTE["bg"], linewidth=2, width=0.6),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_color(PALETTE["text"])
            at.set_fontsize(9)

        ax.legend(
            wedges,
            [f"{l}  {w * 100:.1f}%" for l, w in zip(labels, sizes)],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=min(3, len(labels)),
            framealpha=0,
            labelcolor=PALETTE["text"],
            fontsize=9,
        )

        full_title = title if not subtitle else f"{title}\n{subtitle}"
        ax.set_title(full_title, color=PALETTE["text"], fontsize=13,
                     fontweight="bold", pad=16)
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE HISTORY
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_price_history(
        self,
        price_data: dict[str, pd.DataFrame],
        period: str = "1y",
        show_volume: bool = False,
    ) -> plt.Figure:
        """
        Line chart of historical close prices for one or more tickers.

        Args:
            price_data:   dict mapping ticker -> DataFrame with a 'Close'
                          (and optionally 'Volume') column.
            period:       Label shown in the chart title (e.g. '1y').
            show_volume:  Add a volume sub-panel when viewing a single ticker.

        Returns:
            matplotlib Figure.
        """
        single_asset = len(price_data) == 1
        add_volume   = single_asset and show_volume

        if add_volume:
            fig, (ax, ax_vol) = plt.subplots(
                2, 1, figsize=(11, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )
            _style(fig, [ax, ax_vol])
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            _style(fig, ax)

        for i, (ticker, df) in enumerate(price_data.items()):
            color = SERIES_COLORS[i % len(SERIES_COLORS)]
            close = df["Close"]
            ax.plot(close.index, close.values, color=color, linewidth=1.8, label=ticker)
            if single_asset:
                ax.fill_between(close.index, close.values, alpha=0.12, color=color)

        ax.set_ylabel("Price", color=PALETTE["text"])
        ax.set_title(f"Price History  ·  {period}", color=PALETTE["text"],
                     fontsize=13, fontweight="bold")
        ax.legend(framealpha=0, labelcolor=PALETTE["text"])

        if add_volume:
            ticker = list(price_data.keys())[0]
            df     = list(price_data.values())[0]
            ax_vol.bar(df.index, df["Volume"], color=PALETTE["primary"],
                       alpha=0.45, width=1)
            ax_vol.set_ylabel("Volume", color=PALETTE["text"], fontsize=8)
            ax_vol.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M")
            )

        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO SIMULATION
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_simulation(
        self,
        simulation_result: dict,
        currency: str = "EUR",
        years: int = 15,
        initial_value: float = None,
    ) -> plt.Figure:
        """
        Monte Carlo fan chart showing uncertainty bands.

        Args:
            simulation_result: Output of PortfolioAnalytics.simulate_portfolio().
                               Keys: 'simulations', 'mean', 'percentile_5',
                               'percentile_95'.
            currency:          Y-axis label.
            years:             Simulation horizon (for x-axis).
            initial_value:     Starting value drawn as a dotted reference line.

        Returns:
            matplotlib Figure.
        """
        sims = simulation_result["simulations"]
        mean = simulation_result["mean"]
        p5   = simulation_result["percentile_5"]
        p95  = simulation_result["percentile_95"]
        p25  = np.percentile(sims, 25, axis=0)
        p75  = np.percentile(sims, 75, axis=0)
        x    = np.linspace(0, years, len(mean))

        fig, ax = plt.subplots(figsize=(12, 6))
        _style(fig, ax)

        # Faint individual paths
        step = max(1, len(sims) // 80)
        for path in sims[::step]:
            ax.plot(x, path, color=PALETTE["primary"], alpha=0.03, linewidth=0.4)

        ax.fill_between(x, p5, p95, alpha=0.12, color=PALETTE["primary"],
                        label="5th – 95th percentile")
        ax.fill_between(x, p25, p75, alpha=0.22, color=PALETTE["primary"],
                        label="25th – 75th percentile")
        ax.plot(x, p95,  color=PALETTE["positive"], linewidth=1.0,
                linestyle="--", alpha=0.7, label="95th pct.")
        ax.plot(x, mean, color=PALETTE["accent"],   linewidth=2.2,
                label="Mean", zorder=5)
        ax.plot(x, p5,   color=PALETTE["negative"], linewidth=1.0,
                linestyle="--", alpha=0.7, label="5th pct.")

        if initial_value is not None:
            ax.axhline(initial_value, color=PALETTE["neutral"], linewidth=1.0,
                       linestyle=":", alpha=0.6, label="Initial value")

        ax.set_xlabel("Years", color=PALETTE["text"])
        ax.set_ylabel(f"Portfolio Value ({currency})", color=PALETTE["text"])
        ax.set_title(
            f"Monte Carlo Simulation  ·  {len(sims):,} paths  ·  {years}y horizon",
            color=PALETTE["text"], fontsize=13, fontweight="bold",
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{currency} {v:,.0f}")
        )
        ax.legend(framealpha=0.1, labelcolor=PALETTE["text"], fontsize=9)
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # SHARPE RATIOS
    # ═══════════════════════════════════════════════════════════════════════════

    def show_sharpe_table(
        self,
        sharpe_data: dict,
        label: str = "Asset",
        risk_free_rate: float = None,
    ) -> Table:
        """
        Rich table of Sharpe Ratios with colour-coded interpretation.

        Args:
            sharpe_data:    dict mapping name -> Sharpe Ratio.
            label:          Row-label column header.
            risk_free_rate: Shown in the table caption when provided.

        Returns:
            The rich Table object (also printed).
        """
        def _rating(s: float) -> str:
            if s >= 3:  return "[bold green]Excellent[/bold green]"
            if s >= 2:  return "[green]Very Good[/green]"
            if s >= 1:  return "[yellow]Good[/yellow]"
            if s >= 0:  return "[orange3]Subpar[/orange3]"
            return "[red]Negative[/red]"

        caption = (
            f"Risk-free rate: {risk_free_rate * 100:.2f}%"
            if risk_free_rate is not None else None
        )
        table = Table(
            title=f"Sharpe Ratios by {label}",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
            caption=caption,
        )
        table.add_column(label,   style="bold white", min_width=18)
        table.add_column("Sharpe", justify="right",   min_width=10)
        table.add_column("",       min_width=22)
        table.add_column("Rating", min_width=14)

        for name, s in sorted(sharpe_data.items(), key=lambda x: -x[1]):
            clamped = max(0.0, min(s, 4.0))
            filled  = int(clamped / 4.0 * 20)
            color   = "green" if s >= 1 else ("yellow" if s >= 0 else "red")
            bar     = f"[{color}]" + "█" * filled + f"[/{color}]" + "░" * (20 - filled)
            table.add_row(name, f"{s:.3f}", bar, _rating(s))

        self.console.print(table)
        return table

    def plot_sharpe_bars(
        self,
        sharpe_data: dict,
        title: str = "Sharpe Ratios",
        risk_free_rate: float = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of Sharpe Ratios, colour-coded by quality.

        Args:
            sharpe_data:    dict mapping name -> Sharpe Ratio.
            title:          Chart title.
            risk_free_rate: Shown in the subtitle when provided.

        Returns:
            matplotlib Figure.
        """
        names  = list(sharpe_data.keys())
        values = list(sharpe_data.values())
        colors = [
            PALETTE["positive"] if v >= 1 else
            (PALETTE["accent"]  if v >= 0 else PALETTE["negative"])
            for v in values
        ]

        fig, ax = plt.subplots(figsize=(9, max(3.0, len(names) * 0.6 + 1.5)))
        _style(fig, ax)

        bars = ax.barh(names, values, color=colors, edgecolor=PALETTE["bg"], height=0.6)
        ax.axvline(0, color=PALETTE["neutral"], linewidth=1.0)
        ax.axvline(1, color=PALETTE["positive"], linewidth=1.0,
                   linestyle="--", alpha=0.5, label="Good  (≥ 1)")
        ax.axvline(2, color=PALETTE["positive"], linewidth=1.0,
                   linestyle=":",  alpha=0.4, label="Very Good  (≥ 2)")

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.04, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=PALETTE["text"], fontsize=9,
            )

        suffix = f"  ·  rfr = {risk_free_rate * 100:.2f}%" if risk_free_rate else ""
        ax.set_title(f"{title}{suffix}", color=PALETTE["text"],
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Sharpe Ratio", color=PALETTE["text"])
        ax.legend(framealpha=0, labelcolor=PALETTE["text"], fontsize=8)
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # CORRELATION HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix",
    ) -> plt.Figure:
        """
        Annotated heatmap using a RdBu diverging colormap (red = negative
        correlation, blue = positive).

        Args:
            corr_matrix: Square DataFrame of Pearson correlations (−1 to 1).
            title:       Chart title.

        Returns:
            matplotlib Figure.
        """
        n    = len(corr_matrix)
        size = max(5, n * 1.1)
        fig, ax = plt.subplots(figsize=(size, max(4.0, n * 0.95)))
        _style(fig, ax)

        im = ax.imshow(
            corr_matrix.values, cmap="RdBu_r",
            vmin=-1, vmax=1, aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Pearson r", color=PALETTE["text"], fontsize=9)
        cbar.ax.tick_params(colors=PALETTE["text"], labelsize=8)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right",
                           color=PALETTE["text"], fontsize=9)
        ax.set_yticklabels(corr_matrix.index, color=PALETTE["text"], fontsize=9)

        fs = 9 if n <= 8 else 7
        for i in range(n):
            for j in range(n):
                val        = corr_matrix.iloc[i, j]
                text_color = "white" if abs(val) > 0.55 else PALETTE["text"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=fs, fontweight="bold")

        ax.set_title(title, color=PALETTE["text"], fontsize=13,
                     fontweight="bold", pad=14)
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMAL WEIGHTS  (Markowitz)
    # ═══════════════════════════════════════════════════════════════════════════

    def show_optimal_weights_table(self, optimal_result: dict) -> Table:
        """
        Side-by-side table: current weight vs. Markowitz-optimal weight.
        Prints a metrics panel (return / vol / Sharpe) below the table.

        Args:
            optimal_result: Output of PortfolioAnalytics.get_optimal_weights().

        Returns:
            The rich Table object (also printed).
        """
        cur_w = optimal_result["current_weights"]
        opt_w = optimal_result["optimal_weights"]

        table = Table(
            title="Markowitz Optimal Weights",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Ticker",         style="bold white", min_width=10)
        table.add_column("Current",        justify="right",    min_width=12)
        table.add_column("Optimal",        justify="right",    min_width=12)
        table.add_column("Δ",              justify="right",    min_width=10)

        for ticker in cur_w:
            cw    = cur_w[ticker]
            ow    = opt_w.get(ticker, 0.0)
            delta = ow - cw
            sign  = "+" if delta >= 0 else ""
            style = "green" if delta > 0.005 else ("red" if delta < -0.005 else "dim")
            table.add_row(
                ticker,
                f"{cw * 100:.1f}%",
                f"{ow * 100:.1f}%",
                Text(f"{sign}{delta * 100:.1f}%", style=style),
            )

        self.console.print(table)

        metrics = (
            f"[bold]Current[/bold]   "
            f"Return: [cyan]{optimal_result['current_return'] * 100:.2f}%[/cyan]  "
            f"Vol: [cyan]{optimal_result['current_volatility'] * 100:.2f}%[/cyan]  "
            f"Sharpe: [cyan]{optimal_result['current_sharpe']:.3f}[/cyan]\n"
            f"[bold]Optimal[/bold]   "
            f"Return: [green]{optimal_result['optimal_return'] * 100:.2f}%[/green]  "
            f"Vol: [green]{optimal_result['optimal_volatility'] * 100:.2f}%[/green]  "
            f"Sharpe: [green]{optimal_result['optimal_sharpe']:.3f}[/green]"
        )
        self.console.print(
            Panel(metrics, title="[bold cyan]Risk / Return[/bold cyan]",
                  border_style="cyan", expand=False)
        )
        return table

    def plot_optimal_weights_comparison(self, optimal_result: dict) -> plt.Figure:
        """
        Grouped bar chart: current weights vs. Markowitz-optimal weights.

        Args:
            optimal_result: Output of PortfolioAnalytics.get_optimal_weights().

        Returns:
            matplotlib Figure.
        """
        tickers = list(optimal_result["current_weights"].keys())
        cur     = [optimal_result["current_weights"][t] * 100 for t in tickers]
        opt     = [optimal_result["optimal_weights"].get(t, 0) * 100 for t in tickers]

        x     = np.arange(len(tickers))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(7, len(tickers) * 1.5), 5))
        _style(fig, ax)

        ax.bar(x - width / 2, cur, width, label="Current",
               color=PALETTE["primary"],  alpha=0.85, edgecolor=PALETTE["bg"])
        ax.bar(x + width / 2, opt, width, label="Optimal",
               color=PALETTE["positive"], alpha=0.85, edgecolor=PALETTE["bg"])

        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.set_ylabel("Weight (%)", color=PALETTE["text"])
        ax.set_title(
            f"Current vs Optimal Weights  ·  "
            f"Sharpe  {optimal_result['current_sharpe']:.2f}  →  "
            f"{optimal_result['optimal_sharpe']:.2f}",
            color=PALETTE["text"], fontsize=12, fontweight="bold",
        )
        ax.legend(framealpha=0, labelcolor=PALETTE["text"])
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # EFFICIENT FRONTIER
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_efficient_frontier(
        self,
        frontier_result: dict,
        current_return: float = None,
        current_volatility: float = None,
    ) -> plt.Figure:
        """
        Scatter plot of the efficient frontier coloured by Sharpe Ratio,
        with the max-Sharpe point starred and the current portfolio marked.

        Args:
            frontier_result:     Output of PortfolioAnalytics.get_efficient_frontier().
            current_return:      Annualised return of the current portfolio.
            current_volatility:  Annualised volatility of the current portfolio.

        Returns:
            matplotlib Figure.
        """
        if not frontier_result or not frontier_result.get("returns"):
            fig, ax = plt.subplots(figsize=(8, 5))
            _style(fig, ax)
            ax.text(0.5, 0.5, "Insufficient data for frontier",
                    ha="center", va="center", transform=ax.transAxes,
                    color=PALETTE["text"], fontsize=12)
            return fig

        vols    = [v * 100 for v in frontier_result["volatilities"]]
        rets    = [r * 100 for r in frontier_result["returns"]]
        sharpes = frontier_result["sharpe_ratios"]
        opt     = frontier_result["optimal_point"]

        fig, ax = plt.subplots(figsize=(10, 6))
        _style(fig, ax)

        sc = ax.scatter(vols, rets, c=sharpes, cmap="RdYlGn",
                        s=22, zorder=3,
                        vmin=min(sharpes), vmax=max(sharpes))
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Sharpe Ratio", color=PALETTE["text"], fontsize=9)
        cbar.ax.tick_params(colors=PALETTE["text"], labelsize=8)

        # Max-Sharpe star
        ax.scatter(
            opt["volatility"] * 100, opt["return"] * 100,
            marker="*", s=320, color=PALETTE["accent"], zorder=5,
            label=f"Max Sharpe  ({opt['sharpe']:.2f})",
        )

        # Current portfolio marker
        if current_return is not None and current_volatility is not None:
            ax.scatter(
                current_volatility * 100, current_return * 100,
                marker="D", s=100, color=PALETTE["primary"], zorder=5,
                edgecolors="white", linewidths=0.8,
                label="Current portfolio",
            )

        ax.set_xlabel("Annualised Volatility (%)", color=PALETTE["text"])
        ax.set_ylabel("Annualised Return (%)",     color=PALETTE["text"])
        ax.set_title(
            "Efficient Frontier  ·  Markowitz Mean-Variance Optimisation",
            color=PALETTE["text"], fontsize=13, fontweight="bold",
        )
        ax.legend(framealpha=0.1, labelcolor=PALETTE["text"])
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # BENCHMARK COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════

    def show_benchmark_table(
        self,
        benchmark_result: dict,
        benchmark_label: str = "ACWI",
    ) -> Table:
        """
        Side-by-side metrics table: portfolio vs. benchmark.

        Args:
            benchmark_result: Output of PortfolioAnalytics.get_benchmark_comparison().
            benchmark_label:  Display name of the benchmark.

        Returns:
            The rich Table object (also printed).
        """
        p = benchmark_result["portfolio"]
        b = benchmark_result["benchmark"]

        table = Table(
            title=f"Benchmark Comparison  ·  vs {benchmark_label}",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Metric",         style="bold white", min_width=24)
        table.add_column("Portfolio",       justify="right",    min_width=14)
        table.add_column(benchmark_label,   justify="right",    min_width=14)

        rows = [
            ("Annualised Return",
             f"{p['annualized_return']    * 100:+.2f}%",
             f"{b['annualized_return']    * 100:+.2f}%"),
            ("Annualised Volatility",
             f"{p['annualized_volatility'] * 100:.2f}%",
             f"{b['annualized_volatility'] * 100:.2f}%"),
            ("Sharpe Ratio",
             f"{p['sharpe_ratio']:.3f}",
             f"{b['sharpe_ratio']:.3f}"),
            ("Alpha (vs benchmark)",
             f"{benchmark_result['alpha'] * 100:+.2f}%", "—"),
            ("Tracking Error",
             f"{benchmark_result['tracking_error'] * 100:.2f}%", "—"),
        ]
        for metric, pval, bval in rows:
            table.add_row(metric, pval, bval)

        self.console.print(table)
        return table

    def plot_benchmark_comparison(
        self,
        benchmark_result: dict,
        benchmark_label: str = "ACWI",
    ) -> plt.Figure:
        """
        Cumulative-return chart of portfolio vs. benchmark with alpha shading.

        Args:
            benchmark_result: Output of PortfolioAnalytics.get_benchmark_comparison().
            benchmark_label:  Legend label for the benchmark line.

        Returns:
            matplotlib Figure.
        """
        if not benchmark_result:
            fig, ax = plt.subplots(figsize=(10, 5))
            _style(fig, ax)
            ax.text(0.5, 0.5, "No benchmark data available",
                    ha="center", va="center", transform=ax.transAxes,
                    color=PALETTE["text"], fontsize=12)
            return fig

        port_cum  = (benchmark_result["portfolio"]["cumulative_returns"]  - 1) * 100
        bench_cum = (benchmark_result["benchmark"]["cumulative_returns"]  - 1) * 100
        p = benchmark_result["portfolio"]
        b = benchmark_result["benchmark"]

        fig, ax = plt.subplots(figsize=(11, 5))
        _style(fig, ax)

        ax.plot(port_cum.index,  port_cum.values,  color=PALETTE["primary"],
                linewidth=2.0, label="Portfolio")
        ax.plot(bench_cum.index, bench_cum.values, color=PALETTE["accent"],
                linewidth=2.0, linestyle="--", label=benchmark_label)

        ax.fill_between(
            port_cum.index, port_cum.values, bench_cum.values,
            where=(port_cum.values >= bench_cum.values),
            alpha=0.14, color=PALETTE["positive"], label="Outperforming",
        )
        ax.fill_between(
            port_cum.index, port_cum.values, bench_cum.values,
            where=(port_cum.values < bench_cum.values),
            alpha=0.14, color=PALETTE["negative"], label="Underperforming",
        )
        ax.axhline(0, color=PALETTE["neutral"], linewidth=0.8, linestyle=":")

        alpha = benchmark_result["alpha"]    * 100
        te    = benchmark_result["tracking_error"] * 100
        subtitle = (
            f"α = {'+'if alpha >= 0 else ''}{alpha:.2f}%  ·  "
            f"TE = {te:.2f}%  ·  "
            f"Portfolio Sharpe = {p['sharpe_ratio']:.2f}  ·  "
            f"{benchmark_label} Sharpe = {b['sharpe_ratio']:.2f}"
        )
        ax.set_title(
            f"Portfolio vs {benchmark_label}\n{subtitle}",
            color=PALETTE["text"], fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Cumulative Return (%)", color=PALETTE["text"])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.1f}%")
        )
        ax.legend(framealpha=0.1, labelcolor=PALETTE["text"])
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # ESG SCORES
    # ═══════════════════════════════════════════════════════════════════════════

    def show_esg_table(self, esg_data: dict) -> Table:
        """
        ESG score table per asset.

        Args:
            esg_data: Output of PortfolioAnalytics.get_esg_scores().
                      dict mapping ticker -> {totalEsg, environmentScore,
                      socialScore, governanceScore}.

        Returns:
            The rich Table object (also printed).
        """
        def _fmt(v) -> Text:
            if v in (None, "N/A"):
                return Text("N/A", style="dim")
            return Text(f"{v:.1f}")

        table = Table(
            title="ESG Scores per Asset",
            box=box.ROUNDED,
            header_style="bold cyan",
            border_style="cyan",
            caption="Source: Yahoo Finance  ·  Lower score = lower ESG risk",
        )
        table.add_column("Ticker",      style="bold white", min_width=10)
        table.add_column("Total ESG",   justify="right",    min_width=12)
        table.add_column("Environment", justify="right",    min_width=13)
        table.add_column("Social",      justify="right",    min_width=10)
        table.add_column("Governance",  justify="right",    min_width=12)

        for ticker, scores in esg_data.items():
            table.add_row(
                ticker,
                _fmt(scores.get("totalEsg")),
                _fmt(scores.get("environmentScore")),
                _fmt(scores.get("socialScore")),
                _fmt(scores.get("governanceScore")),
            )

        self.console.print(table)
        return table

    def plot_esg_scores(self, esg_data: dict) -> plt.Figure:
        """
        Grouped bar chart of ESG sub-scores per asset (inverted: lower = better).

        Args:
            esg_data: Output of PortfolioAnalytics.get_esg_scores().

        Returns:
            matplotlib Figure.
        """
        tickers, env_vals, soc_vals, gov_vals = [], [], [], []

        for ticker, scores in esg_data.items():
            e = scores.get("environmentScore")
            s = scores.get("socialScore")
            g = scores.get("governanceScore")
            if any(v not in (None, "N/A") for v in [e, s, g]):
                tickers.append(ticker)
                env_vals.append(e if e not in (None, "N/A") else 0)
                soc_vals.append(s if s not in (None, "N/A") else 0)
                gov_vals.append(g if g not in (None, "N/A") else 0)

        if not tickers:
            fig, ax = plt.subplots(figsize=(6, 4))
            _style(fig, ax)
            ax.text(0.5, 0.5, "No ESG data available for these assets",
                    ha="center", va="center", transform=ax.transAxes,
                    color=PALETTE["text"], fontsize=11)
            return fig

        x     = np.arange(len(tickers))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(tickers) * 1.5), 5))
        _style(fig, ax)

        ax.bar(x - width, env_vals, width, label="Environment",
               color="#2ECC71", alpha=0.85, edgecolor=PALETTE["bg"])
        ax.bar(x,          soc_vals, width, label="Social",
               color="#3498DB", alpha=0.85, edgecolor=PALETTE["bg"])
        ax.bar(x + width,  gov_vals, width, label="Governance",
               color="#9B59B6", alpha=0.85, edgecolor=PALETTE["bg"])

        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.set_ylabel("ESG Score", color=PALETTE["text"])
        ax.set_title("ESG Sub-scores per Asset  ·  Lower = Better ESG Risk",
                     color=PALETTE["text"], fontsize=13, fontweight="bold")
        ax.legend(framealpha=0, labelcolor=PALETTE["text"])
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # CLI HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def show_figure(self, fig: plt.Figure) -> None:
        """
        Display a matplotlib Figure in the terminal session.

        In a Streamlit context use  st.pyplot(fig)  instead of this method.
        """
        plt.figure(fig.number)
        plt.show()
        plt.close(fig)

    def show_error(self, message: str) -> None:
        """Print a formatted error panel."""
        self.console.print(
            Panel(f"[red]{message}[/red]",
                  title="[bold red]Error[/bold red]", border_style="red")
        )

    def show_success(self, message: str) -> None:
        """Print a green success line."""
        self.console.print(f"[bold green]✓[/bold green]  {message}")

    def show_info(self, message: str) -> None:
        """Print a blue info line."""
        self.console.print(f"[bold cyan]ℹ[/bold cyan]  {message}")
