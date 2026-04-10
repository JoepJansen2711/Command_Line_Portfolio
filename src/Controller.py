"""
Controller module — CLI entry point for the Portfolio Tracker.

Wires together Model and View.  Owns all user-input logic so that
neither Model nor View ever reads from stdin.

Run directly:
    python src/Controller.py
"""

import os
import sys
import time

from rich.console import Console
from rich.prompt  import Prompt, Confirm
from rich.panel   import Panel
from rich         import box
from rich.table   import Table

sys.path.insert(0, os.path.dirname(__file__))

from Model import Asset, Portfolio, PortfolioAnalytics
from View  import View


console = Console()


# ── small helpers ───────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    return Prompt.ask(f"[cyan]{prompt}[/cyan]", default=default).strip()

def _ask_float(prompt: str, default: float = None) -> float:
    while True:
        raw = _ask(prompt, str(default) if default is not None else "")
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")

def _ask_int(prompt: str, default: int = None) -> int:
    while True:
        raw = _ask(prompt, str(default) if default is not None else "")
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Please enter a whole number.[/red]")

def _pick(prompt: str, options: list[str]) -> str:
    """Display a numbered pick-list and return the chosen value."""
    for i, opt in enumerate(options, 1):
        console.print(f"  [cyan]{i}.[/cyan] {opt}")
    while True:
        raw = _ask(prompt)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print("[red]Invalid choice.[/red]")

def _spinner(message: str) -> None:
    console.print(f"[dim]{message}…[/dim]")

def _ask_positive_int(prompt: str, default: int = None) -> int:
    """Ask for a positive integer. Rejects zero and negative values."""
    while True:
        val = _ask_int(prompt, default)
        if val > 0:
            return val
        console.print("[red]Please enter a positive number.[/red]")

def _ask_positive_float(prompt: str, default: float = None) -> float:
    """Ask for a positive float. Rejects zero and negative values."""
    while True:
        val = _ask_float(prompt, default)
        if val > 0:
            return val
        console.print("[red]Please enter a positive number.[/red]")

def _ask_non_empty(prompt: str, default: str = "") -> str:
    """Ask for a non-empty string. Rejects empty input."""
    while True:
        val = _ask(prompt, default).strip()
        if val:
            return val
        console.print("[red]Please enter a valid value (cannot be empty).[/red]")


# ── Controller ──────────────────────────────────────────────────────────────────

class Controller:
    """
    Drives the interactive CLI session.

    Holds one Portfolio and one PortfolioAnalytics instance for the
    lifetime of the session.  All user I/O lives here; Model does the
    maths; View does the rendering.
    """

    PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

    def __init__(self):
        self.view      = View()
        self.portfolio : Portfolio          = None
        self.analytics : PortfolioAnalytics = None

    # ───────────────────────────────────────────────────────────────────────────
    # BOOT
    # ───────────────────────────────────────────────────────────────────────────

    def _init_portfolio(self) -> None:
        """Ask for portfolio metadata on first launch."""
        console.print(Panel(
            "[bold cyan]Welcome to Portfolio Tracker[/bold cyan]\n\n"
            "Let's set up your portfolio.",
            border_style="cyan", expand=False,
        ))
        name     = _ask("Portfolio name", "My Portfolio")
        currency = _ask("Base currency (EUR / USD / GBP …)", "EUR").upper()
        cash     = _ask_float("Starting cash balance", 0.0)

        self.portfolio = Portfolio(name, currency=currency, cash_balance=cash)
        self.analytics = PortfolioAnalytics(self.portfolio)
        self.view.show_success(f"Portfolio '{name}' created ({currency})")

    def _refresh_analytics(self) -> None:
        """Re-bind analytics after the portfolio changes."""
        self.analytics = PortfolioAnalytics(self.portfolio)

    # ───────────────────────────────────────────────────────────────────────────
    # MENU
    # ───────────────────────────────────────────────────────────────────────────

    def _print_menu(self) -> None:
        name = self.portfolio.name if self.portfolio else "—"
        n    = len(self.portfolio.assets) if self.portfolio else 0
        hdr  = f"[bold cyan]PORTFOLIO TRACKER[/bold cyan]  ·  {name}  ·  {n} asset{'s' if n != 1 else ''}"

        menu_items = [
            (" 1", "Add asset"),
            (" 2", "Remove asset"),
            (" 3", "View portfolio  (table + summary)"),
            (" 4", "Weights  (by asset / sector / class)"),
            (" 5", "Price history chart"),
            (" 6", "Sharpe ratios"),
            (" 7", "Correlation matrix  (heatmap)"),
            (" 8", "Optimal weights  (Markowitz)"),
            (" 9", "Efficient frontier"),
            ("10", "Monte Carlo simulation"),
            ("11", "Benchmark comparison"),
            ("12", "ESG scores"),
            ("13", "Manage cash"),
            (" 0", "Exit"),
        ]

        table = Table(box=box.ROUNDED, border_style="cyan",
                      show_header=False, padding=(0, 2))
        table.add_column("", style="bold cyan",  width=4)
        table.add_column("", style="white",      min_width=34)

        for num, label in menu_items:
            table.add_row(num, label)

        console.print()
        console.print(Panel(table, title=hdr, border_style="cyan", expand=False))

    # ───────────────────────────────────────────────────────────────────────────
    # ACTIONS
    # ───────────────────────────────────────────────────────────────────────────

    def _add_asset(self) -> None:
        console.print(Panel("[bold]Add Asset[/bold]", border_style="cyan", expand=False))

        ticker         = _ask_non_empty("Ticker symbol (e.g. AAPL)").upper()
        quantity       = _ask_positive_int("Quantity")
        purchase_price = _ask_positive_float("Purchase price per unit")

        # Optional overrides
        sector      = _ask("Sector  (leave blank → auto-detect from Yahoo)", "")
        asset_class = _ask("Asset class  (leave blank → auto-detect)", "")

        _spinner(f"Fetching data for {ticker} from Yahoo Finance")
        try:
            asset = Asset(
                ticker, quantity, purchase_price,
                sector      = sector      or None,
                asset_class = asset_class or None,
            )
            self.portfolio.add_asset(asset)
            self._refresh_analytics()
            self.view.show_success(
                f"Added {quantity}× {ticker}  "
                f"@ {self.portfolio.currency} {purchase_price:,.2f}  "
                f"| Sector: {asset.sector}  | Class: {asset.asset_class}"
            )
        except Exception as exc:
            self.view.show_error(f"Could not add {ticker}: {exc}")

    def _remove_asset(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("No assets in portfolio.")
            return

        tickers = [a.ticker for a in self.portfolio.assets]
        console.print("[cyan]Current assets:[/cyan] " + "  ".join(tickers))
        ticker = _ask_non_empty("Ticker to remove").upper()

        if not any(a.ticker == ticker for a in self.portfolio.assets):
            self.view.show_error(f"'{ticker}' not found in portfolio.")
            return

        if Confirm.ask(f"[yellow]Remove {ticker}?[/yellow]"):
            self.portfolio.remove_asset(ticker)
            self._refresh_analytics()
            self.view.show_success(f"Removed {ticker}.")

    def _view_portfolio(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty. Add some assets first.")
            return

        _spinner("Fetching live prices")
        currency = self.portfolio.currency

        assets_data = []
        for a in self.portfolio.assets:
            try:
                cp  = a.get_current_price()
                cv  = a.get_current_value()
                tv  = a.get_transaction_value()
                pnl = a.get_profit_loss()
                pnl_pct = (pnl / tv * 100) if tv else 0.0
                assets_data.append({
                    "ticker":            a.ticker,
                    "sector":            a.sector,
                    "asset_class":       a.asset_class,
                    "quantity":          a.quantity,
                    "purchase_price":    a.purchase_price,
                    "current_price":     cp,
                    "transaction_value": tv,
                    "current_value":     cv,
                    "pnl":               pnl,
                    "pnl_pct":           pnl_pct,
                })
            except Exception as exc:
                self.view.show_error(f"Error fetching {a.ticker}: {exc}")

        self.view.show_portfolio_table(assets_data, currency=currency)
        self.view.show_portfolio_summary(
            name           = self.portfolio.name,
            currency       = currency,
            total_invested = self.analytics.get_total_invested_value(),
            total_current  = self.analytics.get_total_current_value(),
            total_pnl      = self.analytics.get_total_profit_loss(),
            cash_balance   = self.portfolio.cash_balance,
        )

    def _view_weights(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        console.print("\n[cyan]View weights by:[/cyan]")
        mode = _pick("Choose", ["Asset", "Sector", "Asset class"])

        _spinner("Calculating weights")
        total = self.analytics.get_total_current_value()
        currency = self.portfolio.currency

        if mode == "Asset":
            weights = self.analytics.get_asset_weights()
            label   = "Asset"
        elif mode == "Sector":
            weights = self.analytics.get_weights_by_sector()
            label   = "Sector"
        else:
            weights = self.analytics.get_weights_by_asset_class()
            label   = "Asset class"

        self.view.show_weights_table(weights, label=label,
                                     total_value=total, currency=currency)

        if Confirm.ask("[cyan]Show pie chart?[/cyan]", default=True):
            fig = self.view.plot_weights_pie(
                weights,
                title=f"Weights by {label}",
            )
            self.view.show_figure(fig)

    def _price_history(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        tickers_all = [a.ticker for a in self.portfolio.assets]

        console.print("\n[cyan]Select tickers to plot:[/cyan]")
        console.print("  [dim]Enter comma-separated tickers, or press Enter for all[/dim]")
        raw = _ask("Tickers", ", ".join(tickers_all))
        selected = [t.strip().upper() for t in raw.split(",") if t.strip()]
        selected = [t for t in selected if t in tickers_all] or tickers_all

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        show_vol = len(selected) == 1 and Confirm.ask(
            "[cyan]Show volume sub-chart?[/cyan]", default=False
        )

        _spinner(f"Fetching {period} history for {', '.join(selected)}")
        price_data = {}
        for ticker in selected:
            asset = self.portfolio.get_asset(ticker)
            try:
                price_data[ticker] = asset.get_historical_prices(period=period)
            except Exception as exc:
                self.view.show_error(f"Could not fetch {ticker}: {exc}")

        if price_data:
            fig = self.view.plot_price_history(price_data, period=period,
                                               show_volume=show_vol)
            self.view.show_figure(fig)

    def _sharpe_ratios(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        console.print("\n[cyan]View Sharpe Ratios by:[/cyan]")
        mode = _pick("Choose", ["Per asset", "By sector", "By asset class",
                                "Portfolio (overall)"])

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        rfr = _ask_float("Risk-free rate % (e.g. 4.5)  — leave 0 to auto-fetch", 0.0)
        rfr_decimal = rfr / 100 if rfr else None

        _spinner("Calculating Sharpe Ratios")

        if mode == "Portfolio (overall)":
            sharpe = self.analytics.get_portfolio_sharpe_ratio(
                risk_free_rate=rfr_decimal, period=period
            )
            rfr_used = rfr_decimal if rfr_decimal is not None else self.analytics.get_risk_free_rate()
            self.view.show_sharpe_table(
                {"Portfolio": sharpe},
                label="Portfolio",
                risk_free_rate=rfr_used,
            )
            return

        if mode == "Per asset":
            data  = self.analytics.get_sharpe_ratio_per_asset(
                        risk_free_rate=rfr_decimal, period=period)
            label = "Asset"
        elif mode == "By sector":
            data  = self.analytics.get_sharpe_ratio_by_sector(
                        risk_free_rate=rfr_decimal, period=period)
            label = "Sector"
        else:
            data  = self.analytics.get_sharpe_ratio_by_asset_class(
                        risk_free_rate=rfr_decimal, period=period)
            label = "Asset class"

        rfr_used = rfr_decimal if rfr_decimal is not None else self.analytics.get_risk_free_rate()
        self.view.show_sharpe_table(data, label=label, risk_free_rate=rfr_used)

        if Confirm.ask("[cyan]Show bar chart?[/cyan]", default=True):
            fig = self.view.plot_sharpe_bars(
                data, title=f"Sharpe Ratios — {label}", risk_free_rate=rfr_used
            )
            self.view.show_figure(fig)

    def _correlation_matrix(self) -> None:
        if len(self.portfolio.assets) < 2:
            self.view.show_info("Need at least 2 assets for a correlation matrix.")
            return

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        _spinner("Computing correlation matrix")
        try:
            corr = self.analytics.get_correlation_matrix(period=period)
            fig  = self.view.plot_correlation_heatmap(
                corr, title=f"Correlation Matrix  ·  {period}"
            )
            self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Could not compute correlation matrix: {exc}")

    def _optimal_weights(self) -> None:
        if len(self.portfolio.assets) < 2:
            self.view.show_info("Need at least 2 assets for Markowitz optimisation.")
            return

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        rfr = _ask_float("Risk-free rate % (e.g. 4.5)  — leave 0 to auto-fetch", 0.0)
        rfr_decimal = rfr / 100 if rfr else None

        _spinner("Running Markowitz optimisation")
        try:
            result = self.analytics.get_optimal_weights(
                risk_free_rate=rfr_decimal, period=period
            )
            self.view.show_optimal_weights_table(result)

            if Confirm.ask("[cyan]Show comparison chart?[/cyan]", default=True):
                fig = self.view.plot_optimal_weights_comparison(result)
                self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Optimisation failed: {exc}")

    def _efficient_frontier(self) -> None:
        if len(self.portfolio.assets) < 2:
            self.view.show_info("Need at least 2 assets for the efficient frontier.")
            return

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        n_points = _ask_positive_int("Number of frontier points (e.g. 100)", 100)

        rfr = _ask_float("Risk-free rate % (e.g. 4.5)  — leave 0 to auto-fetch", 0.0)
        rfr_decimal = rfr / 100 if rfr else None

        _spinner("Computing efficient frontier")
        try:
            result = self.analytics.get_efficient_frontier(
                num_points=n_points, risk_free_rate=rfr_decimal, period=period
            )
            opt_w  = self.analytics.get_optimal_weights(
                risk_free_rate=rfr_decimal, period=period
            )
            fig = self.view.plot_efficient_frontier(
                result,
                current_return     = opt_w.get("current_return"),
                current_volatility = opt_w.get("current_volatility"),
            )
            self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Could not compute frontier: {exc}")

    def _monte_carlo(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        years      = _ask_positive_int("Simulation horizon (years)", 15)
        n_paths    = _ask_positive_int("Number of simulated paths (e.g. 10000)", 10000)
        console.print(
            "[dim]  Tip: use 10,000 for a quick preview; "
            "100,000 for high-quality results (slower).[/dim]"
        )

        _spinner(f"Running Monte Carlo  —  {n_paths:,} paths  ·  {years}y")
        try:
            result = self.analytics.simulate_portfolio(
                years=years, num_simulations=n_paths
            )
            fig = self.view.plot_simulation(
                result,
                currency      = self.portfolio.currency,
                years         = years,
                initial_value = self.analytics.get_total_current_value(),
            )
            self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Simulation failed: {exc}")

    def _benchmark_comparison(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        console.print("\n[cyan]Benchmark ticker[/cyan]")
        console.print("  [dim]Common choices: ACWI (MSCI All-World), SPY (S&P 500),"
                      " IUSQ (MSCI World EUR)[/dim]")
        benchmark = _ask_non_empty("Benchmark ticker", "ACWI").upper()

        console.print("\n[cyan]Period:[/cyan]")
        period = _pick("Choose", self.PERIODS)

        rfr = _ask_float("Risk-free rate % (e.g. 4.5)  — leave 0 to auto-fetch", 0.0)
        rfr_decimal = rfr / 100 if rfr else None

        _spinner(f"Comparing portfolio vs {benchmark}")
        try:
            result = self.analytics.get_benchmark_comparison(
                benchmark_ticker=benchmark,
                period=period,
                risk_free_rate=rfr_decimal,
            )
            if not result:
                self.view.show_error(
                    f"No data returned for '{benchmark}'. "
                    "Check the ticker and try again."
                )
                return

            self.view.show_benchmark_table(result, benchmark_label=benchmark)

            if Confirm.ask("[cyan]Show cumulative-return chart?[/cyan]", default=True):
                fig = self.view.plot_benchmark_comparison(result,
                                                          benchmark_label=benchmark)
                self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Benchmark comparison failed: {exc}")

    def _esg_scores(self) -> None:
        if not self.portfolio.assets:
            self.view.show_info("Portfolio is empty.")
            return

        _spinner("Fetching ESG scores from Yahoo Finance")
        try:
            esg = self.analytics.get_esg_scores()
            self.view.show_esg_table(esg)

            if Confirm.ask("[cyan]Show ESG bar chart?[/cyan]", default=True):
                fig = self.view.plot_esg_scores(esg)
                self.view.show_figure(fig)
        except Exception as exc:
            self.view.show_error(f"Could not fetch ESG data: {exc}")

    def _manage_cash(self) -> None:
        cur = self.portfolio.cash_balance
        c   = self.portfolio.currency
        console.print(
            f"\n  Current cash balance: [bold cyan]{c} {cur:,.2f}[/bold cyan]"
        )

        console.print("\n[cyan]Action:[/cyan]")
        action = _pick("Choose", ["Deposit", "Withdraw"])

        amount = _ask_positive_float("Amount")

        try:
            if action == "Deposit":
                self.portfolio.deposit_cash(amount)
                self.view.show_success(
                    f"Deposited {c} {amount:,.2f}  ·  "
                    f"New balance: {c} {self.portfolio.cash_balance:,.2f}"
                )
            else:
                self.portfolio.withdraw_cash(amount)
                self.view.show_success(
                    f"Withdrew {c} {amount:,.2f}  ·  "
                    f"New balance: {c} {self.portfolio.cash_balance:,.2f}"
                )
        except ValueError as exc:
            self.view.show_error(str(exc))

    # ───────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ───────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive CLI session."""
        try:
            self._init_portfolio()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            return

        dispatch = {
            "1":  self._add_asset,
            "2":  self._remove_asset,
            "3":  self._view_portfolio,
            "4":  self._view_weights,
            "5":  self._price_history,
            "6":  self._sharpe_ratios,
            "7":  self._correlation_matrix,
            "8":  self._optimal_weights,
            "9":  self._efficient_frontier,
            "10": self._monte_carlo,
            "11": self._benchmark_comparison,
            "12": self._esg_scores,
            "13": self._manage_cash,
        }

        while True:
            try:
                self._print_menu()
                choice = _ask("Your choice").strip()

                if choice == "0":
                    console.print(
                        Panel("[bold cyan]Goodbye. Happy investing![/bold cyan]",
                              border_style="cyan", expand=False)
                    )
                    break

                action = dispatch.get(choice)
                if action is None:
                    self.view.show_error("Invalid choice — enter a number from the menu.")
                    continue

                action()

            except KeyboardInterrupt:
                console.print("\n[dim]Use 0 to exit.[/dim]")
            except Exception as exc:
                self.view.show_error(f"Unexpected error: {exc}")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Controller().run()
