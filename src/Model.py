import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize



# ASSET

class Asset:
    """
    Represents a single financial asset in the portfolio.

    Stores the ticker symbol, quantity, and purchase price. Connects to
    Yahoo Finance to fetch live market data, historical prices, and
    asset metadata (sector, asset class) when not provided manually.
    """

    def __init__(self, ticker: str, quantity: int, purchase_price: float,
                 sector: str = None, asset_class: str = None):
        """
        Initialize an Asset.

        Args:
            ticker: The stock/ETF/crypto ticker symbol (e.g. 'AAPL').
            quantity: Number of units held.
            purchase_price: Price per unit at time of purchase.
            sector: Optional. If not provided, fetched from Yahoo Finance.
            asset_class: Optional. If not provided, fetched from Yahoo Finance
                         (e.g. 'EQUITY', 'ETF', 'CRYPTOCURRENCY').
        """
        self.ticker = ticker
        self.quantity = quantity
        self.purchase_price = purchase_price
        self._yf_ticker = yf.Ticker(ticker)

        if sector is None or asset_class is None:
            info = self._yf_ticker.info
            self.sector = sector or info.get("sector", "Unknown")
            self.asset_class = asset_class or info.get("quoteType", "Unknown")
        else:
            self.sector = sector
            self.asset_class = asset_class

    def get_current_price(self) -> float:
        """
        Fetch the current market price from Yahoo Finance.

        Returns:
            The current price as a float. Falls back to regularMarketPrice
            if currentPrice is unavailable. Returns 0.0 if neither exists.
        """
        info = self._yf_ticker.info
        return info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

    def get_historical_prices(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data from Yahoo Finance.

        Args:
            period: How far back to look (e.g. '1d', '5d', '1mo', '1y', '5y', 'max').
            interval: Data granularity (e.g. '1d', '1wk', '1mo').

        Returns:
            A pandas DataFrame with columns: Open, High, Low, Close,
            Volume, Dividends, Stock Splits. Indexed by Date.
        """
        return self._yf_ticker.history(period=period, interval=interval)

    def get_transaction_value(self) -> float:
        """
        Calculate the original cost of this position.

        Returns:
            quantity * purchase_price
        """
        return self.quantity * self.purchase_price

    def get_current_value(self) -> float:
        """
        Calculate the current market value of this position.

        Returns:
            quantity * current market price
        """
        return self.quantity * self.get_current_price()

    def get_profit_loss(self) -> float:
        """
        Calculate the unrealized profit or loss.

        Returns:
            current_value - transaction_value (positive = profit, negative = loss)
        """
        return self.get_current_value() - self.get_transaction_value()
    



    def get_daily_returns(self, period: str = "1y") -> pd.Series:
        """
        Calculate daily percentage returns from historical close prices.

        Args:
            period: How far back to look (e.g. '1y', '5y').

        Returns:
            A pandas Series of daily returns, indexed by date.
        """
        hist = self.get_historical_prices(period=period)
        return hist["Close"].pct_change().dropna()


    def get_annualized_return(self, period: str = "1y") -> float:
        """
        Calculate the annualized return based on historical daily returns.

        Args:
            period: How far back to look (e.g. '1y', '5y').

        Returns:
            Annualized return as a float (e.g. 0.12 for 12%).
        """
        daily_returns = self.get_daily_returns(period=period)
        return daily_returns.mean() * 252


    def get_annualized_volatility(self, period: str = "1y") -> float:
        """
        Calculate the annualized volatility (standard deviation of returns).

        Args:
            period: How far back to look (e.g. '1y', '5y').

        Returns:
            Annualized volatility as a float.
        """
        daily_returns = self.get_daily_returns(period=period)
        return daily_returns.std() * np.sqrt(252)


    def get_sharpe_ratio(self, risk_free_rate: float = None, period: str = "1y") -> float:
        """
        Calculate the Sharpe Ratio for this individual asset.

        Sharpe Ratio = (annualized return - risk free rate) / annualized volatility.
        A higher Sharpe Ratio means better risk-adjusted returns.
        Interpretation: < 1 = subpar, 1-2 = good, 2-3 = very good, > 3 = excellent.

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX
                            (13-week US Treasury Bill yield).
            period: How far back to look (e.g. '1y', '5y').

        Returns:
            The Sharpe Ratio as a float.
        """
        if risk_free_rate is None:
            risk_free_rate = self._fetch_risk_free_rate()

        ann_return = self.get_annualized_return(period=period)
        ann_vol = self.get_annualized_volatility(period=period)

        if ann_vol == 0:
            return 0.0

        return (ann_return - risk_free_rate) / ann_vol


    def get_esg_score(self) -> dict:
        """
        Fetch ESG (Environmental, Social, Governance) scores from Yahoo Finance.

        Returns:
            A dict with keys 'totalEsg', 'environmentScore', 'socialScore',
            'governanceScore'. Values are floats or 'N/A' if unavailable.
        """
        try:
            info = self._yf_ticker.info
            return {
                "totalEsg": info.get("totalEsg", "N/A"),
                "environmentScore": info.get("environmentScore", "N/A"),
                "socialScore": info.get("socialScore", "N/A"),
                "governanceScore": info.get("governanceScore", "N/A"),
            }
        except Exception:
            return {
                "totalEsg": "N/A",
                "environmentScore": "N/A",
                "socialScore": "N/A",
                "governanceScore": "N/A",
            }


    @staticmethod
    def _fetch_risk_free_rate() -> float:
        """
        Fetch the current risk-free rate from ^IRX (13-week US Treasury Bill).

        ^IRX quotes the yield as a percentage (e.g. 4.5 means 4.5%), so we
        divide by 100 to get a decimal (0.045).

        Returns:
            The annualized risk-free rate as a decimal. Falls back to 0.02 if unavailable.
        """
        try:
            irx = yf.Ticker("^IRX")
            hist = irx.history(period="5d")
            if not hist.empty:
                return hist["Close"].iloc[-1] / 100
        except Exception:
            pass
        return 0.02


    def __repr__(self) -> str:
        """Return a readable string representation of the Asset."""
        return (f"Asset(ticker='{self.ticker}', sector='{self.sector}', "
                f"class='{self.asset_class}', qty={self.quantity}, "
                f"buy_price={self.purchase_price})")
    











# PORTFOLIO

class Portfolio:
    """
    A container for financial assets and a cash balance.

    Handles adding, removing, and looking up assets. Stores the
    portfolio's name, base currency, and available cash. Does not
    perform calculations — that responsibility belongs to PortfolioAnalytics.
    """

    def __init__(self, name: str, currency: str = "USD", cash_balance: float = 0.0):
        """
        Initialize a Portfolio.

        Args:
            name: A label for this portfolio (e.g. 'My Retirement Fund').
            currency: The base currency for cash (e.g. 'EUR', 'USD').
            cash_balance: Starting cash amount. Defaults to 0.
        """
        self.name = name
        self.currency = currency
        self.cash_balance = cash_balance
        self.assets: list[Asset] = []

    def add_asset(self, asset: Asset):
        """
        Add an asset to the portfolio.

        Args:
            asset: An Asset instance to add.
        """
        self.assets.append(asset)

    def remove_asset(self, ticker: str):
        """
        Remove an asset from the portfolio by its ticker symbol.

        Args:
            ticker: The ticker symbol to remove (e.g. 'AAPL').
                    If not found, the portfolio remains unchanged.
        """
        self.assets = [a for a in self.assets if a.ticker != ticker]

    def get_asset(self, ticker: str) -> Asset:
        """
        Look up a specific asset by ticker symbol.

        Args:
            ticker: The ticker symbol to search for.

        Returns:
            The matching Asset instance, or None if not found.
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
            
        return None

    def deposit_cash(self, amount: float):
        """
        Add cash to the portfolio.

        Args:
            amount: The amount to deposit. Must be positive.
        """
        self.cash_balance += amount

    def withdraw_cash(self, amount: float):
        """
        Remove cash from the portfolio.

        Args:
            amount: The amount to withdraw.

        Raises:
            ValueError: If the withdrawal exceeds the available cash balance.
        """

        if amount > self.cash_balance:
            raise ValueError("Insufficient cash balance.")
        
        self.cash_balance -= amount

    def __repr__(self) -> str:
        """Return a readable string representation of the Portfolio."""
        return (f"Portfolio(name='{self.name}', currency='{self.currency}', "
                f"assets={len(self.assets)}, cash={self.cash_balance})")



# PORTFOLIO ANALYTICS

class PortfolioAnalytics:
    """
    Performs all analytical calculations on a Portfolio.

    Takes a Portfolio instance and provides methods for total valuation,
    profit/loss, weight breakdowns (by asset, sector, asset class),
    and Monte Carlo simulation. This separation keeps Portfolio as a
    simple data container and isolates calculation logic here.
    """

    def __init__(self, portfolio: Portfolio):
        """
        Initialize PortfolioAnalytics.

        Args:
            portfolio: The Portfolio instance to analyze.
        """
        self.portfolio = portfolio

    def get_total_invested_value(self) -> float:
        """
        Calculate the total amount originally invested across all assets.

        Returns:
            Sum of (quantity * purchase_price) for every asset.
        """
        return sum(asset.get_transaction_value() for asset in self.portfolio.assets)

    def get_total_current_value(self) -> float:
        """
        Calculate the total current value of the portfolio.

        Returns:
            Sum of all asset market values + cash balance.
        """
        return (sum(asset.get_current_value() for asset in self.portfolio.assets)
                + self.portfolio.cash_balance)

    def get_total_profit_loss(self) -> float:
        """
        Calculate the total unrealized profit or loss across all assets.

        Returns:
            Sum of (current_value - transaction_value) for every asset.
            Cash is excluded since it doesn't appreciate.
        """
        return sum(asset.get_profit_loss() for asset in self.portfolio.assets)

    def get_asset_weights(self) -> dict:
        """
        Calculate the relative weight of each asset and cash in the portfolio.

        Returns:
            A dict mapping ticker symbols (and 'CASH') to their weight
            as a fraction of total portfolio value.
            Example: {'AAPL': 0.45, 'MSFT': 0.38, 'CASH': 0.17}
        """
        total = self.get_total_current_value()
        if total == 0:
            return {}
        
        weights = {asset.ticker: asset.get_current_value() / total for asset in self.portfolio.assets}
        weights["CASH"] = self.portfolio.cash_balance / total
        return weights

    def get_weights_by_sector(self) -> dict:
        """
        Calculate portfolio weights grouped by sector.

        Returns:
            A dict mapping sector names to their combined weight.
            Example: {'Technology': 0.65, 'Finance': 0.35}
        """
        total = self.get_total_current_value()
        if total == 0:
            return {}
        
        sector_weights = {}

        for asset in self.portfolio.assets:
            value = asset.get_current_value() / total
            sector_weights[asset.sector] = sector_weights.get(asset.sector, 0) + value

        return sector_weights

    def get_weights_by_asset_class(self) -> dict:
        """
        Calculate portfolio weights grouped by asset class.

        Returns:
            A dict mapping asset class names to their combined weight.
            Example: {'EQUITY': 0.80, 'ETF': 0.20}
        """
        total = self.get_total_current_value()

        if total == 0:
            return {}
        
        class_weights = {}

        for asset in self.portfolio.assets:
            value = asset.get_current_value() / total
            class_weights[asset.asset_class] = class_weights.get(asset.asset_class, 0) + value

        return class_weights

    
    def simulate_portfolio(self, years: int = 15, num_simulations: int = 100000) -> dict:
        """
        Run a Monte Carlo simulation using Geometric Brownian Motion (GBM).

        For each asset, estimates drift (mu) and volatility (sigma) from
        historical daily log-returns, then simulates random price paths
        forward. Prices are guaranteed to stay positive.

        Args:
            years: Number of years to simulate forward. Defaults to 15.
            num_simulations: Number of random paths to generate. Defaults to 100,000.

        Returns:
            A dict containing:
                - 'simulations': np.ndarray of shape (num_simulations, trading_days)
                - 'mean': np.ndarray of average portfolio value per day.
                - 'percentile_5': np.ndarray of 5th percentile (worst case).
                - 'percentile_95': np.ndarray of 95th percentile (best case).
        """
        trading_days = years * 252
        dt = 1 / 252  # one trading day as a fraction of a year

        all_simulated_values = np.zeros((num_simulations, trading_days))

        for asset in self.portfolio.assets:
            hist = asset.get_historical_prices(period="5y")

            # Use LOG returns instead of arithmetic returns
            log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

            mu = log_returns.mean() / dt    # annualized drift
            sigma = log_returns.std() / np.sqrt(dt)  # annualized volatility

            # GBM formula: S(t) = S(0) * exp(cumulative sum of daily shocks)
            Z = np.random.standard_normal((num_simulations, trading_days))
            daily_shocks = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            price_paths = asset.get_current_value() * np.exp(np.cumsum(daily_shocks, axis=1))

            all_simulated_values += price_paths

        all_simulated_values += self.portfolio.cash_balance

        return {
            "simulations": all_simulated_values,
            "mean": np.mean(all_simulated_values, axis=0),
            "percentile_5": np.percentile(all_simulated_values, 5, axis=0),
            "percentile_95": np.percentile(all_simulated_values, 95, axis=0),
        }
    



    # EXTENSIONS
    def get_risk_free_rate(self) -> float:
        """
        Fetch the current risk-free rate from ^IRX (13-week US Treasury Bill).

        Returns:
            The annualized risk-free rate as a decimal. Falls back to 0.02 if unavailable.
        """
        return Asset._fetch_risk_free_rate()


    def get_sharpe_ratio_per_asset(self, risk_free_rate: float = None, period: str = "1y") -> dict:
        """
        Calculate the Sharpe Ratio for each asset in the portfolio.

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            A dict mapping ticker symbols to their Sharpe Ratio.
            Example: {'AAPL': 1.45, 'MSFT': 1.82}
        """
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        return {
            asset.ticker: asset.get_sharpe_ratio(risk_free_rate=risk_free_rate, period=period)
            for asset in self.portfolio.assets
        }

    def get_sharpe_ratio_by_sector(self, risk_free_rate: float = None, period: str = "1y") -> dict:
        """
        Calculate a weighted-average Sharpe Ratio per sector.

        Each asset's Sharpe Ratio is weighted by its current value within the sector.

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            A dict mapping sector names to their weighted Sharpe Ratio.
            Example: {'Technology': 1.63, 'Finance': 0.95}
        """
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        sector_values = {}
        sector_weighted_sharpe = {}

        for asset in self.portfolio.assets:
            sector = asset.sector
            value = asset.get_current_value()
            sharpe = asset.get_sharpe_ratio(risk_free_rate=risk_free_rate, period=period)

            sector_values[sector] = sector_values.get(sector, 0) + value
            sector_weighted_sharpe[sector] = sector_weighted_sharpe.get(sector, 0) + sharpe * value

        return {
            sector: sector_weighted_sharpe[sector] / sector_values[sector]
            for sector in sector_values
            if sector_values[sector] > 0
        }

    def get_sharpe_ratio_by_asset_class(self, risk_free_rate: float = None, period: str = "1y") -> dict:
        """
        Calculate a weighted-average Sharpe Ratio per asset class.

        Each asset's Sharpe Ratio is weighted by its current value within the asset class.

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            A dict mapping asset class names to their weighted Sharpe Ratio.
            Example: {'EQUITY': 1.50, 'ETF': 1.20}
        """
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        class_values = {}
        class_weighted_sharpe = {}

        for asset in self.portfolio.assets:
            ac = asset.asset_class
            value = asset.get_current_value()
            sharpe = asset.get_sharpe_ratio(risk_free_rate=risk_free_rate, period=period)

            class_values[ac] = class_values.get(ac, 0) + value
            class_weighted_sharpe[ac] = class_weighted_sharpe.get(ac, 0) + sharpe * value

        return {
            ac: class_weighted_sharpe[ac] / class_values[ac]
            for ac in class_values
            if class_values[ac] > 0
        }

    def get_portfolio_sharpe_ratio(self, risk_free_rate: float = None, period: str = "1y") -> float:
        """
        Calculate the overall Sharpe Ratio for the entire portfolio.

        Uses the portfolio's combined daily returns (weighted by current value)
        rather than averaging individual Sharpe Ratios, which correctly
        accounts for diversification effects.

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            The portfolio-level Sharpe Ratio as a float.
        """
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        total_value = sum(a.get_current_value() for a in self.portfolio.assets)
        if total_value == 0:
            return 0.0

        weights = [a.get_current_value() / total_value for a in self.portfolio.assets]
        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]

        aligned = pd.concat(returns_list, axis=1).dropna()
        if aligned.empty:
            return 0.0
        
        portfolio_returns = aligned.values @ np.array(weights)

        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)

        if ann_vol == 0:
            return 0.0

        return (ann_return - risk_free_rate) / ann_vol

    def get_correlation_matrix(self, period: str = "1y") -> pd.DataFrame:
        """
        Compute the correlation matrix of daily returns across all assets.

        Args:
            period: How far back to look for historical returns.

        Returns:
            A pandas DataFrame where both rows and columns are ticker symbols,
            and each cell contains the Pearson correlation coefficient (-1 to 1).
            Example:
                AAPL  MSFT   JPM
            AAPL  1.00  0.78  0.45
            MSFT  0.78  1.00  0.52
            JPM   0.45  0.52  1.00
        """
        tickers = [a.ticker for a in self.portfolio.assets]
        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]

        df = pd.concat(returns_list, axis=1).dropna()
        if df.empty:
            return pd.DataFrame()
        
        df.columns = tickers
        return df.corr()

    def get_covariance_matrix(self, period: str = "1y") -> pd.DataFrame:
        """
        Compute the annualized covariance matrix of daily returns across all assets.

        Used internally for Markowitz optimization and efficient frontier calculation.

        Args:
            period: How far back to look for historical returns.

        Returns:
            A pandas DataFrame with annualized covariances. Same structure as
            the correlation matrix but with covariance values.
        """
        tickers = [a.ticker for a in self.portfolio.assets]
        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]

        df = pd.concat(returns_list, axis=1).dropna()
        if df.empty:
            return pd.DataFrame()
        
        df.columns = tickers
        return df.cov() * 252

    def get_optimal_weights(self, risk_free_rate: float = None, period: str = "1y") -> dict:
        """
        Calculate optimal portfolio weights by maximizing the Sharpe Ratio
        using Markowitz Mean-Variance Optimization.

        Uses scipy's minimize function with the SLSQP method. Constraints
        ensure weights sum to 1 and each weight is between 0 and 1 (long only).

        Args:
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            A dict containing:
                - 'current_weights': dict of current ticker -> weight
                - 'optimal_weights': dict of optimal ticker -> weight
                - 'current_sharpe': float, Sharpe Ratio with current weights
                - 'optimal_sharpe': float, Sharpe Ratio with optimal weights
                - 'current_return': float, annualized return with current weights
                - 'optimal_return': float, annualized return with optimal weights
                - 'current_volatility': float, annualized vol with current weights
                - 'optimal_volatility': float, annualized vol with optimal weights
        """
        from scipy.optimize import minimize

        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        tickers = [a.ticker for a in self.portfolio.assets]
        n = len(tickers)

        if n == 0:
            return {}

        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]
        df = pd.concat(returns_list, axis=1).dropna()
        if df.empty:
            return {}
        
        df.columns = tickers

        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252

        total_value = sum(a.get_current_value() for a in self.portfolio.assets)
        current_weights = np.array([a.get_current_value() / total_value for a in self.portfolio.assets])

        def neg_sharpe(weights):
            """Negative Sharpe Ratio (we minimize, so negative = maximize)."""
            port_return = weights @ mean_returns.values
            port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            if port_vol == 0:
                return 0
            return -(port_return - risk_free_rate) / port_vol

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]
        initial_guess = np.array([1 / n] * n)

        result = minimize(neg_sharpe, initial_guess, method="SLSQP",
                        bounds=bounds, constraints=constraints)

        optimal_w = result.x

        cur_ret = current_weights @ mean_returns.values
        cur_vol = np.sqrt(current_weights @ cov_matrix.values @ current_weights)
        cur_sharpe = (cur_ret - risk_free_rate) / cur_vol if cur_vol > 0 else 0.0

        opt_ret = optimal_w @ mean_returns.values
        opt_vol = np.sqrt(optimal_w @ cov_matrix.values @ optimal_w)
        opt_sharpe = (opt_ret - risk_free_rate) / opt_vol if opt_vol > 0 else 0.0

        return {
            "current_weights": dict(zip(tickers, current_weights)),
            "optimal_weights": dict(zip(tickers, optimal_w)),
            "current_sharpe": cur_sharpe,
            "optimal_sharpe": opt_sharpe,
            "current_return": cur_ret,
            "optimal_return": opt_ret,
            "current_volatility": cur_vol,
            "optimal_volatility": opt_vol,
        }

    def get_efficient_frontier(self, num_points: int = 100, risk_free_rate: float = None,
                            period: str = "1y") -> dict:
        """
        Compute the efficient frontier: a set of optimal portfolios offering
        the highest return for each level of risk.

        Generates num_points portfolios by targeting equally spaced return levels
        between the minimum and maximum single-asset returns, and finding the
        minimum-volatility portfolio for each target return.

        Args:
            num_points: Number of points along the frontier. Defaults to 100.
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.
            period: How far back to look.

        Returns:
            A dict containing:
                - 'returns': list of annualized returns along the frontier
                - 'volatilities': list of annualized volatilities along the frontier
                - 'weights': list of dicts (ticker -> weight) for each point
                - 'sharpe_ratios': list of Sharpe Ratios for each point
                - 'optimal_point': dict with the single best Sharpe point
                    ('return', 'volatility', 'sharpe', 'weights')
        """
        from scipy.optimize import minimize

        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        tickers = [a.ticker for a in self.portfolio.assets]
        n = len(tickers)

        if n < 2:
            return {}

        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]
        df = pd.concat(returns_list, axis=1).dropna()
        if df.empty:
            return {}
        
        df.columns = tickers

        mean_returns = df.mean() * 252
        cov_matrix = df.cov() * 252

        target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []
        frontier_sharpes = []

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mean_returns.values - t},
            ]
            bounds = [(0, 1) for _ in range(n)]
            initial_guess = np.array([1 / n] * n)

            def portfolio_volatility(weights):
                """Portfolio volatility for given weights."""
                return np.sqrt(weights @ cov_matrix.values @ weights)

            result = minimize(portfolio_volatility, initial_guess, method="SLSQP",
                            bounds=bounds, constraints=constraints)

            if result.success:
                vol = portfolio_volatility(result.x)
                sharpe = (target - risk_free_rate) / vol if vol > 0 else 0.0

                frontier_returns.append(target)
                frontier_volatilities.append(vol)
                frontier_weights.append(dict(zip(tickers, result.x)))
                frontier_sharpes.append(sharpe)

        best_idx = np.argmax(frontier_sharpes) if frontier_sharpes else 0

        return {
            "returns": frontier_returns,
            "volatilities": frontier_volatilities,
            "weights": frontier_weights,
            "sharpe_ratios": frontier_sharpes,
            "optimal_point": {
                "return": frontier_returns[best_idx],
                "volatility": frontier_volatilities[best_idx],
                "sharpe": frontier_sharpes[best_idx],
                "weights": frontier_weights[best_idx],
            } if frontier_sharpes else {},
        }

    def get_benchmark_comparison(self, benchmark_ticker: str = "ACWI",
                                period: str = "1y", risk_free_rate: float = None) -> dict:
        """
        Compare portfolio performance against a benchmark index.

        Fetches the benchmark's historical data and computes return, volatility,
        and Sharpe Ratio for both the portfolio and the benchmark side by side.

        Args:
            benchmark_ticker: Ticker symbol for the benchmark. Defaults to 'ACWI'
                            (MSCI All World). Use 'IUSQ' as a European alternative.
            period: How far back to look.
            risk_free_rate: Annual risk-free rate. If None, fetched from ^IRX.

        Returns:
            A dict containing:
                - 'portfolio': dict with 'annualized_return', 'annualized_volatility',
                            'sharpe_ratio', 'cumulative_returns' (pd.Series)
                - 'benchmark': dict with same keys
                - 'alpha': portfolio annualized return minus benchmark annualized return
                - 'tracking_error': annualized std of (portfolio returns - benchmark returns)
        """
        if risk_free_rate is None:
            risk_free_rate = self.get_risk_free_rate()

        try:
            bench = yf.Ticker(benchmark_ticker)
            bench_hist = bench.history(period=period)
            bench_returns = bench_hist["Close"].pct_change().dropna()
        except Exception:
            if benchmark_ticker != "IUSQ":
                return self.get_benchmark_comparison(
                    benchmark_ticker="IUSQ", period=period, risk_free_rate=risk_free_rate
                )
            return {}

        total_value = sum(a.get_current_value() for a in self.portfolio.assets)
        if total_value == 0:
            return {}

        weights = [a.get_current_value() / total_value for a in self.portfolio.assets]
        returns_list = [a.get_daily_returns(period=period) for a in self.portfolio.assets]

        aligned = pd.concat(returns_list, axis=1).dropna()
        if aligned.empty:
            return {}
        
        port_returns = pd.Series(aligned.values @ np.array(weights), index=aligned.index)

        common_dates = port_returns.index.intersection(bench_returns.index)
        if common_dates.empty:
            return {}
        
        port_returns = port_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]

        port_ann_ret = port_returns.mean() * 252
        port_ann_vol = port_returns.std() * np.sqrt(252)
        port_sharpe = (port_ann_ret - risk_free_rate) / port_ann_vol if port_ann_vol > 0 else 0.0
        port_cumulative = (1 + port_returns).cumprod()

        bench_ann_ret = bench_returns.mean() * 252
        bench_ann_vol = bench_returns.std() * np.sqrt(252)
        bench_sharpe = (bench_ann_ret - risk_free_rate) / bench_ann_vol if bench_ann_vol > 0 else 0.0
        bench_cumulative = (1 + bench_returns).cumprod()

        excess_returns = port_returns - bench_returns
        tracking_error = excess_returns.std() * np.sqrt(252)

        return {
            "portfolio": {
                "annualized_return": port_ann_ret,
                "annualized_volatility": port_ann_vol,
                "sharpe_ratio": port_sharpe,
                "cumulative_returns": port_cumulative,
            },
            "benchmark": {
                "annualized_return": bench_ann_ret,
                "annualized_volatility": bench_ann_vol,
                "sharpe_ratio": bench_sharpe,
                "cumulative_returns": bench_cumulative,
            },
            "alpha": port_ann_ret - bench_ann_ret,
            "tracking_error": tracking_error,
        }

    def get_esg_scores(self) -> dict:
        """
        Collect ESG scores for all assets in the portfolio.

        Returns:
            A dict mapping ticker symbols to their ESG score dicts.
            Example: {'AAPL': {'totalEsg': 16.7, 'environmentScore': 3.4, ...}, ...}
        """
        return {asset.ticker: asset.get_esg_score() for asset in self.portfolio.assets}














