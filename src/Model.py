import yfinance as yf
import numpy as np
import pandas as pd



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
        Run a Monte Carlo simulation to project future portfolio value.

        For each asset, calculates the mean and standard deviation of
        historical daily returns, then simulates random price paths
        forward. Results are aggregated into total portfolio value
        over time.

        Args:
            years: Number of years to simulate forward. Defaults to 15.
            num_simulations: Number of random paths to generate. Defaults to 100,000.

        Returns:
            A dict containing:
                - 'simulations': np.ndarray of shape (num_simulations, trading_days)
                                with total portfolio value per day per simulation.
                - 'mean': np.ndarray of average portfolio value per day.
                - 'percentile_5': np.ndarray of 5th percentile (worst case).
                - 'percentile_95': np.ndarray of 95th percentile (best case).
        """
        trading_days = years * 252

        # Collect historical daily returns for each asset
        all_simulated_values = np.zeros((num_simulations, trading_days))

        for asset in self.portfolio.assets:
            hist = asset.get_historical_prices(period="5y")
            daily_returns = hist["Close"].pct_change().dropna()

            mean_return = daily_returns.mean()
            std_return = daily_returns.std()

            # Generate random daily returns: one row per simulation, one column per day
            random_returns = np.random.normal(mean_return, std_return,
                                            (num_simulations, trading_days))

            # Convert returns to price paths starting from current value
            price_paths = asset.get_current_value() * np.cumprod(1 + random_returns, axis=1)

            all_simulated_values += price_paths

        # Add cash (stays constant, no growth)
        all_simulated_values += self.portfolio.cash_balance

        return {
            "simulations": all_simulated_values,
            "mean": np.mean(all_simulated_values, axis=0),
            "percentile_5": np.percentile(all_simulated_values, 5, axis=0),
            "percentile_95": np.percentile(all_simulated_values, 95, axis=0),
        }












