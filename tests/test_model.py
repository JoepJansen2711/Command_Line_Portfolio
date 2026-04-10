import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from Model import Asset, Portfolio, PortfolioAnalytics


# ──────────────────────────────────────────────
# HELPER: Create a mock yfinance Ticker
# ──────────────────────────────────────────────
def create_mock_ticker(current_price=150.0, sector="Technology", quote_type="EQUITY"):
    """
    Build a fake yf.Ticker object so tests don't hit the internet.
    """
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "currentPrice": current_price,
        "regularMarketPrice": current_price,
        "sector": sector,
        "quoteType": quote_type,
    }

    # Fake 10 days of historical data
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]
    mock_ticker.history.return_value = pd.DataFrame({
        "Open": prices,
        "High": [p + 1 for p in prices],
        "Low": [p - 1 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 10,
        "Dividends": [0.0] * 10,
        "Stock Splits": [0.0] * 10,
    }, index=dates)

    return mock_ticker


# ──────────────────────────────────────────────
# ASSET TESTS
# ──────────────────────────────────────────────
class TestAsset(unittest.TestCase):
    """Tests for the Asset class."""

    @patch("Model.yf.Ticker")
    def setUp(self, mock_yf_ticker):
        """
        Create a sample asset before each test.
        Mocks yfinance so no real API calls are made.
        """
        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=150.0, sector="Technology", quote_type="EQUITY"
        )
        self.asset = Asset("AAPL", quantity=10, purchase_price=120.0)

    def test_init_ticker(self):
        """Verify the ticker symbol is stored correctly."""
        self.assertEqual(self.asset.ticker, "AAPL")

    def test_init_quantity(self):
        """Verify the quantity is stored correctly."""
        self.assertEqual(self.asset.quantity, 10)

    def test_init_purchase_price(self):
        """Verify the purchase price is stored correctly."""
        self.assertEqual(self.asset.purchase_price, 120.0)

    def test_init_sector_from_yahoo(self):
        """Verify sector is auto-fetched from Yahoo Finance when not provided."""
        self.assertEqual(self.asset.sector, "Technology")

    def test_init_asset_class_from_yahoo(self):
        """Verify asset class is auto-fetched from Yahoo Finance when not provided."""
        self.assertEqual(self.asset.asset_class, "EQUITY")

    @patch("Model.yf.Ticker")
    def test_init_sector_manual_override(self, mock_yf_ticker):
        """Verify manually provided sector overrides Yahoo Finance."""
        mock_yf_ticker.return_value = create_mock_ticker()
        asset = Asset("AAPL", 10, 120.0, sector="Tech", asset_class="Stock")
        self.assertEqual(asset.sector, "Tech")

    @patch("Model.yf.Ticker")
    def test_init_asset_class_manual_override(self, mock_yf_ticker):
        """Verify manually provided asset class overrides Yahoo Finance."""
        mock_yf_ticker.return_value = create_mock_ticker()
        asset = Asset("AAPL", 10, 120.0, sector="Tech", asset_class="Stock")
        self.assertEqual(asset.asset_class, "Stock")

    def test_get_current_price(self):
        """Verify current price is fetched correctly from Yahoo Finance."""
        price = self.asset.get_current_price()
        self.assertEqual(price, 150.0)

    def test_get_historical_prices_returns_dataframe(self):
        """Verify historical prices returns a pandas DataFrame."""
        df = self.asset.get_historical_prices(period="1y")
        self.assertIsInstance(df, pd.DataFrame)

    def test_get_historical_prices_has_close_column(self):
        """Verify the historical DataFrame contains a 'Close' column."""
        df = self.asset.get_historical_prices()
        self.assertIn("Close", df.columns)

    def test_get_transaction_value(self):
        """Verify transaction value = quantity * purchase_price."""
        self.assertEqual(self.asset.get_transaction_value(), 10 * 120.0)

    def test_get_current_value(self):
        """Verify current value = quantity * current market price."""
        self.assertEqual(self.asset.get_current_value(), 10 * 150.0)

    def test_get_profit_loss_positive(self):
        """Verify profit is positive when current price > purchase price."""
        pnl = self.asset.get_profit_loss()
        self.assertEqual(pnl, (150.0 - 120.0) * 10)
        self.assertGreater(pnl, 0)

    @patch("Model.yf.Ticker")
    def test_get_profit_loss_negative(self, mock_yf_ticker):
        """Verify loss is negative when current price < purchase price."""
        mock_yf_ticker.return_value = create_mock_ticker(current_price=80.0)
        asset = Asset("TSLA", quantity=5, purchase_price=100.0)
        self.assertLess(asset.get_profit_loss(), 0)

    def test_repr(self):
        """Verify __repr__ contains the ticker symbol."""
        result = repr(self.asset)
        self.assertIn("AAPL", result)


# ──────────────────────────────────────────────
# PORTFOLIO TESTS
# ──────────────────────────────────────────────
class TestPortfolio(unittest.TestCase):
    """Tests for the Portfolio class."""

    def setUp(self):
        """Create a sample portfolio before each test."""
        self.portfolio = Portfolio("Test Portfolio", currency="EUR", cash_balance=5000.0)

    def test_init_name(self):
        """Verify portfolio name is stored correctly."""
        self.assertEqual(self.portfolio.name, "Test Portfolio")

    def test_init_currency(self):
        """Verify currency is stored correctly."""
        self.assertEqual(self.portfolio.currency, "EUR")

    def test_init_cash_balance(self):
        """Verify initial cash balance is stored correctly."""
        self.assertEqual(self.portfolio.cash_balance, 5000.0)

    def test_init_empty_assets(self):
        """Verify portfolio starts with an empty asset list."""
        self.assertEqual(len(self.portfolio.assets), 0)

    @patch("Model.yf.Ticker")
    def test_add_asset(self, mock_yf_ticker):
        """Verify an asset can be added to the portfolio."""
        mock_yf_ticker.return_value = create_mock_ticker()
        asset = Asset("AAPL", 10, 150.0)
        self.portfolio.add_asset(asset)
        self.assertEqual(len(self.portfolio.assets), 1)

    @patch("Model.yf.Ticker")
    def test_add_multiple_assets(self, mock_yf_ticker):
        """Verify multiple assets can be added."""
        mock_yf_ticker.return_value = create_mock_ticker()
        self.portfolio.add_asset(Asset("AAPL", 10, 150.0))
        self.portfolio.add_asset(Asset("MSFT", 5, 380.0))
        self.assertEqual(len(self.portfolio.assets), 2)

    @patch("Model.yf.Ticker")
    def test_remove_asset(self, mock_yf_ticker):
        """Verify an asset can be removed by ticker symbol."""
        mock_yf_ticker.return_value = create_mock_ticker()
        self.portfolio.add_asset(Asset("AAPL", 10, 150.0))
        self.portfolio.remove_asset("AAPL")
        self.assertEqual(len(self.portfolio.assets), 0)

    @patch("Model.yf.Ticker")
    def test_remove_nonexistent_asset(self, mock_yf_ticker):
        """Verify removing a ticker that doesn't exist causes no error."""
        mock_yf_ticker.return_value = create_mock_ticker()
        self.portfolio.add_asset(Asset("AAPL", 10, 150.0))
        self.portfolio.remove_asset("TSLA")
        self.assertEqual(len(self.portfolio.assets), 1)

    @patch("Model.yf.Ticker")
    def test_get_asset_found(self, mock_yf_ticker):
        """Verify get_asset returns the correct asset when found."""
        mock_yf_ticker.return_value = create_mock_ticker()
        self.portfolio.add_asset(Asset("AAPL", 10, 150.0))
        result = self.portfolio.get_asset("AAPL")
        self.assertIsNotNone(result)
        self.assertEqual(result.ticker, "AAPL")

    def test_get_asset_not_found(self):
        """Verify get_asset returns None when ticker doesn't exist."""
        result = self.portfolio.get_asset("TSLA")
        self.assertIsNone(result)

    def test_deposit_cash(self):
        """Verify cash deposit increases the balance."""
        self.portfolio.deposit_cash(1000.0)
        self.assertEqual(self.portfolio.cash_balance, 6000.0)

    def test_withdraw_cash(self):
        """Verify cash withdrawal decreases the balance."""
        self.portfolio.withdraw_cash(2000.0)
        self.assertEqual(self.portfolio.cash_balance, 3000.0)

    def test_withdraw_cash_insufficient(self):
        """Verify withdrawing more than available raises ValueError."""
        with self.assertRaises(ValueError):
            self.portfolio.withdraw_cash(10000.0)

    def test_repr(self):
        """Verify __repr__ contains the portfolio name."""
        result = repr(self.portfolio)
        self.assertIn("Test Portfolio", result)


# ──────────────────────────────────────────────
# PORTFOLIO ANALYTICS TESTS
# ──────────────────────────────────────────────
class TestPortfolioAnalytics(unittest.TestCase):
    """Tests for the PortfolioAnalytics class."""

    @patch("Model.yf.Ticker")
    def setUp(self, mock_yf_ticker):
        """
        Build a sample portfolio with two assets and cash before each test.

        AAPL: 10 shares, bought at €120, now worth €150 each -> value €1,500
        MSFT: 5 shares, bought at €350, now worth €400 each  -> value €2,000
        Cash: €500
        Total: €4,000
        """
        self.portfolio = Portfolio("ING Test Portfolio", currency="EUR", cash_balance=500.0)

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=150.0, sector="Technology", quote_type="EQUITY"
        )
        self.apple = Asset("AAPL", quantity=10, purchase_price=120.0)
        self.portfolio.add_asset(self.apple)

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=400.0, sector="Technology", quote_type="EQUITY"
        )
        self.microsoft = Asset("MSFT", quantity=5, purchase_price=350.0)
        self.portfolio.add_asset(self.microsoft)

        self.analytics = PortfolioAnalytics(self.portfolio)

    def test_get_total_invested_value(self):
        """
        Verify total invested = sum of all transaction values.
        AAPL: 10 * 120 = 1200, MSFT: 5 * 350 = 1750 -> total 2950
        """
        expected = (10 * 120.0) + (5 * 350.0)
        self.assertEqual(self.analytics.get_total_invested_value(), expected)

    def test_get_total_current_value(self):
        """
        Verify total current value = sum of market values + cash.
        AAPL: 10 * 150 = 1500, MSFT: 5 * 400 = 2000, Cash: 500 -> 4000
        """
        expected = (10 * 150.0) + (5 * 400.0) + 500.0
        self.assertEqual(self.analytics.get_total_current_value(), expected)

    def test_get_total_profit_loss(self):
        """
        Verify total P&L = total current value (excl. cash) - total invested.
        (1500 + 2000) - (1200 + 1750) = 550
        """
        expected = ((10 * 150.0) + (5 * 400.0)) - ((10 * 120.0) + (5 * 350.0))
        self.assertEqual(self.analytics.get_total_profit_loss(), expected)

    def test_get_asset_weights_sum_to_one(self):
        """Verify all asset weights (including cash) sum to 1.0."""
        weights = self.analytics.get_asset_weights()
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_get_asset_weights_contains_cash(self):
        """Verify the weights dict includes a CASH entry."""
        weights = self.analytics.get_asset_weights()
        self.assertIn("CASH", weights)

    def test_get_asset_weights_values(self):
        """
        Verify individual weight calculations.
        Total = 4000. AAPL = 1500/4000 = 0.375, MSFT = 2000/4000 = 0.5, CASH = 500/4000 = 0.125
        """
        weights = self.analytics.get_asset_weights()
        self.assertAlmostEqual(weights["AAPL"], 1500.0 / 4000.0)
        self.assertAlmostEqual(weights["MSFT"], 2000.0 / 4000.0)
        self.assertAlmostEqual(weights["CASH"], 500.0 / 4000.0)

    def test_get_asset_weights_empty_portfolio(self):
        """Verify empty portfolio returns empty dict."""
        empty = Portfolio("Empty", cash_balance=0.0)
        analytics = PortfolioAnalytics(empty)
        self.assertEqual(analytics.get_asset_weights(), {})

    def test_get_weights_by_sector(self):
        """
        Verify sector weights. Both assets are Technology.
        Technology = (1500 + 2000) / 4000 = 0.875
        """
        sector_weights = self.analytics.get_weights_by_sector()
        self.assertIn("Technology", sector_weights)
        self.assertAlmostEqual(sector_weights["Technology"], 3500.0 / 4000.0)

    @patch("Model.yf.Ticker")
    def test_get_weights_by_sector_multiple_sectors(self, mock_yf_ticker):
        """Verify weights split correctly across different sectors."""
        portfolio = Portfolio("Multi Sector", currency="EUR", cash_balance=0.0)

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=100.0, sector="Technology", quote_type="EQUITY"
        )
        portfolio.add_asset(Asset("AAPL", 10, 90.0))

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=50.0, sector="Finance", quote_type="EQUITY"
        )
        portfolio.add_asset(Asset("JPM", 10, 40.0))

        analytics = PortfolioAnalytics(portfolio)
        sector_weights = analytics.get_weights_by_sector()

        # AAPL = 1000, JPM = 500, total = 1500
        self.assertAlmostEqual(sector_weights["Technology"], 1000.0 / 1500.0)
        self.assertAlmostEqual(sector_weights["Finance"], 500.0 / 1500.0)

    def test_get_weights_by_asset_class(self):
        """
        Verify asset class weights. Both assets are EQUITY.
        EQUITY = (1500 + 2000) / 4000 = 0.875
        """
        class_weights = self.analytics.get_weights_by_asset_class()
        self.assertIn("EQUITY", class_weights)
        self.assertAlmostEqual(class_weights["EQUITY"], 3500.0 / 4000.0)

    @patch("Model.yf.Ticker")
    def test_get_weights_by_asset_class_mixed(self, mock_yf_ticker):
        """Verify weights split correctly across EQUITY and ETF."""
        portfolio = Portfolio("Mixed", currency="EUR", cash_balance=0.0)

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=100.0, sector="Technology", quote_type="EQUITY"
        )
        portfolio.add_asset(Asset("AAPL", 10, 90.0))

        mock_yf_ticker.return_value = create_mock_ticker(
            current_price=200.0, sector="Blend", quote_type="ETF"
        )
        portfolio.add_asset(Asset("SPY", 5, 180.0))

        analytics = PortfolioAnalytics(portfolio)
        class_weights = analytics.get_weights_by_asset_class()

        # AAPL = 1000, SPY = 1000, total = 2000
        self.assertAlmostEqual(class_weights["EQUITY"], 0.5)
        self.assertAlmostEqual(class_weights["ETF"], 0.5)

    def test_simulate_portfolio_returns_dict(self):
        """Verify simulation returns a dict with the expected keys."""
        result = self.analytics.simulate_portfolio(years=1, num_simulations=100)
        self.assertIsInstance(result, dict)
        self.assertIn("simulations", result)
        self.assertIn("mean", result)
        self.assertIn("percentile_5", result)
        self.assertIn("percentile_95", result)

    def test_simulate_portfolio_shape(self):
        """Verify simulation array has correct shape (simulations x trading days)."""
        result = self.analytics.simulate_portfolio(years=1, num_simulations=100)
        expected_days = 1 * 252
        self.assertEqual(result["simulations"].shape, (100, expected_days))

    def test_simulate_portfolio_mean_shape(self):
        """Verify mean array has one value per trading day."""
        result = self.analytics.simulate_portfolio(years=1, num_simulations=100)
        self.assertEqual(result["mean"].shape, (252,))

    def test_simulate_portfolio_percentiles_order(self):
        """Verify 5th percentile <= mean <= 95th percentile at each time step."""
        result = self.analytics.simulate_portfolio(years=1, num_simulations=1000)
        self.assertTrue(np.all(result["percentile_5"] <= result["mean"]))
        self.assertTrue(np.all(result["mean"] <= result["percentile_95"]))

    def test_simulate_portfolio_positive_values(self):
        """Verify simulated portfolio values are positive (no negative wealth)."""
        result = self.analytics.simulate_portfolio(years=1, num_simulations=100)
        self.assertTrue(np.all(result["simulations"] > 0))


if __name__ == "__main__":
    unittest.main()