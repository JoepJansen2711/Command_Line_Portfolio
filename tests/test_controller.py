"""
Comprehensive test suite for Controller module.

Tests all interactive CLI methods with mocked user input, View, and Model.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from src.Controller import Controller, _ask, _ask_float, _ask_int, _ask_positive_int, _ask_positive_float, _ask_non_empty, _pick
from src.Model import Asset, Portfolio, PortfolioAnalytics


class TestHelperFunctions(unittest.TestCase):
    """Test controller helper functions."""

    @patch("src.Controller.Prompt.ask")
    def test_ask_returns_stripped_string(self, mock_prompt):
        """Verify _ask strips whitespace."""
        mock_prompt.return_value = "  AAPL  "
        result = _ask("Test prompt")
        self.assertEqual(result, "AAPL")

    @patch("src.Controller.Prompt.ask")
    def test_ask_with_default(self, mock_prompt):
        """Verify _ask uses default."""
        mock_prompt.return_value = "MyDefault"
        result = _ask("Test", "MyDefault")
        self.assertEqual(result, "MyDefault")

    @patch("src.Controller.Prompt.ask")
    def test_ask_float_valid(self, mock_prompt):
        """Verify _ask_float parses float."""
        mock_prompt.return_value = "3.14"
        result = _ask_float("Price")
        self.assertAlmostEqual(result, 3.14)

    @patch("src.Controller.Prompt.ask")
    def test_ask_float_invalid_retries(self, mock_prompt):
        """Verify _ask_float retries on invalid input."""
        mock_prompt.side_effect = ["invalid", "2.5"]
        result = _ask_float("Price")
        self.assertAlmostEqual(result, 2.5)

    @patch("src.Controller.Prompt.ask")
    def test_ask_int_valid(self, mock_prompt):
        """Verify _ask_int parses integer."""
        mock_prompt.return_value = "42"
        result = _ask_int("Quantity")
        self.assertEqual(result, 42)

    @patch("src.Controller.Prompt.ask")
    def test_ask_positive_int_valid(self, mock_prompt):
        """Verify _ask_positive_int rejects non-positive values."""
        mock_prompt.side_effect = ["0", "-5", "10"]
        result = _ask_positive_int("Count")
        self.assertEqual(result, 10)

    @patch("src.Controller.Prompt.ask")
    def test_ask_positive_float_valid(self, mock_prompt):
        """Verify _ask_positive_float rejects non-positive values."""
        mock_prompt.side_effect = ["0.0", "-1.5", "5.5"]
        result = _ask_positive_float("Amount")
        self.assertAlmostEqual(result, 5.5)

    @patch("src.Controller.Prompt.ask")
    def test_ask_non_empty_rejects_empty(self, mock_prompt):
        """Verify _ask_non_empty rejects empty strings."""
        mock_prompt.side_effect = ["", "  ", "ValidValue"]
        result = _ask_non_empty("Name")
        self.assertEqual(result, "ValidValue")

    @patch("src.Controller.console")
    @patch("src.Controller.Prompt.ask")
    def test_pick_valid_choice(self, mock_prompt, mock_console):
        """Verify _pick returns selected option."""
        mock_prompt.return_value = "2"
        options = ["Option1", "Option2", "Option3"]
        result = _pick("Choose", options)
        self.assertEqual(result, "Option2")

    @patch("src.Controller.console")
    @patch("src.Controller.Prompt.ask")
    def test_pick_invalid_then_valid(self, mock_prompt, mock_console):
        """Verify _pick retries on invalid input."""
        mock_prompt.side_effect = ["0", "5", "1"]
        options = ["A", "B"]
        result = _pick("Choose", options)
        self.assertEqual(result, "A")


class TestControllerInit(unittest.TestCase):
    """Test Controller initialization."""

    def setUp(self):
        """Create a fresh Controller instance."""
        self.controller = Controller()

    def test_controller_init(self):
        """Verify Controller initializes with empty portfolio."""
        self.assertIsNotNone(self.controller.view)
        self.assertIsNone(self.controller.portfolio)
        self.assertIsNone(self.controller.analytics)


class TestControllerInitPortfolio(unittest.TestCase):
    """Test portfolio initialization."""

    def setUp(self):
        """Create a fresh Controller instance."""
        self.controller = Controller()

    @patch("src.Controller.console")
    @patch("src.Controller._ask")
    @patch("src.Controller._ask_float")
    def test_init_portfolio_creates_portfolio(self, mock_ask_float, mock_ask, mock_console):
        """Verify _init_portfolio creates Portfolio and PortfolioAnalytics."""
        mock_ask.side_effect = ["My Portfolio", "EUR"]
        mock_ask_float.return_value = 10000.0

        self.controller._init_portfolio()

        self.assertIsNotNone(self.controller.portfolio)
        self.assertIsNotNone(self.controller.analytics)
        self.assertEqual(self.controller.portfolio.name, "My Portfolio")
        self.assertEqual(self.controller.portfolio.currency, "EUR")

    @patch("src.Controller.console")
    @patch("src.Controller._ask")
    @patch("src.Controller._ask_float")
    def test_init_portfolio_with_defaults(self, mock_ask_float, mock_ask, mock_console):
        """Verify _init_portfolio uses defaults."""
        mock_ask.side_effect = ["My Portfolio", "GBP"]
        mock_ask_float.return_value = 0.0

        self.controller._init_portfolio()

        self.assertEqual(self.controller.portfolio.cash_balance, 0.0)


class TestControllerAddAsset(unittest.TestCase):
    """Test adding assets."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller.Asset")
    @patch("src.Controller._ask_non_empty")
    @patch("src.Controller._ask_positive_int")
    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._ask")
    @patch("src.Controller.console")
    def test_add_asset_success(self, mock_console, mock_ask, mock_ask_pos_float,
                               mock_ask_pos_int, mock_ask_ne, mock_asset_class, mock_confirm):
        """Verify _add_asset creates and adds asset."""
        mock_ask_pos_int.return_value = 10
        mock_ask_pos_float.return_value = 150.0
        mock_ask.side_effect = ["Technology", "EQUITY"]
        mock_ask_ne.return_value = "AAPL"

        mock_asset = MagicMock()
        mock_asset.sector = "Technology"
        mock_asset.asset_class = "EQUITY"
        mock_asset_class.return_value = mock_asset
        self.controller.portfolio.currency = "EUR"

        self.controller._add_asset()

        self.controller.portfolio.add_asset.assert_called_once_with(mock_asset)
        self.controller.view.show_success.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller.Asset")
    @patch("src.Controller._ask_non_empty")
    @patch("src.Controller._ask_positive_int")
    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._ask")
    @patch("src.Controller.console")
    def test_add_asset_failure(self, mock_console, mock_ask, mock_ask_pos_float,
                               mock_ask_pos_int, mock_ask_ne, mock_asset_class, mock_confirm):
        """Verify _add_asset shows error and asks to retry on invalid ticker."""
        mock_ask_pos_int.return_value = 10
        mock_ask_pos_float.return_value = 150.0
        mock_ask.side_effect = ["", ""]
        mock_ask_ne.return_value = "INVALID"
        mock_asset_class.side_effect = Exception("Bad ticker")
        mock_confirm.return_value = False  # User declines to retry

        self.controller._add_asset()

        self.controller.view.show_error.assert_called_once()
        self.controller.portfolio.add_asset.assert_not_called()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller.Asset")
    @patch("src.Controller._ask_non_empty")
    @patch("src.Controller._ask_positive_int")
    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._ask")
    @patch("src.Controller.console")
    def test_add_asset_retry_succeeds(self, mock_console, mock_ask, mock_ask_pos_float,
                                      mock_ask_pos_int, mock_ask_ne, mock_asset_class, mock_confirm):
        """Verify _add_asset retries and succeeds with valid ticker after first failure."""
        mock_ask_pos_int.return_value = 10
        mock_ask_pos_float.return_value = 150.0
        mock_ask.side_effect = ["Technology", "EQUITY"]
        mock_ask_ne.side_effect = ["INVALID", "AAPL"]  # First attempt invalid, second valid
        mock_confirm.return_value = True  # User wants to retry

        mock_invalid_asset = MagicMock()
        mock_valid_asset = MagicMock()
        mock_valid_asset.sector = "Technology"
        mock_valid_asset.asset_class = "EQUITY"

        # First call raises exception, second succeeds
        mock_asset_class.side_effect = [Exception("Bad ticker"), mock_valid_asset]
        self.controller.portfolio.currency = "EUR"

        self.controller._add_asset()

        # Verify error was shown for first attempt
        self.controller.view.show_error.assert_called_once()
        # Verify success was shown for second attempt
        self.controller.view.show_success.assert_called_once()
        # Verify asset was added
        self.controller.portfolio.add_asset.assert_called_once_with(mock_valid_asset)


class TestControllerRemoveAsset(unittest.TestCase):
    """Test removing assets."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._ask_non_empty")
    def test_remove_asset_confirmed(self, mock_ask_ne, mock_confirm):
        """Verify _remove_asset removes when confirmed."""
        mock_asset = MagicMock()
        mock_asset.ticker = "AAPL"
        self.controller.portfolio.assets = [mock_asset]

        mock_ask_ne.return_value = "AAPL"
        mock_confirm.return_value = True

        self.controller._remove_asset()

        self.controller.portfolio.remove_asset.assert_called_once_with("AAPL")
        self.controller.view.show_success.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._ask_non_empty")
    def test_remove_asset_cancelled(self, mock_ask_ne, mock_confirm):
        """Verify _remove_asset cancels when user declines."""
        mock_asset = MagicMock()
        mock_asset.ticker = "AAPL"
        self.controller.portfolio.assets = [mock_asset]

        mock_ask_ne.return_value = "AAPL"
        mock_confirm.return_value = False

        self.controller._remove_asset()

        self.controller.portfolio.remove_asset.assert_not_called()

    def test_remove_asset_empty_portfolio(self):
        """Verify _remove_asset shows info when portfolio empty."""
        self.controller.portfolio.assets = []

        self.controller._remove_asset()

        self.controller.view.show_info.assert_called_once()


class TestControllerViewPortfolio(unittest.TestCase):
    """Test viewing portfolio."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_view_portfolio_empty(self):
        """Verify _view_portfolio shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._view_portfolio()

        self.controller.view.show_info.assert_called_once()

    def test_view_portfolio_success(self):
        """Verify _view_portfolio displays portfolio."""
        mock_asset = MagicMock()
        mock_asset.ticker = "AAPL"
        mock_asset.sector = "Technology"
        mock_asset.asset_class = "EQUITY"
        mock_asset.quantity = 10
        mock_asset.purchase_price = 150.0
        mock_asset.get_current_price.return_value = 160.0
        mock_asset.get_current_value.return_value = 1600.0
        mock_asset.get_transaction_value.return_value = 1500.0
        mock_asset.get_profit_loss.return_value = 100.0

        self.controller.portfolio.assets = [mock_asset]
        self.controller.portfolio.currency = "EUR"
        self.controller.portfolio.name = "Test Portfolio"
        self.controller.portfolio.cash_balance = 500.0
        self.controller.analytics.get_total_invested_value.return_value = 1500.0
        self.controller.analytics.get_total_current_value.return_value = 1600.0
        self.controller.analytics.get_total_profit_loss.return_value = 100.0

        self.controller._view_portfolio()

        self.controller.view.show_portfolio_table.assert_called_once()
        self.controller.view.show_portfolio_summary.assert_called_once()


class TestControllerViewWeights(unittest.TestCase):
    """Test viewing weights."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_view_weights_empty_portfolio(self):
        """Verify _view_weights shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._view_weights()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    def test_view_weights_by_asset(self, mock_pick, mock_confirm):
        """Verify _view_weights displays asset weights."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]
        self.controller.portfolio.currency = "EUR"

        mock_pick.return_value = "Asset"
        mock_confirm.return_value = False

        self.controller.analytics.get_total_current_value.return_value = 10000.0
        self.controller.analytics.get_asset_weights.return_value = {"AAPL": 0.5, "MSFT": 0.5}

        self.controller._view_weights()

        self.controller.view.show_weights_table.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    def test_view_weights_with_pie_chart(self, mock_pick, mock_confirm):
        """Verify _view_weights plots pie chart when requested."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]
        self.controller.portfolio.currency = "EUR"

        mock_pick.return_value = "Sector"
        mock_confirm.return_value = True

        self.controller.analytics.get_total_current_value.return_value = 10000.0
        self.controller.analytics.get_weights_by_sector.return_value = {"Tech": 0.6, "Finance": 0.4}
        self.controller.view.plot_weights_pie.return_value = MagicMock()

        self.controller._view_weights()

        self.controller.view.plot_weights_pie.assert_called_once()
        self.controller.view.show_figure.assert_called_once()


class TestControllerPriceHistory(unittest.TestCase):
    """Test price history display."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_price_history_empty_portfolio(self):
        """Verify _price_history shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._price_history()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask")
    def test_price_history_all_assets(self, mock_ask, mock_pick, mock_confirm):
        """Verify _price_history fetches and plots all assets."""
        mock_asset1 = MagicMock()
        mock_asset1.ticker = "AAPL"
        mock_asset2 = MagicMock()
        mock_asset2.ticker = "MSFT"

        self.controller.portfolio.assets = [mock_asset1, mock_asset2]
        self.controller.portfolio.get_asset.return_value = mock_asset1

        mock_ask.return_value = ""
        mock_pick.return_value = "1y"
        mock_confirm.return_value = False

        import pandas as pd
        mock_asset1.get_historical_prices.return_value = pd.DataFrame({"Close": [150, 160]})

        self.controller._price_history()

        self.controller.view.plot_price_history.assert_called_once()


class TestControllerSharpeRatios(unittest.TestCase):
    """Test Sharpe ratio display."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_sharpe_ratios_empty_portfolio(self):
        """Verify _sharpe_ratios shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._sharpe_ratios()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask_float")
    def test_sharpe_ratios_per_asset(self, mock_ask_float, mock_pick, mock_confirm):
        """Verify _sharpe_ratios per asset mode."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_pick.side_effect = ["Per asset", "1y"]
        mock_ask_float.return_value = 0.0
        mock_confirm.return_value = False

        self.controller.analytics.get_risk_free_rate.return_value = 0.04
        self.controller.analytics.get_sharpe_ratio_per_asset.return_value = {"AAPL": 0.8}

        self.controller._sharpe_ratios()

        self.controller.view.show_sharpe_table.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask_float")
    def test_sharpe_ratios_portfolio_overall(self, mock_ask_float, mock_pick, mock_confirm):
        """Verify _sharpe_ratios portfolio mode."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_pick.side_effect = ["Portfolio (overall)", "1y"]
        mock_ask_float.return_value = 0.0

        self.controller.analytics.get_risk_free_rate.return_value = 0.04
        self.controller.analytics.get_portfolio_sharpe_ratio.return_value = 0.75

        self.controller._sharpe_ratios()

        self.controller.view.show_sharpe_table.assert_called_once()


class TestControllerCorrelationMatrix(unittest.TestCase):
    """Test correlation matrix display."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_correlation_matrix_insufficient_assets(self):
        """Verify _correlation_matrix requires 2+ assets."""
        self.controller.portfolio.assets = []

        self.controller._correlation_matrix()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller._pick")
    def test_correlation_matrix_success(self, mock_pick):
        """Verify _correlation_matrix computes and plots."""
        mock_asset1 = MagicMock()
        mock_asset2 = MagicMock()
        self.controller.portfolio.assets = [mock_asset1, mock_asset2]

        mock_pick.return_value = "1y"

        import pandas as pd
        mock_corr = pd.DataFrame([[1.0, 0.7], [0.7, 1.0]])
        self.controller.analytics.get_correlation_matrix.return_value = mock_corr

        self.controller.view.plot_correlation_heatmap.return_value = MagicMock()

        self.controller._correlation_matrix()

        self.controller.view.plot_correlation_heatmap.assert_called_once()
        self.controller.view.show_figure.assert_called_once()


class TestControllerOptimalWeights(unittest.TestCase):
    """Test optimal weights (Markowitz) calculation."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_optimal_weights_insufficient_assets(self):
        """Verify _optimal_weights requires 2+ assets."""
        self.controller.portfolio.assets = []

        self.controller._optimal_weights()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask_float")
    def test_optimal_weights_success(self, mock_ask_float, mock_pick, mock_confirm):
        """Verify _optimal_weights calculates and displays."""
        mock_asset1 = MagicMock()
        mock_asset2 = MagicMock()
        self.controller.portfolio.assets = [mock_asset1, mock_asset2]

        mock_pick.return_value = "1y"
        mock_ask_float.return_value = 0.0
        mock_confirm.return_value = True

        self.controller.analytics.get_risk_free_rate.return_value = 0.04
        opt_result = {
            "optimal_weights": {"AAPL": 0.6, "MSFT": 0.4},
            "expected_return": 0.12,
            "expected_volatility": 0.18,
        }
        self.controller.analytics.get_optimal_weights.return_value = opt_result
        self.controller.view.plot_optimal_weights_comparison.return_value = MagicMock()

        self.controller._optimal_weights()

        self.controller.view.show_optimal_weights_table.assert_called_once()
        self.controller.view.plot_optimal_weights_comparison.assert_called_once()


class TestControllerEfficientFrontier(unittest.TestCase):
    """Test efficient frontier calculation."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_efficient_frontier_insufficient_assets(self):
        """Verify _efficient_frontier requires 2+ assets."""
        self.controller.portfolio.assets = []

        self.controller._efficient_frontier()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller._pick")
    @patch("src.Controller._ask_positive_int")
    @patch("src.Controller._ask_float")
    def test_efficient_frontier_success(self, mock_ask_float, mock_ask_int, mock_pick):
        """Verify _efficient_frontier computes and plots."""
        mock_asset1 = MagicMock()
        mock_asset2 = MagicMock()
        self.controller.portfolio.assets = [mock_asset1, mock_asset2]

        mock_pick.return_value = "1y"
        mock_ask_int.return_value = 100
        mock_ask_float.return_value = 0.0

        self.controller.analytics.get_risk_free_rate.return_value = 0.04
        frontier = {
            "returns": [0.05, 0.10, 0.15],
            "volatilities": [0.10, 0.15, 0.20],
        }
        opt_weights = {
            "current_return": 0.10,
            "current_volatility": 0.15,
        }
        self.controller.analytics.get_efficient_frontier.return_value = frontier
        self.controller.analytics.get_optimal_weights.return_value = opt_weights
        self.controller.view.plot_efficient_frontier.return_value = MagicMock()

        self.controller._efficient_frontier()

        self.controller.view.plot_efficient_frontier.assert_called_once()


class TestControllerMonteCarlo(unittest.TestCase):
    """Test Monte Carlo simulation."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_monte_carlo_empty_portfolio(self):
        """Verify _monte_carlo shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._monte_carlo()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller._ask_positive_int")
    def test_monte_carlo_success(self, mock_ask_int):
        """Verify _monte_carlo runs simulation."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]
        self.controller.portfolio.currency = "EUR"

        mock_ask_int.side_effect = [15, 10000]

        import numpy as np
        sim_result = {"simulations": np.random.randn(10000, 250)}
        self.controller.analytics.simulate_portfolio.return_value = sim_result
        self.controller.analytics.get_total_current_value.return_value = 10000.0
        self.controller.view.plot_simulation.return_value = MagicMock()

        self.controller._monte_carlo()

        self.controller.analytics.simulate_portfolio.assert_called_once_with(
            years=15, num_simulations=10000
        )
        self.controller.view.plot_simulation.assert_called_once()


class TestControllerBenchmarkComparison(unittest.TestCase):
    """Test benchmark comparison."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_benchmark_comparison_empty_portfolio(self):
        """Verify _benchmark_comparison shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._benchmark_comparison()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask_float")
    @patch("src.Controller._ask_non_empty")
    def test_benchmark_comparison_success(self, mock_ask_ne, mock_ask_float,
                                          mock_pick, mock_confirm):
        """Verify _benchmark_comparison displays comparison."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_ask_ne.return_value = "ACWI"
        mock_pick.return_value = "1y"
        mock_ask_float.return_value = 0.0
        mock_confirm.return_value = False

        self.controller.analytics.get_risk_free_rate.return_value = 0.04
        benchmark_result = {
            "portfolio": {"annual_return": 0.12, "annual_volatility": 0.15},
            "benchmark": {"annual_return": 0.10, "annual_volatility": 0.14},
            "alpha": 0.02,
            "tracking_error": 0.05,
        }
        self.controller.analytics.get_benchmark_comparison.return_value = benchmark_result

        self.controller._benchmark_comparison()

        self.controller.view.show_benchmark_table.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    @patch("src.Controller._ask_float")
    @patch("src.Controller._ask_non_empty")
    def test_benchmark_comparison_no_data(self, mock_ask_ne, mock_ask_float,
                                          mock_pick, mock_confirm):
        """Verify _benchmark_comparison handles missing data."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_ask_ne.return_value = "BADTICKER"
        mock_pick.return_value = "1y"
        mock_ask_float.return_value = 0.0

        self.controller.analytics.get_benchmark_comparison.return_value = {}

        self.controller._benchmark_comparison()

        self.controller.view.show_error.assert_called_once()


class TestControllerRiskMetrics(unittest.TestCase):
    """Test risk metrics display."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    def test_risk_metrics_empty_portfolio(self):
        """Verify _risk_metrics shows info when empty."""
        self.controller.portfolio.assets = []

        self.controller._risk_metrics()

        self.controller.view.show_info.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    def test_risk_metrics_per_asset(self, mock_pick, mock_confirm):
        """Verify _risk_metrics per asset mode."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_pick.side_effect = ["Per asset", "1y"]
        mock_confirm.return_value = False

        self.controller.analytics.get_risk_metrics_per_asset.return_value = {
            "AAPL": {"var_95": 0.05, "cvar_95": 0.07}
        }

        self.controller._risk_metrics()

        self.controller.view.show_risk_metrics_table.assert_called_once()

    @patch("src.Controller.Confirm.ask")
    @patch("src.Controller._pick")
    def test_risk_metrics_portfolio_overall(self, mock_pick, mock_confirm):
        """Verify _risk_metrics portfolio mode."""
        mock_asset = MagicMock()
        self.controller.portfolio.assets = [mock_asset]

        mock_pick.side_effect = ["Portfolio (overall)", "1y"]
        mock_confirm.return_value = True

        portfolio_metrics = {"var_95": 0.06, "cvar_95": 0.08}
        self.controller.analytics.get_portfolio_risk_metrics.return_value = portfolio_metrics
        self.controller.view.plot_risk_metrics_bars.return_value = MagicMock()

        self.controller._risk_metrics()

        self.controller.view.show_risk_metrics_table.assert_called_once()
        self.controller.view.plot_risk_metrics_bars.assert_called_once()


class TestControllerManageCash(unittest.TestCase):
    """Test cash management."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()
        self.controller.portfolio = MagicMock(spec=Portfolio)
        self.controller.analytics = MagicMock(spec=PortfolioAnalytics)
        self.controller.view = MagicMock()

    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._pick")
    def test_manage_cash_deposit(self, mock_pick, mock_ask_float):
        """Verify _manage_cash deposits money."""
        self.controller.portfolio.cash_balance = 1000.0
        self.controller.portfolio.currency = "EUR"

        mock_pick.return_value = "Deposit"
        mock_ask_float.return_value = 500.0

        self.controller._manage_cash()

        self.controller.portfolio.deposit_cash.assert_called_once_with(500.0)
        self.controller.view.show_success.assert_called_once()

    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._pick")
    def test_manage_cash_withdraw(self, mock_pick, mock_ask_float):
        """Verify _manage_cash withdraws money."""
        self.controller.portfolio.cash_balance = 1000.0
        self.controller.portfolio.currency = "EUR"

        mock_pick.return_value = "Withdraw"
        mock_ask_float.return_value = 200.0

        self.controller._manage_cash()

        self.controller.portfolio.withdraw_cash.assert_called_once_with(200.0)
        self.controller.view.show_success.assert_called_once()

    @patch("src.Controller._ask_positive_float")
    @patch("src.Controller._pick")
    def test_manage_cash_error(self, mock_pick, mock_ask_float):
        """Verify _manage_cash handles errors."""
        self.controller.portfolio.cash_balance = 100.0
        self.controller.portfolio.currency = "EUR"

        mock_pick.return_value = "Withdraw"
        mock_ask_float.return_value = 500.0
        self.controller.portfolio.withdraw_cash.side_effect = ValueError("Insufficient funds")

        self.controller._manage_cash()

        self.controller.view.show_error.assert_called_once()


class TestControllerMainLoop(unittest.TestCase):
    """Test main interactive loop."""

    def setUp(self):
        """Create controller with mock portfolio."""
        self.controller = Controller()

    @patch("src.Controller.Controller._init_portfolio")
    @patch("src.Controller.Controller._print_menu")
    @patch("src.Controller._ask")
    def test_run_exit_directly(self, mock_ask, mock_print_menu, mock_init):
        """Verify run exits when user chooses 0."""
        mock_ask.return_value = "0"

        self.controller.run()

        mock_init.assert_called_once()
        mock_print_menu.assert_called_once()

    @patch("src.Controller.Controller._print_menu")
    @patch("src.Controller.Controller._add_asset")
    @patch("src.Controller._ask")
    def test_run_dispatch_add_asset(self, mock_ask, mock_add, mock_print_menu):
        """Verify run dispatches to _add_asset."""
        mock_ask.side_effect = ["1", "0"]

        with patch.object(self.controller, "_init_portfolio"):
            self.controller.run()

        mock_add.assert_called_once()

    @patch("src.Controller.Controller._print_menu")
    @patch("src.Controller._ask")
    def test_run_invalid_choice(self, mock_ask, mock_print_menu):
        """Verify run handles invalid menu choices."""
        with patch.object(self.controller, "_init_portfolio"):
            with patch.object(self.controller, "view"):
                mock_ask.side_effect = ["99", "0"]
                self.controller.run()
                self.controller.view.show_error.assert_called()


if __name__ == "__main__":
    unittest.main()
