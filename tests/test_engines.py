"""Unit tests for the pure risk engines (no DB/Redis required).

Covers the optimizer, correlation regime, stress engine, and the bootstrap
Monte Carlo path — previously untested surface area.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from app.models import Position
from app.services.correlation_service import compute_correlation_regime
from app.services.optimizer_service import run_portfolio_optimizer
from app.services.stress_engine import calculate_beta, run_stress_tests
from app.services.var_engine import compute_monte_carlo_var


def test_optimizer_returns_frontier_and_actions() -> None:
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame(
        {
            "AAA": rng.normal(0.0006, 0.020, n),
            "BBB": rng.normal(0.0004, 0.030, n),
            "CCC": rng.normal(0.0003, 0.015, n),
        }
    )
    tickers = ["AAA", "BBB", "CCC"]
    current_weights = np.array([0.5, 0.3, 0.2])
    prices = {"AAA": Decimal("100"), "BBB": Decimal("50"), "CCC": Decimal("25")}
    qty = {"AAA": Decimal("500"), "BBB": Decimal("600"), "CCC": Decimal("800")}

    res = run_portfolio_optimizer(df, tickers, current_weights, Decimal("100000"), prices, qty)

    assert res["current_var_95"] >= 0
    assert res["optimized_var_95"] >= 0
    assert len(res["efficient_frontier"]) == 20
    assert len(res["rebalancing_actions"]) == 3
    assert {a["action"] for a in res["rebalancing_actions"]}.issubset({"BUY", "SELL", "HOLD"})
    # Optimized weights respect the [0.05, 0.60] bounds and sum to ~1.
    assert abs(sum(res["optimized_weights"]) - 1.0) < 1e-6
    assert all(0.04 <= w <= 0.61 for w in res["optimized_weights"])


def test_correlation_regime_detects_stress() -> None:
    rng = np.random.default_rng(2)
    n = 252
    a = rng.normal(0, 0.01, n)
    b = rng.normal(0, 0.01, n)  # independent over the full window
    shared = rng.normal(0, 0.02, 30)  # last 30 days move together
    a[-30:] = shared
    b[-30:] = shared
    df = pd.DataFrame({"AAA": a, "BBB": b})

    res = compute_correlation_regime(df)

    assert res["regime"] == "STRESS"
    assert res["correlation_spike"] > 0.20
    assert res["most_correlated_pair"]["ticker_a"] in {"AAA", "BBB"}
    assert set(res["matrix_30d"].keys()) == {"AAA", "BBB"}


def test_correlation_regime_handles_single_asset() -> None:
    df = pd.DataFrame({"AAA": [0.01, -0.02, 0.015, 0.0]})
    res = compute_correlation_regime(df)
    assert res["regime"] == "NORMAL"
    assert res["avg_correlation_30d"] == 0.0


def test_beta_recovers_known_slope() -> None:
    rng = np.random.default_rng(3)
    market = rng.normal(0, 0.01, 200)
    asset = 1.2 * market + rng.normal(0, 0.0005, 200)
    assert abs(calculate_beta(asset, market) - 1.2) < 0.1


def test_stress_losses_increase_with_severity() -> None:
    rng = np.random.default_rng(4)
    market = rng.normal(0, 0.01, 120)
    asset = 1.2 * market + rng.normal(0, 0.0005, 120)
    positions = [Position(ticker="AAA", quantity=Decimal("10"), purchase_price=Decimal("100"))]
    losses = run_stress_tests(
        positions,
        {"AAA": asset},
        market,
        {"AAA": Decimal("1000")},
    )
    assert losses["severe"] > losses["moderate"] > losses["mild"] > 0


def test_monte_carlo_bootstrap_is_reproducible_and_captures_tails() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 0.01, size=(252, 1))
    returns[::20] = -0.15  # periodic crash days -> left skew, fat tail
    weights = np.array([1.0])

    first = compute_monte_carlo_var(returns, weights, Decimal("100000"))
    second = compute_monte_carlo_var(returns, weights, Decimal("100000"))

    # Deterministic seed -> identical results across calls.
    assert first.mc_var_95 == second.mc_var_95
    assert len(first.histogram) == 50
    # 99% loss is at least the 95% loss, and the bootstrap surfaces the left skew.
    assert first.mc_var_99 >= first.mc_var_95
    assert first.skewness < 0
