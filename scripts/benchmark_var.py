from __future__ import annotations

import statistics
import sys
import timeit
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.var_engine import compute_monte_carlo_var, compute_var_cvar


ITERATIONS = 10
LOOKBACK_DAYS = 252
N_MONTE_CARLO_PATHS = 10_000
CONFIDENCE_95 = 0.05
CONFIDENCE_99 = 0.01


@dataclass(frozen=True)
class BenchmarkPosition:
    ticker: str
    quantity: Decimal
    purchase_price: Decimal


@dataclass(frozen=True)
class BenchmarkData:
    positions: list[BenchmarkPosition]
    prices: dict[str, Decimal]
    returns_matrix: np.ndarray
    market_returns: np.ndarray


def build_fake_market_data() -> BenchmarkData:
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    starting_prices = np.array([190.0, 410.0, 880.0, 205.0, 115.0])
    daily_mu = np.array([0.00045, 0.00040, 0.00060, 0.00025, 0.00020])
    daily_vol = np.array([0.018, 0.016, 0.028, 0.014, 0.017])

    base_corr = np.full((len(tickers), len(tickers)), 0.35)
    np.fill_diagonal(base_corr, 1.0)
    covariance = np.outer(daily_vol, daily_vol) * base_corr

    returns = rng.multivariate_normal(daily_mu, covariance, LOOKBACK_DAYS)
    price_paths = starting_prices * np.cumprod(1 + returns, axis=0)

    market_returns = returns @ np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    market_returns += rng.normal(0.0001, 0.004, LOOKBACK_DAYS)

    positions = [
        BenchmarkPosition("AAPL", Decimal("120"), Decimal("185")),
        BenchmarkPosition("MSFT", Decimal("80"), Decimal("390")),
        BenchmarkPosition("NVDA", Decimal("35"), Decimal("820")),
        BenchmarkPosition("JPM", Decimal("150"), Decimal("195")),
        BenchmarkPosition("XOM", Decimal("220"), Decimal("105")),
    ]
    latest_prices = {
        ticker: Decimal(str(round(float(price_paths[-1, idx]), 4)))
        for idx, ticker in enumerate(tickers)
    }

    return BenchmarkData(
        positions=positions,
        prices=latest_prices,
        returns_matrix=returns,
        market_returns=market_returns,
    )


DATA = build_fake_market_data()


def prepare_portfolio() -> tuple[Decimal, dict[str, Decimal], np.ndarray]:
    position_values = {
        position.ticker: DATA.prices[position.ticker] * position.quantity
        for position in DATA.positions
    }
    portfolio_value = sum(position_values.values(), Decimal("0"))
    weights = np.array(
        [float(position_values[position.ticker] / portfolio_value) for position in DATA.positions],
        dtype=float,
    )
    return portfolio_value, position_values, weights


def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    min_len = min(len(asset_returns), len(market_returns))
    if min_len < 5:
        return 1.0
    x = market_returns[:min_len]
    y = asset_returns[:min_len]
    design = np.column_stack([np.ones(min_len), x])
    _, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(beta)


def run_stress_tests(
    returns_by_ticker: dict[str, np.ndarray],
    market_returns: np.ndarray,
    position_values: dict[str, Decimal],
) -> dict[str, Decimal]:
    shocks = {"mild": -0.10, "moderate": -0.20, "severe": -0.30}
    losses: dict[str, Decimal] = {}
    for name, shock in shocks.items():
        total = Decimal("0")
        for position in DATA.positions:
            beta = calculate_beta(returns_by_ticker[position.ticker], market_returns)
            initial = position_values[position.ticker]
            stressed = initial * Decimal(str(1 + beta * shock))
            total += initial - stressed
        losses[name] = Decimal(str(round(float(total), 2)))
    return losses


def compute_linear_shap_attribution(
    scenarios: np.ndarray,
    weights: np.ndarray,
    portfolio_value: Decimal,
) -> dict[str, float]:
    y = np.abs(scenarios @ weights[: scenarios.shape[1]]) * float(portfolio_value)
    design = np.column_stack([np.ones(scenarios.shape[0]), scenarios])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0][1:]
    contributions = coefficients * weights[: scenarios.shape[1]] * float(portfolio_value)
    return {
        position.ticker: float(round(contributions[idx], 2))
        for idx, position in enumerate(DATA.positions)
    }


def benchmark_historical_var() -> None:
    portfolio_value, _, weights = prepare_portfolio()
    portfolio_returns = DATA.returns_matrix @ weights
    compute_var_cvar(portfolio_returns, portfolio_value, CONFIDENCE_95, CONFIDENCE_99)


def benchmark_monte_carlo_var() -> None:
    portfolio_value, _, weights = prepare_portfolio()
    compute_monte_carlo_var(
        DATA.returns_matrix,
        weights,
        portfolio_value,
        n_simulations=N_MONTE_CARLO_PATHS,
    )


def benchmark_full_risk_pipeline() -> None:
    portfolio_value, position_values, weights = prepare_portfolio()
    portfolio_returns = DATA.returns_matrix @ weights
    compute_var_cvar(portfolio_returns, portfolio_value, CONFIDENCE_95, CONFIDENCE_99)

    returns_by_ticker = {
        position.ticker: DATA.returns_matrix[:, idx]
        for idx, position in enumerate(DATA.positions)
    }
    run_stress_tests(returns_by_ticker, DATA.market_returns, position_values)
    compute_linear_shap_attribution(DATA.returns_matrix, weights, portfolio_value)


def run_case(label: str, func) -> tuple[str, float, float]:
    timer = timeit.Timer(func)
    samples_ms = [sample * 1000 for sample in timer.repeat(repeat=ITERATIONS, number=1)]
    mean_ms = statistics.mean(samples_ms)
    p95_ms = statistics.quantiles(samples_ms, n=20, method="inclusive")[18]
    return label, mean_ms, p95_ms


def main() -> None:
    cases = [
        ("Historical VaR (5 positions)", benchmark_historical_var),
        ("Monte Carlo 10k paths", benchmark_monte_carlo_var),
        ("Full risk pipeline", benchmark_full_risk_pipeline),
    ]
    results = [run_case(label, func) for label, func in cases]

    print("VaR Benchmark Summary")
    print(f"Iterations: {ITERATIONS}")
    print(f"Lookback window: {LOOKBACK_DAYS} days")
    print()
    print(f"{'Operation':<32} {'Mean':>10} {'p95':>10}")
    print("-" * 54)
    for label, mean_ms, p95_ms in results:
        print(f"{label:<32} {mean_ms:>9.2f}ms {p95_ms:>9.2f}ms")


if __name__ == "__main__":
    main()
