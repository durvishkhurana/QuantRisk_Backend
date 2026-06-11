from __future__ import annotations

from decimal import Decimal
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.services.var_engine import compute_var_cvar


def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(weights.T @ cov @ weights)


def _min_variance_for_target(
    cov: np.ndarray,
    mean_returns: np.ndarray,
    target_return: float,
    n_assets: int,
) -> tuple[float, np.ndarray]:
    bounds = [(0.05, 0.60) for _ in range(n_assets)]
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, tr=target_return: float(w @ mean_returns) - tr},
    ]
    x0 = np.full(n_assets, 1.0 / n_assets)

    def objective(w: np.ndarray) -> float:
        return _portfolio_variance(w, cov)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        return float("inf"), x0
    return _portfolio_variance(result.x, cov), result.x


def _optimize_min_variance(cov: np.ndarray, n_assets: int) -> np.ndarray:
    bounds = [(0.05, 0.60) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.full(n_assets, 1.0 / n_assets)

    def objective(w: np.ndarray) -> float:
        return _portfolio_variance(w, cov)

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    return x0


def _var_for_weights(
    asset_returns: np.ndarray,
    weights: np.ndarray,
    portfolio_value: Decimal,
) -> Decimal:
    port_series = asset_returns @ weights
    var_95, _, _ = compute_var_cvar(port_series, portfolio_value, 0.05, 0.01)
    return var_95


def run_portfolio_optimizer(
    returns_df: pd.DataFrame,
    tickers: list[str],
    current_weights: np.ndarray,
    portfolio_value: Decimal,
    prices: dict[str, Decimal],
    quantities: dict[str, Decimal],
) -> dict:
    asset_returns = returns_df[tickers].dropna().to_numpy()
    if asset_returns.shape[0] < 30:
        raise ValueError("Insufficient return history for optimization")

    cov = np.cov(asset_returns, rowvar=False)
    mean_returns = np.mean(asset_returns, axis=0)
    n_assets = len(tickers)

    current_var = _var_for_weights(asset_returns, current_weights, portfolio_value)
    opt_weights = _optimize_min_variance(cov, n_assets)
    optimized_var = _var_for_weights(asset_returns, opt_weights, portfolio_value)

    reduction = 0.0
    if float(current_var) > 0:
        reduction = float((current_var - optimized_var) / current_var * Decimal("100"))

    rebalancing: list[dict] = []
    for i, ticker in enumerate(tickers):
        cw = float(current_weights[i])
        tw = float(opt_weights[i])
        delta_w = tw - cw
        if abs(delta_w) < 0.005:
            action = "HOLD"
        elif delta_w > 0:
            action = "BUY"
        else:
            action = "SELL"
        price = float(prices.get(ticker, Decimal("1")))
        if price <= 0:
            price = 1.0
        delta_value = delta_w * float(portfolio_value)
        delta_shares = delta_value / price
        rebalancing.append(
            {
                "ticker": ticker,
                "current_weight": round(cw, 4),
                "target_weight": round(tw, 4),
                "action": action,
                "delta_shares_approx": round(delta_shares, 2),
            }
        )

    ret_min, ret_max = float(np.min(mean_returns)), float(np.max(mean_returns))
    if ret_max <= ret_min:
        targets = [ret_min] * 20
    else:
        targets = np.linspace(ret_min, ret_max, 20)

    frontier: list[dict] = []
    for target in targets:
        variance, weights = _min_variance_for_target(cov, mean_returns, float(target), n_assets)
        frontier.append(
            {
                "target_return": float(target),
                "min_variance": float(variance),
                "weights": [float(w) for w in weights],
            }
        )

    return {
        "current_var_95": current_var,
        "optimized_var_95": optimized_var,
        "var_reduction_pct": round(reduction, 2),
        "rebalancing_actions": rebalancing,
        "efficient_frontier": frontier,
        "optimized_weights": opt_weights,
    }
