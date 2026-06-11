from decimal import Decimal
import numpy as np
from sklearn.linear_model import LinearRegression
from app.models import Position


def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    min_len = min(len(asset_returns), len(market_returns))
    if min_len < 5:
        return 1.0
    model = LinearRegression().fit(market_returns[:min_len].reshape(-1, 1), asset_returns[:min_len])
    return float(model.coef_[0])


def run_stress_tests(
    positions: list[Position],
    returns_by_ticker: dict[str, np.ndarray],
    market_returns: np.ndarray,
    position_values: dict[str, Decimal],
) -> dict[str, Decimal]:
    shocks = {"mild": -0.10, "moderate": -0.20, "severe": -0.30}
    losses: dict[str, Decimal] = {}
    for name, shock in shocks.items():
        total = Decimal("0")
        for pos in positions:
            asset_returns = returns_by_ticker.get(pos.ticker, np.array([]))
            beta = calculate_beta(asset_returns, market_returns) if len(asset_returns) and len(market_returns) else 1.0
            initial = position_values[pos.ticker]
            stressed = initial * Decimal(str(1 + beta * shock))
            total += initial - stressed
        losses[name] = Decimal(str(round(float(total), 2)))
    return losses
