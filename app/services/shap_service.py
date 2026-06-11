from decimal import Decimal
import numpy as np
from sklearn.linear_model import LinearRegression
from app.models import Position


def compute_shap_like_attribution(
    scenarios: np.ndarray,
    weights: np.ndarray,
    portfolio_value: Decimal,
    positions: list[Position],
) -> dict[str, float]:
    if scenarios.size == 0 or weights.size == 0:
        return {}
    y = np.abs(scenarios @ weights[: scenarios.shape[1]]) * float(portfolio_value)
    reg = LinearRegression().fit(scenarios, y)
    contrib = reg.coef_ * weights[: scenarios.shape[1]] * float(portfolio_value)
    return {positions[idx].ticker: float(round(contrib[idx], 2)) for idx in range(min(len(positions), len(contrib)))}
