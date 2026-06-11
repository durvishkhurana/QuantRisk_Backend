from decimal import Decimal
import numpy as np
from sklearn.linear_model import LinearRegression
from app.models import Position


def compute_kernel_shap(
    scenarios: np.ndarray,
    weights: np.ndarray,
    portfolio_value: Decimal,
    positions: list[Position],
) -> dict[str, float]:
    if scenarios.size == 0 or weights.size == 0:
        return {}
    if scenarios.shape[1] != len(positions):
        return {}

    y = np.abs(scenarios @ weights) * float(portfolio_value)
    model = LinearRegression().fit(scenarios, y)

    # Use a representative "current" market scenario (latest observation).
    x_current = scenarios[-1:].copy()
    bg_size = min(25, len(scenarios))
    background = scenarios[-bg_size:]

    try:
        import shap  # local import: optional heavy dependency

        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(x_current, nsamples=min(100, 2 * scenarios.shape[1] + 20))
        values = np.array(shap_values)[0]
    except Exception:
        # Fallback when SHAP is unavailable or unstable in environment.
        values = model.coef_ * x_current[0]

    result: dict[str, float] = {}
    for idx, pos in enumerate(positions):
        result[pos.ticker] = float(round(values[idx] * float(portfolio_value), 2))
    return result
