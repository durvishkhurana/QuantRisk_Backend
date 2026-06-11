import numpy as np
from decimal import Decimal
from app.services.var_engine import compute_monte_carlo_var, compute_var_cvar


def test_var_cvar_outputs_non_negative_losses() -> None:
    returns = np.array([0.02, -0.01, 0.015, -0.03, 0.01, -0.02, 0.005], dtype=float)
    var_95, var_99, cvar_95 = compute_var_cvar(returns, Decimal("100000"), 0.05, 0.01)
    assert var_95 >= 0
    assert var_99 >= 0
    assert cvar_95 >= 0


def test_monte_carlo_var_reproducible() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, size=(252, 3))
    weights = np.array([0.4, 0.35, 0.25])
    first = compute_monte_carlo_var(returns, weights, Decimal("100000"))
    second = compute_monte_carlo_var(returns, weights, Decimal("100000"))
    assert first.mc_var_95 == second.mc_var_95
    assert first.mc_var_99 == second.mc_var_99
    assert len(first.histogram) == 50
