from dataclasses import dataclass
from decimal import Decimal
import numpy as np
from scipy import stats


@dataclass
class MonteCarloResult:
    mc_var_95: Decimal
    mc_var_99: Decimal
    mc_cvar_95: Decimal
    skewness: float
    kurtosis: float
    histogram: list[dict]
    sim_returns: np.ndarray


def compute_monte_carlo_var(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    portfolio_value: Decimal,
    n_simulations: int = 10000,
    horizon_days: int = 1,
) -> MonteCarloResult:
    np.random.seed(42)
    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 2:
        raise ValueError("returns_matrix must be 2D with at least 2 rows")

    mu = np.mean(returns_matrix, axis=0)
    cov = np.cov(returns_matrix, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    simulated = np.random.multivariate_normal(mu, cov, n_simulations)
    if horizon_days != 1:
        simulated = simulated * np.sqrt(horizon_days)
    sim_portfolio = simulated @ weights

    q95 = float(np.quantile(sim_portfolio, 0.05))
    q99 = float(np.quantile(sim_portfolio, 0.01))
    pv = float(portfolio_value)
    mc_var_95 = Decimal(str(round(-q95 * pv, 2)))
    mc_var_99 = Decimal(str(round(-q99 * pv, 2)))
    tail = sim_portfolio[sim_portfolio <= q95]
    mc_cvar_95 = Decimal(str(round(-float(np.mean(tail)) * pv, 2))) if len(tail) else Decimal("0")

    skewness = float(stats.skew(sim_portfolio))
    kurtosis = float(stats.kurtosis(sim_portfolio, fisher=False))

    counts, edges = np.histogram(sim_portfolio, bins=50)
    histogram = [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    return MonteCarloResult(
        mc_var_95=mc_var_95,
        mc_var_99=mc_var_99,
        mc_cvar_95=mc_cvar_95,
        skewness=skewness,
        kurtosis=kurtosis,
        histogram=histogram,
        sim_returns=sim_portfolio,
    )


def compute_var_cvar(
    portfolio_returns: np.ndarray,
    portfolio_value: Decimal,
    confidence_95: float,
    confidence_99: float,
) -> tuple[Decimal, Decimal, Decimal]:
    var_95 = Decimal(str(round(-np.quantile(portfolio_returns, confidence_95) * float(portfolio_value), 2)))
    var_99 = Decimal(str(round(-np.quantile(portfolio_returns, confidence_99) * float(portfolio_value), 2)))
    var_fraction = float(var_95 / portfolio_value) if portfolio_value > 0 else 0.0
    tail = portfolio_returns[portfolio_returns <= -var_fraction]
    cvar = float(-np.mean(tail) * float(portfolio_value)) if len(tail) else 0.0
    cvar_95 = Decimal(str(round(cvar, 2)))
    return var_95, var_99, cvar_95
