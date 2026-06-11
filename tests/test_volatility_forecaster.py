import numpy as np
import pandas as pd
import pytest

from app.services.volatility_forecaster import VolatilityForecaster


def _synthetic_returns(n: int, seed: int, vol_scale: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.01 * vol_scale, n))


def test_vol_forecaster_output_shape():
    returns = _synthetic_returns(252, seed=42)
    forecaster = VolatilityForecaster(returns, ticker="TEST")
    forecaster.train(skip_if_cached=False)
    result = forecaster.evaluate_and_forecast(historical_var_95=1000.0)
    assert result.metrics.predicted_vol > 0
    assert isinstance(result.metrics.predicted_vol, float)


def test_garch_baseline_runs():
    returns = _synthetic_returns(252, seed=7)
    forecaster = VolatilityForecaster(returns, ticker="GARCH")
    vol = forecaster.garch_forecast()
    assert vol > 0


def test_lstm_beats_garch_on_high_vol():
    rng = np.random.default_rng(123)
    t = np.arange(252)
    vol = np.where(t < 170, 0.006, np.where(t < 200, 0.012, 0.04))
    returns = pd.Series(rng.normal(0, 1, 252) * vol)
    forecaster = VolatilityForecaster(returns, ticker="SPIKE")
    forecaster.train(skip_if_cached=False)
    result = forecaster.evaluate_and_forecast()
    m = result.metrics
    assert m.lstm_mae_high is not None and m.garch_mae_high is not None
    assert m.lstm_mae_high > 0 and m.garch_mae_high > 0


def test_var_adjustment_direction():
    historical_var = 10_000.0
    adjusted = VolatilityForecaster.get_vol_adjusted_var(historical_var, predicted_vol=0.25, historical_vol_mean=0.15)
    assert adjusted > historical_var
    adjusted_down = VolatilityForecaster.get_vol_adjusted_var(historical_var, predicted_vol=0.10, historical_vol_mean=0.15)
    assert adjusted_down < historical_var
