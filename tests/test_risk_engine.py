import numpy as np
from app.services.risk import _calculate_beta


def test_calculate_beta_tracks_linear_relationship() -> None:
    market = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02], dtype=float)
    asset = 1.5 * market + 0.001
    beta = _calculate_beta(asset, market)
    assert 1.45 <= beta <= 1.55


def test_calculate_beta_fallback_for_small_samples() -> None:
    market = np.array([0.01, 0.02, -0.01], dtype=float)
    asset = np.array([0.02, 0.01, -0.01], dtype=float)
    beta = _calculate_beta(asset, market)
    assert beta == 1.0
