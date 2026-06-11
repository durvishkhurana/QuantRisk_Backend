from decimal import Decimal
import numpy as np
from app.models import Position


def compute_position_values(positions: list[Position], price_map: dict[str, Decimal]) -> dict[str, Decimal]:
    values: dict[str, Decimal] = {}
    for pos in positions:
        values[pos.ticker] = price_map[pos.ticker] * Decimal(pos.quantity)
    return values


def compute_portfolio_value(position_values: dict[str, Decimal]) -> Decimal:
    return sum(position_values.values(), Decimal("0"))


def compute_weights(positions: list[Position], position_values: dict[str, Decimal], portfolio_value: Decimal) -> np.ndarray:
    if portfolio_value <= 0:
        return np.array([])
    return np.array([float(position_values[p.ticker] / portfolio_value) for p in positions], dtype=float)
