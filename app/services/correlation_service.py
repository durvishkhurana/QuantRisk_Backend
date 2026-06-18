from __future__ import annotations

import numpy as np
import pandas as pd


def _avg_upper_triangle(corr: pd.DataFrame) -> float:
    if corr.empty or corr.shape[0] < 2:
        return 0.0
    values = corr.where(~pd.DataFrame(np.eye(len(corr), dtype=bool), index=corr.index, columns=corr.columns)).stack()
    if values.empty:
        return 0.0
    return float(values.mean())


def _most_correlated_pair(corr: pd.DataFrame) -> dict:
    tickers = list(corr.columns)
    best = (tickers[0], tickers[0], 0.0)
    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b = tickers[j]
            val = float(corr.loc[a, b])
            if val > best[2]:
                best = (a, b, val)
    return {"ticker_a": best[0], "ticker_b": best[1], "correlation_30d": round(best[2], 4)}


def compute_correlation_regime(returns_matrix_df: pd.DataFrame) -> dict:
    if returns_matrix_df.empty or returns_matrix_df.shape[1] < 2:
        return {
            "avg_correlation_30d": 0.0,
            "avg_correlation_252d": 0.0,
            "correlation_spike": 0.0,
            "regime": "NORMAL",
            "most_correlated_pair": {"ticker_a": "", "ticker_b": "", "correlation_30d": 0.0},
            "matrix_30d": {},
        }

    tail_30 = returns_matrix_df.tail(30)
    corr_30 = tail_30.corr()
    corr_252 = returns_matrix_df.corr()

    avg_30 = _avg_upper_triangle(corr_30)
    avg_252 = _avg_upper_triangle(corr_252)
    spike = avg_30 - avg_252

    if spike > 0.20:
        regime = "STRESS"
    elif spike > 0.10:
        regime = "ELEVATED"
    else:
        regime = "NORMAL"

    matrix_30d = {
        t: {other: round(float(corr_30.loc[t, other]), 4) for other in corr_30.columns}
        for t in corr_30.columns
    }

    return {
        "avg_correlation_30d": round(avg_30, 4),
        "avg_correlation_252d": round(avg_252, 4),
        "correlation_spike": round(spike, 4),
        "regime": regime,
        "most_correlated_pair": _most_correlated_pair(corr_30),
        "matrix_30d": matrix_30d,
    }
