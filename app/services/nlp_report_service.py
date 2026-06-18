from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import get_settings

settings = get_settings()

NarrativeSource = Literal["anthropic", "template"]


@dataclass(frozen=True)
class NarrativeResult:
    text: str
    source: NarrativeSource


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _top_shap(risk_result: dict) -> tuple[str, float]:
    shap = risk_result.get("shap_attribution") or []
    if not shap:
        return "your largest position", 0.0
    first = shap[0]
    if isinstance(first, dict):
        return str(first.get("ticker", "?")), _as_float(first.get("pct_of_var"))
    return str(getattr(first, "ticker", "?")), _as_float(getattr(first, "pct_of_var", 0))


def _fallback_narrative(risk_result: dict) -> NarrativeResult:
    var_95 = _as_float(risk_result.get("var_95"))
    cvar_95 = _as_float(risk_result.get("cvar_95"))
    portfolio_value = _as_float(risk_result.get("portfolio_value"))
    utilization = _as_float(risk_result.get("margin_utilization")) * 100
    status = str(risk_result.get("margin_status", "OK"))
    top_ticker, top_pct = _top_shap(risk_result)
    stress = risk_result.get("stress_tests") or {}
    severe = stress.get("severe") or {}
    severe_pct = _as_float(severe.get("pct") if isinstance(severe, dict) else getattr(severe, "pct", 0))
    mc = risk_result.get("monte_carlo") or {}
    mc_var = _as_float(mc.get("var_95") if isinstance(mc, dict) else getattr(mc, "var_95", 0))

    margin_clause = (
        "Margin utilization is elevated relative to your limit — review exposure."
        if status in {"WARNING", "BREACH"}
        else "Margin utilization is within your configured limit."
    )

    text = (
        f"Portfolio value is about ${portfolio_value:,.0f} with 95% one-day VaR near ${var_95:,.0f} "
        f"(CVaR ${cvar_95:,.0f}); status is {status} at {utilization:.1f}% of the VaR cap. "
        f"SHAP-style attribution points to {top_ticker} as the dominant driver (~{abs(top_pct):.1f}% of VaR). "
        f"{margin_clause} "
        f"A severe stress scenario implies roughly {severe_pct:.1f}% drawdown"
        + (f"; Monte Carlo 95% VaR is about ${mc_var:,.0f}." if mc_var > 0 else ".")
    )
    return NarrativeResult(text=text, source="template")


async def generate_risk_narrative(risk_result: dict) -> NarrativeResult:
    api_key = (settings.anthropic_api_key or "").strip()
    if not api_key:
        return _fallback_narrative(risk_result)

    system_prompt = (
        "You are a concise financial risk analyst. Given structured portfolio risk data, write exactly 3 sentences "
        "in plain English for a non-expert portfolio manager. Sentence 1: overall risk level and VaR. "
        "Sentence 2: biggest risk driver (from SHAP attribution). Sentence 3: margin status or stress test warning. "
        "Never use jargon without defining it. Never hallucinate data not in the input."
    )
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 220,
        "system": system_prompt,
        "messages": [{"role": "user", "content": json.dumps(risk_result, default=str)}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            blocks = data.get("content") or []
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            if text.strip():
                return NarrativeResult(text=text.strip(), source="anthropic")
    except Exception:
        pass
    return _fallback_narrative(risk_result)
