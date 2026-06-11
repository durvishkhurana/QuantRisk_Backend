from __future__ import annotations

import json
import httpx
from app.config import get_settings

settings = get_settings()


def _fallback_narrative(risk_result: dict) -> str:
    var_95 = risk_result.get("var_95", 0)
    status = risk_result.get("margin_status", "OK")
    shap = risk_result.get("shap_attribution") or []
    top = shap[0]["ticker"] if shap else "the largest holding"
    stress = risk_result.get("stress_tests") or {}
    severe = stress.get("severe", {})
    severe_pct = severe.get("pct", 0)
    return (
        f"Your portfolio's 95% Value at Risk (VaR) is about ${var_95:,.0f}, with margin status {status}. "
        f"The largest contributor to that risk is {top}, based on attribution analysis. "
        f"A severe stress scenario could imply roughly {severe_pct}% of portfolio value in losses."
    )


async def generate_risk_narrative(risk_result: dict) -> str:
    api_key = getattr(settings, "anthropic_api_key", None)
    if not api_key:
        return _fallback_narrative(risk_result)

    system_prompt = (
        "You are a concise financial risk analyst. Given structured portfolio risk data, write exactly 3 sentences "
        "in plain English for a non-expert portfolio manager. Sentence 1: overall risk level and VaR. "
        "Sentence 2: biggest risk driver (from SHAP attribution). Sentence 3: what changed vs last computation "
        "or a stress test warning. Never use jargon without defining it. Never hallucinate data not in the input."
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
                return text.strip()
    except Exception:
        pass
    return _fallback_narrative(risk_result)
