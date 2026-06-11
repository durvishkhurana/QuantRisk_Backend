from decimal import Decimal
from app.config import get_settings

settings = get_settings()


def evaluate_margin(var_95: Decimal, margin_limit: Decimal, portfolio_value: Decimal) -> tuple[Decimal, str]:
    denominator = Decimal(margin_limit) * portfolio_value
    utilization = (var_95 / denominator) if denominator > 0 else Decimal("0")
    if utilization > Decimal("1.0"):
        status = "BREACH"
    elif utilization > Decimal(str(settings.margin_warning_threshold)):
        status = "WARNING"
    else:
        status = "NORMAL"
    return utilization.quantize(Decimal("0.0001")), status
