import uuid
from datetime import date, datetime
from decimal import Decimal
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ErrorResponse(BaseModel):
    error: str
    message: str
    status: int


class AuthRegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class AuthLoginIn(BaseModel):
    email: EmailStr
    password: str


class AuthOut(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    token: str
    expires_at: datetime


class PortfolioCreateIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    margin_limit: Decimal = Field(default=Decimal("0.05"))


class PortfolioPatchIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    margin_limit: Decimal | None = Field(default=None)



class PortfolioOut(BaseModel):
    portfolio_id: uuid.UUID
    name: str
    margin_limit: Decimal
    positions_count: int
    total_value: Decimal
    latest_risk: dict | None = None


class PositionCreateIn(BaseModel):
    ticker: str = Field(min_length=1, max_length=10)
    quantity: Decimal
    purchase_price: Decimal


class PositionPatchIn(BaseModel):
    quantity: Decimal | None = None
    purchase_price: Decimal | None = None


class PositionOut(BaseModel):
    position_id: uuid.UUID
    ticker: str
    quantity: Decimal
    purchase_price: Decimal
    current_price: Decimal
    market_value: Decimal
    sector: str | None = None


class StressScenarioOut(BaseModel):
    loss: Decimal
    pct: Decimal


class ShapContributionOut(BaseModel):
    ticker: str
    contribution: Decimal
    pct_of_var: Decimal


class MonteCarloOut(BaseModel):
    var_95: Decimal
    var_99: Decimal
    cvar_95: Decimal
    skewness: float
    kurtosis: float
    n_simulations: int = 10000
    histogram: list[dict] | None = None


class RebalancingAction(BaseModel):
    ticker: str
    current_weight: float
    target_weight: float
    action: str
    delta_shares_approx: float


class EfficientFrontierPoint(BaseModel):
    target_return: float
    min_variance: float
    weights: list[float] | None = None


class OptimizationResult(BaseModel):
    current_var_95: Decimal
    optimized_var_95: Decimal
    var_reduction_pct: float
    rebalancing_actions: list[RebalancingAction]
    efficient_frontier: list[EfficientFrontierPoint]


class CorrelationPairOut(BaseModel):
    ticker_a: str
    ticker_b: str
    correlation_30d: float


class CorrelationRegimeOut(BaseModel):
    avg_correlation_30d: float
    avg_correlation_252d: float
    correlation_spike: float
    regime: str
    most_correlated_pair: CorrelationPairOut
    matrix_30d: dict[str, dict[str, float]] | None = None


class PortfolioRiskBreakdown(BaseModel):
    portfolio_id: uuid.UUID
    name: str
    value: Decimal
    var_95: Decimal
    var_pct_of_total: float
    margin_status: str


class AggregateRiskResponse(BaseModel):
    total_portfolio_value: Decimal
    aggregate_var_95: Decimal
    portfolio_count: int
    breakdown: list[PortfolioRiskBreakdown]
    most_exposed_portfolio_id: uuid.UUID | None = None
    most_diversifying_portfolio_id: uuid.UUID | None = None


class BacktestSeriesPoint(BaseModel):
    date: date
    var_95: Decimal
    violated: bool


class KupiecResult(BaseModel):
    # "model_valid" collides with pydantic's protected "model_" namespace; opt out.
    model_config = ConfigDict(protected_namespaces=())

    total_days: int
    expected_violations: float
    actual_violations: int
    violation_rate: float
    kupiec_lr_statistic: float | None = None
    model_valid: bool | None = None
    calibration: str
    violation_dates: list[date]
    series: list[BacktestSeriesPoint] = Field(default_factory=list)
    message: str | None = None


class VolForecastOut(BaseModel):
    ticker: str
    predicted_vol: float
    garch_vol: float
    lstm_mae: float
    garch_mae: float
    improvement_pct: float | None = None
    vol_regime: str
    adjusted_var_95: float | None = None


class VolForecastHistoryPoint(BaseModel):
    computed_at: datetime
    predicted_vol: float
    garch_vol: float


class VolForecastDetailOut(VolForecastOut):
    lstm_rmse: float
    garch_rmse: float
    direction_accuracy: float
    history: list[VolForecastHistoryPoint] = Field(default_factory=list)


class RiskOut(BaseModel):
    computed_at: datetime
    portfolio_value: Decimal
    var_95: Decimal
    var_99: Decimal
    cvar_95: Decimal
    margin_utilization: Decimal
    margin_status: str
    stress_tests: dict[str, StressScenarioOut]
    shap_attribution: list[ShapContributionOut]
    computation_ms: int | None
    monte_carlo: MonteCarloOut | None = None
    risk_narrative: str | None = None
    vol_forecasts: list[VolForecastOut] | None = None
    adjusted_var_95_portfolio: float | None = None


class AlertEventOut(BaseModel):
    id: uuid.UUID
    portfolio_id: uuid.UUID
    portfolio_name: str
    event_type: str
    triggered_at: datetime
    var_95: Decimal
    margin_limit: Decimal
    margin_utilization: Decimal
    message: str
    acknowledged_at: datetime | None = None
    acknowledged: bool = False


class AlertsListResponse(BaseModel):
    items: list[AlertEventOut]
    total: int
    limit: int
    offset: int


class AlertShapAttributionOut(BaseModel):
    ticker: str
    contribution: Decimal
    pct_of_var: Decimal | None = None


class AlertDetailOut(BaseModel):
    id: uuid.UUID
    portfolio_id: uuid.UUID
    portfolio_name: str
    event_type: str
    triggered_at: datetime
    var_95: Decimal
    margin_limit: Decimal
    margin_utilization: Decimal
    message: str
    acknowledged_at: datetime | None = None
    acknowledged: bool = False
    cvar_95: Decimal | None = None
    shap_attributions: list[AlertShapAttributionOut] = Field(default_factory=list)
    stress_loss_moderate: Decimal | None = None
    risk_computed_at: datetime | None = None
