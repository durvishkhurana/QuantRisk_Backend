import time
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_LATENCY = Histogram(
    "quantrisk_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint", "status_code"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
REQUEST_COUNT = Counter(
    "quantrisk_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
RISK_COMPUTATION_LATENCY = Histogram(
    "quantrisk_risk_computation_duration_ms",
    "Risk computation latency in ms",
    ["triggered_by"],
    buckets=[50, 100, 200, 500, 1000, 2000],
)
RISK_COMPUTATION_COUNT = Counter(
    "quantrisk_risk_computations_total",
    "Total risk computations",
    ["portfolio_id", "status"],
)
MARGIN_BREACH_COUNT = Counter(
    "quantrisk_margin_breaches_total",
    "Total margin breaches",
    ["portfolio_id"],
)
CACHE_HIT_RATE = Counter("quantrisk_redis_cache_hits_total", "Redis cache hits", ["cache_type"])
CACHE_MISS_RATE = Counter("quantrisk_redis_cache_misses_total", "Redis cache misses", ["cache_type"])


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return route.path
    return request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        endpoint = _endpoint_label(request)
        status = str(response.status_code)
        REQUEST_COUNT.labels(request.method, endpoint, status).inc()
        REQUEST_LATENCY.labels(request.method, endpoint, status).observe(elapsed)
        return response
