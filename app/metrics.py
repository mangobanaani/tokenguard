"""Prometheus metrics for monitoring."""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Optional


# Request metrics
requests_total = Counter(
    "tokenguard_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"]
)

requests_duration = Histogram(
    "tokenguard_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"]
)

# Rate limiting metrics
rate_limit_hits_total = Counter(
    "tokenguard_rate_limit_hits_total",
    "Total number of rate limit hits",
    ["scope"]
)

rate_limit_rejections_total = Counter(
    "tokenguard_rate_limit_rejections_total",
    "Total number of requests rejected due to rate limits",
    ["scope"]
)

# Token metrics
tokens_consumed_total = Counter(
    "tokenguard_tokens_consumed_total",
    "Total tokens consumed",
    ["scope", "model"]
)

tokens_estimated = Histogram(
    "tokenguard_tokens_estimated",
    "Estimated tokens per request",
    ["model"]
)

tokens_actual = Histogram(
    "tokenguard_tokens_actual",
    "Actual tokens used per request",
    ["model"]
)

token_estimation_error = Histogram(
    "tokenguard_token_estimation_error",
    "Token estimation error (actual - estimated)",
    ["model"]
)

# Bucket metrics
bucket_tokens_current = Gauge(
    "tokenguard_bucket_tokens_current",
    "Current tokens in bucket",
    ["scope", "id"]
)

bucket_capacity = Gauge(
    "tokenguard_bucket_capacity",
    "Bucket capacity",
    ["scope", "id"]
)

bucket_refill_rate = Gauge(
    "tokenguard_bucket_refill_rate",
    "Bucket refill rate per second",
    ["scope", "id"]
)

# OpenAI metrics
openai_requests_total = Counter(
    "tokenguard_openai_requests_total",
    "Total OpenAI API requests",
    ["model", "status"]
)

openai_request_duration = Histogram(
    "tokenguard_openai_request_duration_seconds",
    "OpenAI API request duration",
    ["model"]
)

openai_errors_total = Counter(
    "tokenguard_openai_errors_total",
    "Total OpenAI API errors",
    ["error_type"]
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "tokenguard_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half_open, 2=open)",
    ["service"]
)

circuit_breaker_failures_total = Counter(
    "tokenguard_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["service"]
)

# Redis metrics
redis_operations_total = Counter(
    "tokenguard_redis_operations_total",
    "Total Redis operations",
    ["operation", "status"]
)

redis_operation_duration = Histogram(
    "tokenguard_redis_operation_duration_seconds",
    "Redis operation duration",
    ["operation"]
)

# Application info
app_info = Info("tokenguard_app", "Application information")


def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record request metrics."""
    requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    requests_duration.labels(method=method, endpoint=endpoint).observe(duration)


def record_rate_limit(scope: str, rejected: bool = False):
    """Record rate limit event."""
    rate_limit_hits_total.labels(scope=scope).inc()
    if rejected:
        rate_limit_rejections_total.labels(scope=scope).inc()


def record_tokens(
    scope: str,
    model: str,
    estimated: Optional[int] = None,
    actual: Optional[int] = None
):
    """Record token usage."""
    if estimated:
        tokens_estimated.labels(model=model).observe(estimated)
    if actual:
        tokens_actual.labels(model=model).observe(actual)
        tokens_consumed_total.labels(scope=scope, model=model).inc(actual)
    if estimated and actual:
        error = actual - estimated
        token_estimation_error.labels(model=model).observe(error)


def record_bucket_state(scope: str, id: str, tokens: float, capacity: float, refill_rate: float):
    """Record bucket state."""
    bucket_tokens_current.labels(scope=scope, id=id).set(tokens)
    bucket_capacity.labels(scope=scope, id=id).set(capacity)
    bucket_refill_rate.labels(scope=scope, id=id).set(refill_rate)


def record_openai_request(model: str, status: str, duration: float):
    """Record OpenAI API request."""
    openai_requests_total.labels(model=model, status=status).inc()
    openai_request_duration.labels(model=model).observe(duration)


def record_openai_error(error_type: str):
    """Record OpenAI API error."""
    openai_errors_total.labels(error_type=error_type).inc()


def record_circuit_breaker_state(service: str, state: str):
    """Record circuit breaker state."""
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    circuit_breaker_state.labels(service=service).set(state_map.get(state, 0))


def record_circuit_breaker_failure(service: str):
    """Record circuit breaker failure."""
    circuit_breaker_failures_total.labels(service=service).inc()


def record_redis_operation(operation: str, status: str, duration: float):
    """Record Redis operation."""
    redis_operations_total.labels(operation=operation, status=status).inc()
    redis_operation_duration.labels(operation=operation).observe(duration)


def set_app_info(version: str, python_version: str, backend: str):
    """Set application information."""
    app_info.info({
        "version": version,
        "python_version": python_version,
        "backend": backend
    })
