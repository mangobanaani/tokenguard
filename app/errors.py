"""Error codes and exception classes."""
from enum import Enum
from typing import Dict, Any, Optional


class ErrorCode(str, Enum):
    """Standardized error codes."""
    RATE_LIMITED = "rate_limited"
    INVALID_SCOPE = "invalid_scope"
    INVALID_INPUT = "invalid_input"
    OPENAI_ERROR = "openai_error"
    REDIS_ERROR = "redis_error"
    TOKEN_ADJUSTMENT_FAILED = "token_adjustment_failed"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    VALIDATION_ERROR = "validation_error"
    INTERNAL_ERROR = "internal_error"
    MISSING_API_KEY = "missing_api_key"
    INVALID_API_KEY = "invalid_api_key"
    TIMEOUT_ERROR = "timeout_error"
    MISSING_REQUIRED_FIELD = "missing_required_field"


class TokenGuardException(Exception):
    """Base exception for TokenGuard errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.error_code.value,
            "message": self.message,
            "details": self.details
        }


class RateLimitError(TokenGuardException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, buckets: Dict[str, Any]):
        super().__init__(
            error_code=ErrorCode.RATE_LIMITED,
            message=message,
            details={"buckets": buckets},
            status_code=429
        )


class ValidationError(TokenGuardException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            error_code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
            status_code=400
        )


class CircuitBreakerOpenError(TokenGuardException):
    """Raised when circuit breaker is open."""

    def __init__(self, service: str):
        super().__init__(
            error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            message=f"Circuit breaker is open for {service}",
            details={"service": service},
            status_code=503
        )
