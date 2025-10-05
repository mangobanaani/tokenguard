"""Circuit breaker pattern implementation."""
import time
import threading
from enum import Enum
from typing import Callable, TypeVar, Optional, Any
from functools import wraps
from app.errors import CircuitBreakerOpenError
from app.constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


T = TypeVar("T")


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.

    States:
    - CLOSED: Normal operation, requests go through
    - OPEN: Too many failures, reject all requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Auto-transition from OPEN to HALF_OPEN after recovery timeout
            if (
                self._state == CircuitState.OPEN
                and self._last_failure_time
                and time.time() - self._last_failure_time >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._failure_count = 0
            return self._state

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Recovery successful, close circuit
                self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt, reopen circuit
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open circuit
                self._state = CircuitState.OPEN

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(self.name)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator form of circuit breaker."""
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper

    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.recovery_timeout
        }


class AsyncCircuitBreaker:
    """Async version of circuit breaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        expected_exception: type = Exception
    ):
        self._breaker = CircuitBreaker(name, failure_threshold, recovery_timeout, expected_exception)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._breaker.state

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute async function with circuit breaker protection."""
        current_state = self._breaker.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(self._breaker.name)

        try:
            result = await func(*args, **kwargs)
            self._breaker._on_success()
            return result
        except self._breaker.expected_exception as e:
            self._breaker._on_failure()
            raise e

    def __call__(self, func: Callable) -> Callable:
        """Decorator form."""
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await self.call(func, *args, **kwargs)
        return wrapper

    def reset(self):
        """Manually reset circuit breaker."""
        self._breaker.reset()

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return self._breaker.get_stats()
