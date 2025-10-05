"""Application configuration using Pydantic Settings."""
import os
from typing import Dict, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.constants import (
    DEFAULT_CAPACITY,
    DEFAULT_REFILL_PER_SEC,
    SYNC_INTERVAL_SECONDS,
    TOKEN_ADJUSTMENT_THRESHOLD,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
)


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Redis
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    redis_max_connections: int = Field(default=50, description="Redis max connections")
    redis_socket_timeout: float = Field(default=5.0, description="Redis socket timeout")
    redis_socket_connect_timeout: float = Field(default=5.0, description="Redis connect timeout")

    # Default capacities
    bucket_capacity: float = Field(default=DEFAULT_CAPACITY, description="Default bucket capacity")
    bucket_refill_per_sec: float = Field(default=DEFAULT_REFILL_PER_SEC, description="Default refill rate")

    cap_global: float = Field(default=DEFAULT_CAPACITY, description="Global scope capacity")
    cap_team: float = Field(default=DEFAULT_CAPACITY, description="Team scope capacity")
    cap_rolegroup: float = Field(default=DEFAULT_CAPACITY, description="Rolegroup scope capacity")
    cap_role: float = Field(default=DEFAULT_CAPACITY, description="Role scope capacity")
    cap_user: float = Field(default=DEFAULT_CAPACITY, description="User scope capacity")

    # Default refill rates
    rate_global: float = Field(default=DEFAULT_REFILL_PER_SEC, description="Global scope refill rate")
    rate_team: float = Field(default=DEFAULT_REFILL_PER_SEC, description="Team scope refill rate")
    rate_rolegroup: float = Field(default=DEFAULT_REFILL_PER_SEC, description="Rolegroup scope refill rate")
    rate_role: float = Field(default=DEFAULT_REFILL_PER_SEC, description="Role scope refill rate")
    rate_user: float = Field(default=DEFAULT_REFILL_PER_SEC, description="User scope refill rate")

    # Features
    enforce_global_scope: bool = Field(default=False, description="Enforce global scope rate limiting")
    enable_background_sync: bool = Field(default=True, description="Enable background OpenAI sync")
    enable_burst: bool = Field(default=True, description="Enable burst allowance")
    enable_soft_limits: bool = Field(default=True, description="Enable soft limit warnings")
    enable_ip_rate_limiting: bool = Field(default=False, description="Enable IP-based rate limiting")

    # Token adjustment
    token_adjustment_threshold: float = Field(
        default=TOKEN_ADJUSTMENT_THRESHOLD,
        description="Threshold for token adjustment (percentage)"
    )

    # Sync settings
    sync_interval: int = Field(default=SYNC_INTERVAL_SECONDS, description="OpenAI sync interval in seconds")

    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(
        default=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        description="Failures before opening circuit"
    )
    circuit_breaker_recovery_timeout: float = Field(
        default=CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        description="Seconds before attempting recovery"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    enable_sentry: bool = Field(default=False, description="Enable Sentry error tracking")
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN")
    sentry_environment: str = Field(default="production", description="Sentry environment")
    sentry_traces_sample_rate: float = Field(default=0.1, description="Sentry traces sample rate")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Secrets management
    use_aws_secrets: bool = Field(default=False, description="Load secrets from AWS Secrets Manager")
    aws_secret_name: Optional[str] = Field(default=None, description="AWS secret name")

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if v and not v.startswith("sk-"):
            if v != "test-key":  # Allow test key for testing
                raise ValueError("OpenAI API key must start with 'sk-'")
        return v

    @field_validator("token_adjustment_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate token adjustment threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Token adjustment threshold must be between 0 and 1")
        return v

    @property
    def default_capacities(self) -> Dict[str, float]:
        """Get default capacities for all scopes."""
        return {
            "global": self.cap_global,
            "team": self.cap_team,
            "rolegroup": self.cap_rolegroup,
            "role": self.cap_role,
            "user": self.cap_user,
        }

    @property
    def default_rates(self) -> Dict[str, float]:
        """Get default refill rates for all scopes."""
        return {
            "global": self.rate_global,
            "team": self.rate_team,
            "rolegroup": self.rate_rolegroup,
            "role": self.rate_role,
            "user": self.rate_user,
        }

    @property
    def is_test_mode(self) -> bool:
        """Check if running in test mode."""
        return self.openai_api_key == "test-key" or not self.openai_api_key


# Global settings instance
settings = Settings()
