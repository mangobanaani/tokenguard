"""Application constants and configuration values."""

# Token estimation constants
CHARS_PER_TOKEN = 4.0
CHAT_FORMAT_OVERHEAD = 1.2  # 20% overhead for chat formatting
BASE_CHAT_TOKENS = 4  # Base tokens for chat format
ROLE_FORMATTING_TOKENS = 4  # Tokens per message for role formatting
COMPLETION_PREP_TOKENS = 3  # Additional overhead for completion

# Default token estimates
DEFAULT_OUTPUT_TOKENS = 150
INPUT_OUTPUT_SPLIT = 0.7  # 70% input, 30% output when using requestedTokens

# Token adjustment
TOKEN_ADJUSTMENT_THRESHOLD = 0.1  # 10% threshold for token adjustment

# Default capacity and refill rates
DEFAULT_CAPACITY = 1000.0
DEFAULT_REFILL_PER_SEC = 5.0

# OpenAI sync settings
SYNC_INTERVAL_SECONDS = 300  # 5 minutes
DEFAULT_RESET_SECONDS = 3600  # 1 hour fallback

# Request settings
DEFAULT_REQUEST_TIMEOUT = 30.0
DEFAULT_SYNC_TIMEOUT = 10.0
GLOBAL_REQUEST_TIMEOUT = 60.0

# Circuit breaker settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60.0
CIRCUIT_BREAKER_EXPECTED_EXCEPTION = Exception

# Rate limiting
RETRY_AFTER_SECONDS = 1
DEFAULT_BURST_MULTIPLIER = 1.5  # Allow 50% burst above capacity
SOFT_LIMIT_THRESHOLD = 0.8  # Warn at 80%

# Input validation
MAX_MESSAGE_LENGTH = 100000  # characters
MAX_MESSAGES_COUNT = 1000
MAX_TOKEN_REQUEST = 100000

# Redis settings
REDIS_MAX_CONNECTIONS = 50
REDIS_SOCKET_TIMEOUT = 5.0
REDIS_SOCKET_CONNECT_TIMEOUT = 5.0

# Model token ratios (chars per token)
MODEL_TOKEN_RATIOS = {
    "gpt-4": 4.0,
    "gpt-4-turbo": 4.0,
    "gpt-4o": 4.0,
    "gpt-4o-mini": 4.0,
    "gpt-3.5-turbo": 4.0,
    "text-davinci-003": 4.0,
}

# Monitoring
METRICS_PORT = 9090
HEALTH_CHECK_TIMEOUT = 5.0
