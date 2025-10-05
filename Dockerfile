# syntax=docker/dockerfile:1.11-labs

# =============================================================================
# Builder Stage - Install dependencies and build application
# =============================================================================
FROM python:3.13-slim AS builder

# Build arguments for cache control
ARG PYTHON_VERSION=3.13
ARG PIP_VERSION=24.3.1

# Python optimization flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    VENV_PATH=/opt/venv

# Create virtual environment
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/pip install --upgrade pip==${PIP_VERSION}

# Set PATH to use virtual environment
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install build dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# Copy dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e . && \
    pip install gunicorn uvicorn[standard]

# =============================================================================
# Runtime Stage - Minimal production image
# =============================================================================
FROM python:3.13-slim AS runtime

# Runtime arguments and labels
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=latest

# OCI Labels for better metadata
LABEL org.opencontainers.image.title="TokenGuard" \
      org.opencontainers.image.description="Hierarchical token bucket rate limiter for LLM APIs" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="TokenGuard Project" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/mangobanaani/tokenguard"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    VENV_PATH=/opt/venv \
    APP_HOME=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=info \
    TIMEOUT=120 \
    GRACEFUL_TIMEOUT=30 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100

# Install minimal runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        tini \
        && rm -rf /var/lib/apt/lists/* \
        && apt-get clean

# Set PATH to use virtual environment
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -d ${APP_HOME} -s /sbin/nologin appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup ${VENV_PATH} ${VENV_PATH}

# Set working directory and copy application code
WORKDIR ${APP_HOME}
COPY --chown=appuser:appgroup app ./app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE ${PORT}

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Use tini as init system to handle signals properly
ENTRYPOINT ["tini", "--"]

# Optimized CMD with better configuration
CMD ["sh", "-c", "exec gunicorn app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind ${HOST}:${PORT} \
    --workers ${WORKERS} \
    --timeout ${TIMEOUT} \
    --graceful-timeout ${GRACEFUL_TIMEOUT} \
    --max-requests ${MAX_REQUESTS} \
    --max-requests-jitter ${MAX_REQUESTS_JITTER} \
    --log-level ${LOG_LEVEL} \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance"]
