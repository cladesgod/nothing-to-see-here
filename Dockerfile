FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (install first for Docker layer caching)
COPY pyproject.toml .
COPY agents.toml .
RUN pip install --no-cache-dir -e ".[api]"

# Application code
COPY src/ src/
COPY run.py .
COPY examples/ examples/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Default: start API server
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
