# CAVISA (Computer-Aided Visual Sperm Analysis)
# Production Container for Railway, Render, VPS, & Docker Compose

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/Backend/app/models \
    FRONTEND_DIR=/app/Frontend \
    PORT=8000

WORKDIR /app

# Install curl untuk healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Cache dependensi Python
COPY Backend/requirements.txt ./Backend/
RUN pip install --no-cache-dir -r Backend/requirements.txt

# Salin source code Backend dan Frontend
COPY Backend ./Backend
COPY Frontend ./Frontend

WORKDIR /app/Backend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# Railway meneruskan port dinamis via environment variable $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
