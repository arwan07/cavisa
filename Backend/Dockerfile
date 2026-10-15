# Container untuk API Klasifikasi Sperma.
# Build:  docker build -t sperm-classifier-backend .
# Run:    docker run --rm -p 8000:8000 \
#           -v "$(pwd)/app/models:/srv/app/models:ro" \
#           sperm-classifier-backend

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/srv/app/models

WORKDIR /srv

# curl dipakai oleh healthcheck di bawah
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]