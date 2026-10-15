#!/bin/sh
set -e

cd /app/Backend 2>/dev/null || cd Backend 2>/dev/null || true
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
