# syntax=docker/dockerfile:1
# Dockerfile for Smart Dispatch Forecast API (Phase 5.1)
#
# Build:
#   docker build -t smart-dispatch-forecast:latest .
# The security-agents toolkit pinned in requirements.txt is a public
# repository, so the plain build resolves it anonymously. Only if that
# repository is ever made private again, pass a read-only token as a BuildKit
# secret (it is mounted for the pip step only and never written to a layer):
#   TOOLKIT_TOKEN=<PAT> docker build --secret id=toolkit_token,env=TOOLKIT_TOKEN -t smart-dispatch-forecast:latest .
# 3.11: the pinned, non-vulnerable python-dotenv/pytest releases require >=3.10,
# and the test matrix already runs 3.10/3.12 -- 3.9 was the odd one out.
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=backend/phase_5_1_forecast_api.py
# The app resolves REPO_ROOT as the parent of backend/, i.e. /app, which is
# where compose/k8s mount market_data.db and where forecasting_model/ lives.
ENV PYTHONPATH=/app
ENV FLASK_ENV=production

# Install system dependencies
# curl is needed by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies. The optional toolkit_token secret (see header)
# is honoured if present and otherwise ignored; the git pin resolves anonymously.
RUN --mount=type=secret,id=toolkit_token,required=false \
    if [ -s /run/secrets/toolkit_token ]; then \
      git config --global url."https://x-access-token:$(cat /run/secrets/toolkit_token)@github.com/".insteadOf "https://github.com/"; \
    fi \
    && pip install --no-cache-dir -r requirements.txt \
    && git config --global --unset-all url."https://x-access-token:$(cat /run/secrets/toolkit_token 2>/dev/null)@github.com/".insteadOf || true

# Copy application code -- the API module plus the packages it imports
# (forecasting_model.walk_forward -> common.db), laid out as in the repo so
# REPO_ROOT (= parent of backend/) is /app.
COPY backend/phase_5_1_forecast_api.py ./backend/
COPY forecasting_model/ ./forecasting_model/
COPY common/ ./common/

# market_data.db is deliberately NOT copied in. It is generated data, not
# source, so it is gitignored and absent from a fresh clone -- baking it in
# would break the build. docker-compose mounts it at runtime instead.
#
# .env is deliberately NOT copied in either. Secrets belong in the runtime
# environment, not in an image layer anyone with the image can read.

# Create directories for model files
RUN mkdir -p /app/models

# Run as an unprivileged user (SECURITY.md RC-9). The k8s manifests set
# runAsNonRoot and a read-only root filesystem; this makes the plain
# `docker run` path match them.
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Serve through gunicorn, not Flask's development server. Binding every
# interface is correct *inside* the container: the port mapping, the k8s
# Service and the ingress (which carries authn and rate limits) are the
# perimeter. The application also requires a bearer token on /forecast*.
CMD ["gunicorn", "--chdir", "/app/backend", "--bind", "0.0.0.0:5001", \
     "--workers", "2", "--threads", "4", "--timeout", "60", \
     "--access-logfile", "-", "phase_5_1_forecast_api:app"]