# Dockerfile for Smart Dispatch Forecast API (Phase 5.1)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=phase_5_1_forecast_api.py
ENV FLASK_ENV=production

# Install system dependencies
# curl is needed by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/phase_5_1_forecast_api.py .

# market_data.db is deliberately NOT copied in. It is generated data, not
# source, so it is gitignored and absent from a fresh clone -- baking it in
# would break the build. docker-compose mounts it at runtime instead.
#
# .env is deliberately NOT copied in either. Secrets belong in the runtime
# environment, not in an image layer anyone with the image can read.

# Create directories for model files
RUN mkdir -p /app/models

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Run the application
CMD ["python", "phase_5_1_forecast_api.py"]