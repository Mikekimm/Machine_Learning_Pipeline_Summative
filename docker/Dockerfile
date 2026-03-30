FROM python:3.10-slim-bookworm

LABEL maintainer="ml-engineer"
LABEL description="Intel Image Classifier API"

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Create data directories
RUN mkdir -p data/train data/test data/uploads

# Environment
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/intel_classifier_final.h5

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--access-log", "--log-level", "info"]
