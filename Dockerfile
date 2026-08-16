# Ultra-minimal Docker image for production
FROM python:3.11-slim

LABEL authors="Diret"
LABEL description="Weed Segmentation Flask PyTorch Application - Ultra Minimal"

# Install absolute minimum runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    curl \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Copy requirements and optimize for minimal install
COPY requirements.txt .
RUN sed 's/opencv-python==/opencv-python-headless==/g' requirements.txt > requirements_headless.txt

# Install minimal Python packages
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_headless.txt && \
    pip cache purge

# Create directories with proper ownership
RUN mkdir -p uploads results models static templates && \
    chown -R app:app /app

# Copy files with proper ownership
COPY --chown=app:app app.py weed_predictor.py logger_config.py ./
COPY --chown=app:app appTest.png ./
COPY --chown=app:app models/ ./models/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app templates/ ./templates/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1

USER app
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
