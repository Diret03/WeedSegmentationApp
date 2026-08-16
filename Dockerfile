# Multi-stage build: wheels are compiled in the builder, only the installed
# packages reach the runtime image.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# OpenCV needs no GUI bindings inside a container
RUN sed 's/opencv-python==/opencv-python-headless==/g' requirements.txt > requirements_headless.txt

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir \
        torch==2.0.1+cpu \
        torchvision==0.15.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_headless.txt


FROM python:3.11-slim AS runtime

LABEL authors="Diret"
LABEL description="Weed Segmentation Flask PyTorch Application"

# libglib2/libgomp are the only shared libraries opencv-headless and torch need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p uploads results models logs && chown -R app:app /app

# Only the checkpoint actually served is baked into the image
ARG MODEL_FILE=weed_segmentation_S-TTA.pth

COPY --chown=app:app app.py weed_predictor.py logger_config.py ./
COPY --chown=app:app models/${MODEL_FILE} ./models/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app templates/ ./templates/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/.torch \
    OMP_NUM_THREADS=4 \
    MODEL_PATH=models/${MODEL_FILE}

USER app
EXPOSE 5000

# /health reports the model state, unlike / which answers 200 regardless
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
