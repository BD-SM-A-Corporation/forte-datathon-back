# Small setup image: no CUDA, PyTorch, OpenCV, or model weights baked in.
FROM python:3.11-slim AS model-init
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
RUN pip install --no-cache-dir 'huggingface_hub==0.36.0'
COPY app/__init__.py ./app/__init__.py
COPY app/core/__init__.py app/core/logging.py ./app/core/
COPY app/workers/__init__.py app/workers/model_init.py ./app/workers/
CMD ["python", "-m", "app.workers.model_init"]

FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
ARG INSTALL_MODEL=true
RUN if [ "$INSTALL_MODEL" = "true" ]; then pip install --no-cache-dir '.[model,quantization]'; \
    else pip install --no-cache-dir .; fi
COPY rules ./rules
RUN useradd --uid 10001 --create-home engine && mkdir -p /app/storage/uploads /models \
    && chown -R engine:engine /app/storage /models
USER engine
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
