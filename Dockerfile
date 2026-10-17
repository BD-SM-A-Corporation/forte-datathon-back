FROM python:3.11-slim
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
RUN python -c "from pathlib import Path; from app.services.rules import RuleEngine; RuleEngine.load(Path('/app/rules/regulations.json'))"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
