FROM python:3.11-slim

# Install system dependencies (ffmpeg, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml /app/pyproject.toml

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY rules/ /app/rules/
COPY app/ /app/app/
COPY weights/ /app/weights/

# Ensure storage directories exist
RUN mkdir -p /app/storage/uploads /app/storage/frames

ENV PYTHONPATH=/app
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
