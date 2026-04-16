# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

# Only the libs OpenCV needs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

COPY app.py .

# Writable dirs for uploads / models (bind-mounted in compose)
RUN mkdir -p /app/uploads /app/models

EXPOSE 8000

# Run with uvloop + httptools for max throughput
CMD ["python", "-m", "uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--loop", "uvloop", \
    "--http", "httptools", \
    "--workers", "1"]