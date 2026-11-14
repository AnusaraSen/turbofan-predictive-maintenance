# --- Stage 1: Build React + Vite frontend ---
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Empty VITE_API_URL makes the frontend use same-origin relative API paths
ENV VITE_API_URL=""
RUN npm run build

# --- Stage 2: FastAPI + TensorFlow backend serving API + static frontend ---
FROM python:3.11-slim
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY results/metrics_summary.csv ./results/metrics_summary.csv
COPY data/raw/RUL_*.txt ./data/raw/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", ".venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
