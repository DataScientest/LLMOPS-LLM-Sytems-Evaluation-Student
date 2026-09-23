FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --frozen

COPY dashboard_push.py .

CMD ["uv", "run", "python", "dashboard_push.py"]