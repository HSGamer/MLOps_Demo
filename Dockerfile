FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim
WORKDIR /app

COPY . .

RUN uv sync --locked --no-editable

EXPOSE 5000

CMD ["uv", "run", "app.py"]