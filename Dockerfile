FROM python:3.12.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY fastapi_app/ fastapi_app/

EXPOSE 8001

# Run with uv (automatically uses .venv)
CMD ["uv", "run", "uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
