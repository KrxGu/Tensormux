FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tensormux/ tensormux/

RUN pip install --no-cache-dir .

COPY config.yaml .

CMD ["uvicorn", "tensormux.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
