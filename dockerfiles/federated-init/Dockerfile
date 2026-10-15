FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/

WORKDIR /app

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY prepare_data.py .