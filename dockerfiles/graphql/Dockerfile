FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/graphql/requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY dockerfiles/graphql/app.py /app/app.py
COPY src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8801