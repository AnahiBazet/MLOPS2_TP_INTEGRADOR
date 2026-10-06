FROM python:3.12-slim

WORKDIR /app

# Dependencias del servicio FastAPI
COPY dockerfiles/fastapi/requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Aplicación FastAPI
COPY dockerfiles/fastapi/app.py /app/app.py

# Código compartido por las APIs
COPY src /app/src

# Permite importar mlops2_tp desde /app/src
ENV PYTHONPATH=/app/src

EXPOSE 8800
