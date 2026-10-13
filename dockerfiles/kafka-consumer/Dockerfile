FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias
COPY dockerfiles/kafka-consumer/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del consumidor
COPY dockerfiles/kafka-consumer/consumer.py /app/consumer.py
COPY dockerfiles/kafka-consumer/event_schema.py /app/event_schema.py

# Copiar lógica compartida de inferencia
COPY src /app/src

ENV PYTHONPATH=/app/src