FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias
COPY dockerfiles/kafka-producer/requirements.txt /app/requirements.txt
COPY dockerfiles/kafka-producer/test_invalid_event.py /app/test_invalid_event.py
COPY dockerfiles/kafka-producer/test_invalid_signature.py /app/test_invalid_signature.py

RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del productor
COPY dockerfiles/kafka-producer/producer.py /app/producer.py