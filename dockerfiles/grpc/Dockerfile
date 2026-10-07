FROM python:3.12-slim

WORKDIR /app

COPY dockerfiles/grpc/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY dockerfiles/grpc/server.py /app/server.py
COPY dockerfiles/grpc/scoring_pb2.py /app/scoring_pb2.py
COPY dockerfiles/grpc/scoring_pb2_grpc.py /app/scoring_pb2_grpc.py

COPY src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 50051