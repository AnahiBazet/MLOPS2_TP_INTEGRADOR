FROM python:3.13-slim

WORKDIR /app

COPY dockerfiles/grpc-client/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY dockerfiles/grpc-client/client.py /app/client.py
COPY dockerfiles/grpc/scoring_pb2.py /app/scoring_pb2.py
COPY dockerfiles/grpc/scoring_pb2_grpc.py /app/scoring_pb2_grpc.py
COPY src/mlops2_tp/test_cases /app/test_cases