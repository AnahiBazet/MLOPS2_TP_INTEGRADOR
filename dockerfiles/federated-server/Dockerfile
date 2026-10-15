# Imagen base de Python.
FROM python:3.11-slim

# Copia uv para instalar las dependencias.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/

# Directorio de trabajo del servidor federado.
WORKDIR /app

# Copia e instala las dependencias del servidor.
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copia la aplicación Flower y el módulo de evaluación.
COPY server.py .
COPY evaluate.py .

# Crea el directorio donde se guardarán los resultados.
RUN mkdir -p /app/results