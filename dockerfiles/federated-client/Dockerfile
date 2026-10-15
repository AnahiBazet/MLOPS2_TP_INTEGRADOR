# Imagen base de Python.
FROM python:3.11-slim

# Copia uv para instalar las dependencias.
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /bin/

# Directorio de trabajo del cliente federado.
WORKDIR /app

# Copia e instala las dependencias.
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copia la aplicación Flower y sus módulos auxiliares.
COPY client.py .
COPY model.py .
COPY data.py .