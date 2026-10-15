# Imagen oficial de Flower con SuperExec.
FROM flwr/superexec:1.38.0

# Directorio de trabajo.
WORKDIR /app

# Copia la configuración de la aplicación Flower.
COPY --chown=app:app dockerfiles/federated-app/pyproject.toml .

# Crea el paquete Python de Flower dentro de la imagen.
# Los archivos originales permanecen en sus carpetas.
RUN mkdir -p water_federated

# Código del cliente federado.
COPY --chown=app:app dockerfiles/federated-client/client.py water_federated/
COPY --chown=app:app dockerfiles/federated-client/model.py water_federated/
COPY --chown=app:app dockerfiles/federated-client/data.py water_federated/

# Código del servidor federado.
COPY --chown=app:app dockerfiles/federated-server/server.py water_federated/
COPY --chown=app:app dockerfiles/federated-server/evaluate.py water_federated/
COPY --chown=app:app dockerfiles/federated-server/plots.py water_federated/

# Identifica el directorio como paquete Python.
RUN touch water_federated/__init__.py

# Flower ya está instalado en la imagen base.
# Excluye su dependencia para evitar reinstalarlo.
# Instala las demás dependencias y nuestro paquete Python.
RUN sed -i '/^[[:space:]]*"flwr==1\.38\.0",/d' pyproject.toml \
    && python -m pip install --no-cache-dir .

# Componente que ejecutará las aplicaciones de Flower.
ENTRYPOINT ["flower-superexec"]