
import io
import json
import os
import time
import uuid
import hashlib
import hmac

from datetime import datetime, timezone

import boto3
import pandas as pd

from confluent_kafka import Producer


# ============================================================
# Configuración
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092"
)

KAFKA_TOPIC = "water.measurements"

MINIO_ENDPOINT = os.getenv(
    "AWS_ENDPOINT_URL_S3",
    "http://s3:9000"
)

SOURCE_BUCKET = "data"
SOURCE_KEY = "final/test/water_X_test.csv"

PUBLISH_INTERVAL = float(
    os.getenv("PUBLISH_INTERVAL", "2")
)

STATIONS = {
    "station_A": 0,
    "station_B": 1,
}

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

HMAC_SECRET = os.environ["KAFKA_HMAC_SECRET"].encode("utf-8")

# ============================================================
# Lectura del dataset desde MinIO
# ============================================================

def load_test_data():
    """
    Lee el dataset de test desde MinIO.
    """

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT
    )

    response = s3.get_object(
        Bucket=SOURCE_BUCKET,
        Key=SOURCE_KEY
    )

    dataframe = pd.read_csv(
        io.BytesIO(response["Body"].read())
    )

    # Verificar que estén las variables necesarias.
    missing = set(FEATURE_COLUMNS) - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el dataset: {sorted(missing)}"
        )

    dataframe = dataframe[FEATURE_COLUMNS]

    # Evitar publicar valores no válidos como NaN.
    if dataframe.isna().any().any():
        raise ValueError(
            "El dataset contiene valores faltantes."
        )

    return dataframe


# ============================================================
# Construcción del evento
# ============================================================

def build_event(row, station_id):
    """
    Construye un evento de medición para Kafka.
    """

    return {
        "event_id": str(uuid.uuid4()),
        "station_id": station_id,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "features": {
            column: float(row[column])
            for column in FEATURE_COLUMNS
        }
    }


# ============================================================
# Confirmación de publicación
# ============================================================

delivery_errors = []


def delivery_callback(error, message):
    """
    Se ejecuta cuando Kafka confirma o rechaza
    la entrega de un mensaje.
    """

    if error is not None:
        delivery_errors.append(str(error))
        print(f"Error al publicar: {error}", flush=True)


# ============================================================
# Productor Kafka
# ============================================================

def main():

    print("Iniciando productor Kafka...", flush=True)

    # 1. Cargar dataset desde MinIO.
    dataframe = load_test_data()

    print(
        f"Dataset cargado: {len(dataframe)} muestras",
        flush=True
    )

    # 2. Crear productor.
    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "client.id": "water-monitoring-producer",
        "enable.idempotence": True,
    })

    # 3. Verificar que el tópico tenga dos particiones.
    metadata = producer.list_topics(
        topic=KAFKA_TOPIC,
        timeout=15
    )

    topic_metadata = metadata.topics.get(KAFKA_TOPIC)

    if (
        topic_metadata is None
        or topic_metadata.error is not None
        or len(topic_metadata.partitions) != 2
    ):
        raise RuntimeError(
            f"El tópico {KAFKA_TOPIC} debe existir "
            "y tener exactamente 2 particiones."
        )

    # 4. Publicar las mediciones progresivamente.
    for index, (_, row) in enumerate(dataframe.iterrows()):

        station_id = (
            "station_A"
            if index % 2 == 0
            else "station_B"
        )

        partition = STATIONS[station_id]

        event = build_event(row, station_id)

        # Procesar confirmaciones pendientes.
        producer.poll(0)

        payload, signature = sign_event(event)

        producer.produce(
            topic=KAFKA_TOPIC,
            key=station_id.encode("utf-8"),
            value=payload,
            partition=partition,
            headers={"hmac_signature": signature},
            callback=delivery_callback
        )

        print(
            f"Medición {index + 1}/{len(dataframe)} "
            f"| {station_id} "
            f"| partición {partition} "
            f"| event_id={event['event_id']}",
            flush=True
        )

        time.sleep(PUBLISH_INTERVAL)

    # 5. Esperar confirmaciones de Kafka.
    pending = producer.flush(timeout=30)

    if pending > 0 or delivery_errors:
        raise RuntimeError(
            f"Publicación incompleta. "
            f"Pendientes: {pending}. "
            f"Errores: {delivery_errors}"
        )

    print(
        "Todas las mediciones fueron publicadas.",
        flush=True
    )


# ============================================================
# Generación de firma en el HEADER
# ============================================================    

def sign_event(event: dict) -> bytes:
    """
    Serializa el evento y genera su firma HMAC-SHA256 a partir
     del contenido del evento y de la clave secreta.

    La firma se incorpora al mensaje Kafka como header.
    """
    payload = json.dumps(
        event,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False
    ).encode("utf-8")

    signature = hmac.new(
        HMAC_SECRET,
        payload,
        hashlib.sha256
    ).hexdigest()

    return payload, signature

if __name__ == "__main__":
    main()
