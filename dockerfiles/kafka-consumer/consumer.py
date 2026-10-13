
import json
import os
import signal
from pydantic import ValidationError
from event_schema import EventoValido
from datetime import datetime, timezone

import boto3
from confluent_kafka import Consumer, KafkaException

from mlops2_tp.model_loader import load_champion_model
from mlops2_tp.inference import (
    load_data_dictionary,
    predict_water_potability,
)


# ============================================================
# Configuración
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka:9092"
)

KAFKA_TOPIC = "water.measurements"
KAFKA_GROUP_ID = "water-inference-group"

MINIO_ENDPOINT = os.getenv(
    "AWS_ENDPOINT_URL_S3",
    "http://s3:9000"
)

INFERENCE_BUCKET = os.getenv(
    "INFERENCE_BUCKET_NAME",
    "inference"
)

STATION_PARTITIONS = {
    "station_A": 0,
    "station_B": 1,
}

running = True


# ============================================================
# Finalización controlada
# ============================================================

def stop_consumer(signum, frame):
    """
    Solicita la finalización del consumidor.
    """
    global running
    running = False


# ============================================================
# Almacenamiento en MinIO
# ============================================================

def save_prediction(s3, event, prediction, probability, model_version):
    """
    Guarda la medición original y su predicción en MinIO.
    """

    station_id = event["station_id"]
    timestamp = event["timestamp"]
    event_id = event["event_id"]

    # Organizar los resultados por estación y fecha.
    date = datetime.fromisoformat(
        timestamp.replace("Z", "+00:00")
    ).astimezone(timezone.utc).date().isoformat()

    object_key = (
        f"{station_id}/{date}/{event_id}.json"
    )

    result = {
        "event_id": event_id,
        "station_id": station_id,
        "timestamp": timestamp,
        "features": event["features"],
        "prediction": int(prediction),
        "label": (
            "potable"
            if int(prediction) == 1
            else "no potable"
        ),
        "probability": float(probability),
        "model_version": int(model_version),
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    s3.put_object(
        Bucket=INFERENCE_BUCKET,
        Key=object_key,
        Body=json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False
        ).encode("utf-8"),
        ContentType="application/json",
    )

    return object_key

# ============================================================
# Almacenamiento de eventos rechazados
# ============================================================
def save_rejected_event(s3, message, reason):
    """
    Guarda en cuarentena un evento que no superó la validación.
    """
    date = datetime.now(timezone.utc).date().isoformat()

    object_key = (
        f"dlq/{date}/"
        f"partition_{message.partition()}_offset_{message.offset()}.json"
    )

    rejected_event = {
        "topic": message.topic(),
        "partition": message.partition(),
        "offset": message.offset(),
        "reason": reason,
        "payload": message.value().decode("utf-8", errors="replace"),
        "rejected_at": datetime.now(timezone.utc).isoformat()
    }

    s3.put_object(
        Bucket=INFERENCE_BUCKET,
        Key=object_key,
        Body=json.dumps(rejected_event, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )

    return object_key

# ============================================================
# Procesamiento de mediciones
# ============================================================

def process_message(
    message,
    model,
    model_version,
    data_dict,
    s3,
):
    """
    Valida el origen del evento, ejecuta la inferencia
    y almacena el resultado en MinIO.
    """

    event = json.loads(message.value().decode())

    try:
        validated_event = EventoValido.model_validate(event)
    except ValidationError as error:
        object_key = save_rejected_event(
            s3=s3,
            message=message,
            reason=str(error)
        )

        print(
            f"Evento rechazado | Partición: {message.partition()} "
            f"| Offset: {message.offset()} | MinIO: s3://{INFERENCE_BUCKET}/{object_key}",
            flush=True
        )

        return

    event = validated_event.model_dump(mode="json")

    station_id = event["station_id"]

    if station_id not in STATION_PARTITIONS:
        raise ValueError(
            f"Estación desconocida: {station_id}"
        )

    expected_partition = STATION_PARTITIONS[station_id]

    if message.partition() != expected_partition:
        raise ValueError(
            f"Partición incorrecta para {station_id}. "
            f"Esperada: {expected_partition}, "
            f"recibida: {message.partition()}"
        )

    features = event["features"]

    # Reutilizar la lógica común de inferencia.
    prediction, probability = predict_water_potability(
        model=model,
        features=features,
        data_dictionary=data_dict,
    )

    # Persistir antes de confirmar el offset en Kafka.
    object_key = save_prediction(
        s3=s3,
        event=event,
        prediction=prediction,
        probability=probability,
        model_version=model_version,
    )

    print(
        f"Estación: {station_id} "
        f"| Partición: {message.partition()} "
        f"| Offset: {message.offset()} "
        f"| Predicción: {int(prediction)} "
        f"| MinIO: s3://{INFERENCE_BUCKET}/{object_key}",
        flush=True,
    )


# ============================================================
# Consumidor Kafka
# ============================================================

def main():

    print("Iniciando consumidor Kafka...", flush=True)

    # 1. Cargar modelo Champion una sola vez.
    model, model_version = load_champion_model()

    # 2. Cargar información de preprocesamiento.
    data_dict = load_data_dictionary()

    print(
        f"Modelo Champion cargado. Versión: {model_version}",
        flush=True,
    )

    # 3. Crear cliente MinIO.
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
    )

    # El bucket debe existir antes de iniciar el consumidor.
    s3.head_bucket(Bucket=INFERENCE_BUCKET)

    # 4. Configurar consumidor Kafka.
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    # 5. Suscribirse al tópico.
    consumer.subscribe([KAFKA_TOPIC])

    print(
        f"Escuchando tópico: {KAFKA_TOPIC}",
        flush=True,
    )

    try:
        # 6. Consumir mensajes continuamente.
        while running:

            message = consumer.poll(timeout=1.0)

            if message is None:
                continue

            if message.error():
                raise KafkaException(message.error())

            # Ejecutar inferencia y guardar en MinIO.
            process_message(
                message=message,
                model=model,
                model_version=model_version,
                data_dict=data_dict,
                s3=s3,
            )

            # 7. Confirmar después de guardar correctamente.
            consumer.commit(
                message=message,
                asynchronous=False,
            )

    finally:
        consumer.close()

        print(
            "Consumidor Kafka finalizado.",
            flush=True,
        )


# ============================================================
# Punto de entrada
# ============================================================

if __name__ == "__main__":

    signal.signal(signal.SIGINT, stop_consumer)
    signal.signal(signal.SIGTERM, stop_consumer)

    main()
