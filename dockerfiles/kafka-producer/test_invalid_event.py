
import json
import os
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
)

KAFKA_TOPIC = "water.measurements"


def main():
    """
    Publica un evento inválido para probar la validación
    de rangos mediante Pydantic y su almacenamiento en DLQ.
    """

    event = {
        "event_id": str(uuid.uuid4()),
        "station_id": "station_A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": {
            "ph": 15.0,  # Inválido: el rango permitido es 0-14
            "Hardness": 192.45,
            "Solids": 18257.57,
            "Chloramines": 7.47,
            "Sulfate": 313.78,
            "Conductivity": 462.40,
            "Organic_carbon": 13.93,
            "Trihalomethanes": 53.19,
            "Turbidity": 4.36,
        }
    }

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS
    })

    producer.produce(
        topic=KAFKA_TOPIC,
        partition=0,
        key=b"station_A",
        value=json.dumps(event).encode("utf-8")
    )

    pending = producer.flush(timeout=15)

    if pending:
        raise RuntimeError("No se pudo confirmar la publicación.")

    print(f"Evento de prueba publicado: {event['event_id']}")
    print("pH enviado: 15.0")


if __name__ == "__main__":
    main()
