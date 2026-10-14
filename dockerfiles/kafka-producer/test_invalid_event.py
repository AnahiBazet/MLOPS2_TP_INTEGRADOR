
import json
import os
import uuid
from datetime import datetime, timezone
import hashlib
import hmac

from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
)

KAFKA_TOPIC = "water.measurements"

HMAC_SECRET = os.environ["KAFKA_HMAC_SECRET"].encode("utf-8")


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

    producer.produce(
        topic=KAFKA_TOPIC,
        partition=0,
        key=b"station_A",
        value=payload,
        headers={"hmac_signature": signature}
    )

    pending = producer.flush(timeout=15)

    if pending:
        raise RuntimeError("No se pudo confirmar la publicación.")

    print(f"Evento de prueba publicado: {event['event_id']}")
    print("pH enviado: 15.0")


if __name__ == "__main__":
    main()
