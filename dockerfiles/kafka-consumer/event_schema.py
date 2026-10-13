from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, AwareDatetime


class WaterFeatures(BaseModel):
    """
    Valida las características fisicoquímicas del agua.

    - pH: entre 0 y 14.
    - Otras ocho variables: valores no negativos.
    - Valores faltantes: admite null para permitir su imputación.
    - Estructura: exige las nueve variables y rechaza campos adicionales.
    """
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    ph: float | None = Field(ge=0, le=14)
    Hardness: float | None = Field(ge=0)
    Solids: float | None = Field(ge=0)
    Chloramines: float | None = Field(ge=0)
    Sulfate: float | None = Field(ge=0)
    Conductivity: float | None = Field(ge=0)
    Organic_carbon: float | None = Field(ge=0)
    Trihalomethanes: float | None = Field(ge=0)
    Turbidity: float | None = Field(ge=0)


class EventoValido(BaseModel):
    """
    Valida la estructura y los metadatos del evento Kafka.

    - UUID: formato válido.
    - Timestamp: fecha y hora válidas con zona horaria.
    - Estación: solamente station_A o station_B.
    - Estructura: rechaza campos adicionales.
    """
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    station_id: Literal["station_A", "station_B"]
    timestamp: AwareDatetime
    features: WaterFeatures

