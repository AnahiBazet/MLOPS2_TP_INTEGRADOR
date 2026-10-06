import os

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from mlops2_tp.model_loader import load_champion_model
from mlops2_tp.inference import (
    load_data_dictionary,
    predict_water_potability,
)


# =========================================================
# CARGA DEL MODELO Y METADATA
# =========================================================

model, version_model = load_champion_model()
data_dict = load_data_dictionary()


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Water Potability API",
    version=str(version_model),
    description="API REST para predecir si una muestra de agua es potable.",
)


# =========================================================
# SEGURIDAD
# =========================================================

API_KEY_HEADER = APIKeyHeader(
    name="X-API-KEY",
    auto_error=False,
)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "La variable de entorno API_KEY no está definida."
    )


def validate_api_key(
    api_key: str = Security(API_KEY_HEADER),
):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="API key inválida o ausente.",
        )

    return api_key


# =========================================================
# INPUT
# =========================================================

class WaterFeatures(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    ph: float | None = Field(
        ...,
        description="pH del agua",
    )

    Hardness: float | None = Field(
        ...,
        description="Dureza del agua",
    )

    Solids: float | None = Field(
        ...,
        description="Sólidos disueltos",
    )

    Chloramines: float | None = Field(
        ...,
        description="Concentración de cloraminas",
    )

    Sulfate: float | None = Field(
        ...,
        description="Concentración de sulfatos",
    )

    Conductivity: float | None = Field(
        ...,
        description="Conductividad del agua",
    )

    Organic_carbon: float | None = Field(
        ...,
        description="Carbono orgánico",
    )

    Trihalomethanes: float | None = Field(
        ...,
        description="Concentración de trihalometanos",
    )

    Turbidity: float | None = Field(
        ...,
        description="Turbidez del agua",
    )


# =========================================================
# OUTPUT
# =========================================================

class PredictionResponse(BaseModel):

    prediction: int

    label: str

    probability: float

    model_version: int

# =========================================================
# RAIZ
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Water Potability API",
        "docs": "/docs",
        "health": "/health",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model": "water_potability_model_prod",
        "model_version": version_model,
    }


# =========================================================
# PREDICT
# =========================================================

@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    data: WaterFeatures,
    api_key: str = Security(validate_api_key),
):

    features = data.model_dump()

    prediction, probability = predict_water_potability(
        model=model,
        features=features,
        data_dictionary=data_dict,
    )

    label = (
        "potable"
        if prediction == 1
        else "no potable"
    )

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probability=probability,
        model_version=version_model,
    )