import os

import strawberry
from fastapi import FastAPI, Request
from graphql import GraphQLError
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from mlops2_tp.model_loader import load_champion_model
from mlops2_tp.inference import (
    load_data_dictionary,
    predict_water_potability,
)


# Cargar el modelo Champion y la información necesaria para inferencia.
model, model_version = load_champion_model()
data_dict = load_data_dictionary()

API_KEY = os.getenv("API_KEY")

def verify_api_key(info: Info):
    request: Request = info.context["request"]
    api_key = request.headers.get("X-API-KEY")

    if not API_KEY or api_key != API_KEY:
        raise GraphQLError("Invalid or missing API key")


@strawberry.type
class Prediction:
    prediction: int
    label: str
    probability: float
    model_version: int


@strawberry.type
class Query:

    @strawberry.field
    def predict(
        self,
        info: Info,
        ph: float | None = None,
        hardness: float | None = None,
        solids: float | None = None,
        chloramines: float | None = None,
        sulfate: float | None = None,
        conductivity: float | None = None,
        organic_carbon: float | None = None,
        trihalomethanes: float | None = None,
        turbidity: float | None = None,
    ) -> Prediction:

        verify_api_key(info)

        features = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity,
        }

        prediction, probability = predict_water_potability(
            model=model,
            features=features,
            data_dictionary=data_dict,
        )

        label = "potable" if prediction == 1 else "no potable"

        return Prediction(
            prediction=prediction,
            label=label,
            probability=probability,
            model_version=model_version,
        )


schema = strawberry.Schema(query=Query)

async def get_context(request: Request):
    return {"request": request}


graphql_app = GraphQLRouter(
    schema,
    context_getter=get_context,
)

app = FastAPI(
    title="Water Potability GraphQL API",
    version=str(model_version),
)

app.include_router(
    graphql_app,
    prefix="/graphql",
)

@app.get("/")
def root():
    return {
        "message": "Water Potability GraphQL API",
        "graphql": "/graphql",
        "health": "/health",
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "water_potability_model_prod",
        "model_version": model_version,
    }