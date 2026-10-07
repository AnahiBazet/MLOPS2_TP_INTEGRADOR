import json
import os
import grpc
import requests
import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

FASTAPI_URL = "http://localhost:8800/v1/predict"
GRAPHQL_URL = "http://localhost:8801/graphql"
GRPC_ADDRESS = "localhost:50051"

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no está definida en el archivo .env")

TEST_CASES_DIR = Path(__file__).parent / "test_cases"

GRPC_DIR = Path(__file__).resolve().parents[2] / "dockerfiles" / "grpc"
sys.path.append(str(GRPC_DIR))

import scoring_pb2
import scoring_pb2_grpc

GRAPHQL_QUERY = """
query Predict(
  $ph: Float
  $hardness: Float
  $solids: Float
  $chloramines: Float
  $sulfate: Float
  $conductivity: Float
  $organicCarbon: Float
  $trihalomethanes: Float
  $turbidity: Float
) {
  predict(
    ph: $ph
    hardness: $hardness
    solids: $solids
    chloramines: $chloramines
    sulfate: $sulfate
    conductivity: $conductivity
    organicCarbon: $organicCarbon
    trihalomethanes: $trihalomethanes
    turbidity: $turbidity
  ) {
    prediction
    label
    probability
    modelVersion
  }
}
"""

def load_test_case(filename: str) -> dict:
    file_path = TEST_CASES_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def test_fastapi(data: dict) -> dict:
    headers = {
        "X-API-KEY": API_KEY,
    }

    response = requests.post(
        FASTAPI_URL,
        json=data,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()

def test_graphql(data: dict) -> dict:
    headers = {
        "X-API-KEY": API_KEY,
    }

    variables = {
        "ph": data["ph"],
        "hardness": data["Hardness"],
        "solids": data["Solids"],
        "chloramines": data["Chloramines"],
        "sulfate": data["Sulfate"],
        "conductivity": data["Conductivity"],
        "organicCarbon": data["Organic_carbon"],
        "trihalomethanes": data["Trihalomethanes"],
        "turbidity": data["Turbidity"],
    }

    response = requests.post(
        GRAPHQL_URL,
        json={
            "query": GRAPHQL_QUERY,
            "variables": variables,
        },
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    result = response.json()

    if "errors" in result:
        raise RuntimeError(result["errors"])

    return result["data"]["predict"]

def test_grpc(data: dict) -> dict:
    channel = grpc.insecure_channel(GRPC_ADDRESS)

    stub = scoring_pb2_grpc.ScoringStub(channel)

    metadata = (
        ("x-api-key", API_KEY),
    )

    request = scoring_pb2.Features(
        ph=data["ph"],
        hardness=data["Hardness"],
        solids=data["Solids"],
        chloramines=data["Chloramines"],
        sulfate=data["Sulfate"],
        conductivity=data["Conductivity"],
        organic_carbon=data["Organic_carbon"],
        trihalomethanes=data["Trihalomethanes"],
        turbidity=data["Turbidity"],
    )

    response = stub.Predict(
        request,
        metadata=metadata,
    )

    return {
        "prediction": response.prediction,
        "label": response.label,
        "probability": response.probability,
        "model_version": response.model_version,
    }

def main():
    test_cases = {
        "potable": "potable.json",
        "non_potable": "non_potable.json",
    }

    for case_name, filename in test_cases.items():

        print(f"\n=== Caso: {case_name} ===")

        data = load_test_case(filename)

        fastapi_result = test_fastapi(data)

        print(
            f"FastAPI -> "
            f"{fastapi_result['label']} "
            f"(prediction={fastapi_result['prediction']}, "
            f"probability={fastapi_result['probability']:.4f}, "
            f"model_version={fastapi_result['model_version']})"
        )
    
        graphql_result = test_graphql(data)

        print(
            f"GraphQL -> "
            f"{graphql_result['label']} "
            f"(prediction={graphql_result['prediction']}, "
            f"probability={graphql_result['probability']:.4f}, "
            f"model_version={graphql_result['modelVersion']})"
        )

        grpc_result = test_grpc(data)

        print(
            f"gRPC -> "
            f"{grpc_result['label']} "
            f"(prediction={grpc_result['prediction']}, "
            f"probability={grpc_result['probability']:.4f}, "
            f"model_version={grpc_result['model_version']})"
        )

if __name__ == "__main__":
    main()