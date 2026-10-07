import json
import os
from pathlib import Path

import grpc
from dotenv import load_dotenv

import scoring_pb2
import scoring_pb2_grpc


# ============================================================
# Configuración
# ============================================================

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY no está definida en el archivo .env")


GRPC_ADDRESS = "localhost:50051"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_CASES_DIR = (
    PROJECT_ROOT
    / "src"
    / "mlops2_tp"
    / "test_cases"
)


# ============================================================
# Funciones auxiliares
# ============================================================

def load_test_case(filename: str) -> dict:
    """
    Carga uno de los casos JSON de prueba.
    """

    file_path = TEST_CASES_DIR / filename

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def dict_to_grpc_features(data: dict):
    """
    Convierte el diccionario del JSON al mensaje Features
    definido en scoring.proto.
    """

    return scoring_pb2.Features(
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


# ============================================================
# Predicción unary
# ============================================================

def test_predict(stub, metadata, data: dict):
    """
    Envía una sola muestra mediante el método unary Predict.
    """

    request = dict_to_grpc_features(data)

    response = stub.Predict(
        request,
        metadata=metadata,
    )

    return response


# ============================================================
# Predicción mediante server-streaming
# ============================================================

def test_predict_stream(stub, metadata, data_list: list[dict]):
    """
    Envía varias muestras y recibe las predicciones
    una por una mediante server-streaming.
    """

    instances = [
        dict_to_grpc_features(data)
        for data in data_list
    ]

    request = scoring_pb2.BatchFeatures(
        instances=instances,
    )

    responses = stub.PredictStream(
        request,
        metadata=metadata,
    )

    return responses


# ============================================================
# Cliente
# ============================================================

def main():

    channel = grpc.insecure_channel(GRPC_ADDRESS)

    stub = scoring_pb2_grpc.ScoringStub(channel)

    metadata = (
        ("x-api-key", API_KEY),
    )

    potable = load_test_case("potable.json")
    non_potable = load_test_case("non_potable.json")


    # --------------------------------------------------------
    # Unary
    # --------------------------------------------------------

    print("\n=== Predict unary ===")

    response = test_predict(
        stub,
        metadata,
        potable,
    )

    print(
        f"gRPC -> "
        f"{response.label} "
        f"(prediction={response.prediction}, "
        f"probability={response.probability:.4f}, "
        f"model_version={response.model_version})"
    )


    # --------------------------------------------------------
    # Server-streaming
    # --------------------------------------------------------

    print("\n=== PredictStream ===")

    responses = test_predict_stream(
        stub,
        metadata,
        [potable, non_potable],
    )

    for i, response in enumerate(responses, start=1):
        print(
            f"Predicción {i} -> "
            f"{response.label} "
            f"(prediction={response.prediction}, "
            f"probability={response.probability:.4f}, "
            f"model_version={response.model_version})"
        )


if __name__ == "__main__":
    main()