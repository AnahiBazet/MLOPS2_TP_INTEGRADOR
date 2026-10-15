import csv
from pathlib import Path

import numpy as np
import pandas as pd

from flwr.app import ArrayRecord, MetricRecord
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Archivos generados previamente por federated-init.
DATA_DIR = Path("/app/data")
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Directorio para almacenar los resultados de evaluación.
RESULTS_DIR = Path("/app/results")
VALIDATION_METRICS_PATH = RESULTS_DIR / "validation_metrics.csv"
TEST_METRICS_PATH = RESULTS_DIR / "test_metrics.csv"

TARGET_COLUMN = "Potability"


def load_dataset(path: Path):
    """
    Carga el dataset de evaluación.

    Los datos ya fueron imputados y estandarizados
    por federated-init utilizando estadísticas de train.
    """
    df = pd.read_csv(path)

    X = df.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)

    return X, y


def calculate_metrics(arrays: ArrayRecord, X, y) -> dict:
    """
    Evalúa la regresión logística utilizando los parámetros globales.

    1. Recupera los coeficientes y el intercepto.
    2. Calcula las probabilidades de potabilidad.
    3. Clasifica con un umbral de 0.5.
    4. Calcula las métricas de clasificación binaria.

    La clase positiva (1) representa agua potable.
    """

    weights, bias = arrays.to_numpy_ndarrays()

    # Calcula las probabilidades mediante la función sigmoide.
    z = X @ weights + float(bias.item())
    z = np.clip(z, -500, 500)

    probabilities = 1.0 / (1.0 + np.exp(-z))

    # Convierte las probabilidades en predicciones binarias.
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def save_metrics(path: Path, metrics: dict) -> None:
    """
    Guarda las métricas en un archivo CSV.

    Si el archivo no existe, crea el encabezado.
    Si ya existe, incorpora un nuevo registro.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


def evaluate_validation(
    server_round: int,
    arrays: ArrayRecord
) -> MetricRecord:
    """
    Evaluación centralizada ejecutada por Flower.

    - Utiliza exclusivamente validation.csv.
    - Se ejecuta antes de entrenar (ronda 0).
    - Se ejecuta después de cada ronda de FedAvg.
    - Registra las métricas para generar gráficos.
    """

    X_val, y_val = load_dataset(VALIDATION_PATH)

    metrics = calculate_metrics(arrays, X_val, y_val)

    # Al comenzar un nuevo entrenamiento, reinicia el historial.
    if server_round == 0:
        VALIDATION_METRICS_PATH.unlink(missing_ok=True)

    # Agrega el número de ronda al registro de métricas.
    record = {
        "round": server_round,
        **metrics,
    }

    save_metrics(VALIDATION_METRICS_PATH, record)

    return MetricRecord(metrics)


def evaluate_test(arrays: ArrayRecord) -> MetricRecord:
    """
    Evaluación final del modelo global.

    - Utiliza exclusivamente test.csv.
    - Debe ejecutarse una sola vez, después del entrenamiento.
    - Guarda las métricas finales en un archivo independiente.
    """

    X_test, y_test = load_dataset(TEST_PATH)

    metrics = calculate_metrics(arrays, X_test, y_test)

    # Reemplaza las métricas de test de cualquier ejecución anterior.
    TEST_METRICS_PATH.unlink(missing_ok=True)
    save_metrics(TEST_METRICS_PATH, metrics)

    print(f"Evaluación final - Test: {metrics}")

    return MetricRecord(metrics)