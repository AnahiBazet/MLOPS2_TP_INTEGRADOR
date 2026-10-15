"""Prepara datos para el experimento federado de Water Potability.

Este paso de inicialización utiliza datos centralizados de MinIO de manera
intencional. El preprocesamiento federado queda fuera del alcance del TP.
Los clientes Flower entrenarán luego exclusivamente con su partición IID.
"""

import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


SOURCE_URI = os.getenv("FEDERATED_SOURCE_URI", "s3://data/raw/water_potability.csv")
CLIENT_DATA_DIR = Path("/app/client-data")
SERVER_DATA_DIR = Path("/app/server-data")
TARGET_COLUMN = "Potability"
TEST_SIZE = float(os.getenv("FEDERATED_TEST_SIZE", "0.2"))
VALIDATION_SIZE = float(os.getenv("FEDERATED_VALIDATION_SIZE", "0.1"))
RANDOM_SEED = int(os.getenv("FEDERATED_RANDOM_SEED", "42"))


def read_s3_csv(uri: str) -> pd.DataFrame:
    """Lee un CSV desde MinIO usando las credenciales S3 del contenedor."""
    if not uri.startswith("s3://"):
        raise ValueError("FEDERATED_SOURCE_URI debe comenzar con s3://")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not separator or not bucket or not key:
        raise ValueError(f"Ruta S3 inválida: {uri}")

    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        return pd.read_csv(response["Body"])
    finally:
        response["Body"].close()


def main() -> None:
    if not 0 < TEST_SIZE < 1:
        raise ValueError("FEDERATED_TEST_SIZE debe estar entre 0 y 1")
    if not 0 < VALIDATION_SIZE < 1 - TEST_SIZE:
        raise ValueError("FEDERATED_VALIDATION_SIZE debe ser positiva y dejar datos para train")

    df = read_s3_csv(SOURCE_URI)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo: {TARGET_COLUMN}")

    # Misma limpieza básica que el ETL existente.
    df = df.drop_duplicates().dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError("Potability debe contener ambas clases: 0 y 1")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Todas las variables predictoras deben ser numéricas")

    # Separamos primero el test; nunca participa en el ajuste de transformaciones.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Del conjunto restante separamos validation (10 % del total por defecto).
    # Así, con TEST_SIZE=0.2 y VALIDATION_SIZE=0.1, train conserva el 70 %.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        stratify=y_train_val,
        random_state=RANDOM_SEED,
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Imputer y scaler se ajustan solo con train: no hay fuga de información.
    X_train_imputed = imputer.fit_transform(X_train)
    X_validation_imputed = imputer.transform(X_validation)
    X_test_imputed = imputer.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_validation_scaled = scaler.transform(X_validation_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Un único preprocesamiento común, calculado centralizadamente en init.
    columns = X.columns.tolist()
    train_df = pd.DataFrame(X_train_scaled, columns=columns, index=X_train.index)
    train_df[TARGET_COLUMN] = y_train
    validation_df = pd.DataFrame(
        X_validation_scaled, columns=columns, index=X_validation.index
    )
    validation_df[TARGET_COLUMN] = y_validation
    test_df = pd.DataFrame(X_test_scaled, columns=columns, index=X_test.index)
    test_df[TARGET_COLUMN] = y_test

    # Partición IID aproximada: asignación aleatoria y estratificada por clase.
    client_a, client_b = train_test_split(
        train_df,
        test_size=0.5,
        stratify=train_df[TARGET_COLUMN],
        random_state=RANDOM_SEED,
    )


    # Conservar las cantidades originales para ponderar FedAvg.
    original_sizes = {
        "client_a": len(client_a),
        "client_b": len(client_b),
    }

    # Aplicar SMOTE independientemente a cada laboratorio.
    def apply_smote(client_df):
        X_client = client_df.drop(columns=[TARGET_COLUMN])
        y_client = client_df[TARGET_COLUMN]

        smote = SMOTE(random_state=RANDOM_SEED)
        X_resampled, y_resampled = smote.fit_resample(X_client, y_client)

        result = pd.DataFrame(X_resampled, columns=X_client.columns)
        result[TARGET_COLUMN] = np.asarray(y_resampled)
        return result

    client_a = apply_smote(client_a)
    client_b = apply_smote(client_b)

    paths = {
        "client_a": CLIENT_DATA_DIR / "client_a" / "train.csv",
        "client_b": CLIENT_DATA_DIR / "client_b" / "train.csv",
        "validation": SERVER_DATA_DIR / "validation.csv",
        "test": SERVER_DATA_DIR / "test.csv",
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    client_a.to_csv(paths["client_a"], index=False)
    client_b.to_csv(paths["client_b"], index=False)
    validation_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)

    print(f"Fuente: {SOURCE_URI}")
    for name, dataframe in (
        ("client_a", client_a),
        ("client_b", client_b),
        ("validation", validation_df),
        ("test", test_df),
    ):
        counts = dataframe[TARGET_COLUMN].value_counts().sort_index().to_dict()
        print(f"{name}: {len(dataframe)} registros | clases: {counts} | {paths[name]}")
    print("Preparación finalizada. Transformaciones ajustadas solo sobre train.")


if __name__ == "__main__":
    main()
