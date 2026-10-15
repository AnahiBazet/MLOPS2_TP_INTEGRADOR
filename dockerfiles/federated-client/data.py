import os

import numpy as np
import pandas as pd


def load_data():
    """Carga los datos locales del laboratorio."""

    data_path = os.getenv(
        "FEDERATED_DATA_PATH",
        "/app/data/train.csv"
    )

    dataset = pd.read_csv(data_path)

    X = dataset.drop(columns=["Potability"]).to_numpy(
        dtype=np.float64
    )

    y = dataset["Potability"].to_numpy(
        dtype=np.float64
    )

    return X, y