from pathlib import Path

import numpy as np

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .evaluate import evaluate_validation, evaluate_test
from .plots import (
    generate_plots,
    generate_classification_plot,
    generate_final_test_barplot,
)


# Cantidad de variables predictoras de Water Potability.
NUM_FEATURES = 9

# Cantidad de rondas de entrenamiento federado.
NUM_ROUNDS = 20

# Ruta para guardar el modelo global final.
MODEL_PATH = Path("/app/results/global_model.npz")


# Define la aplicación del servidor Flower.
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Coordina el entrenamiento federado y la evaluación centralizada.

    1. Inicializa los parámetros globales.
    2. Configura FedAvg para los dos laboratorios.
    3. Ejecuta 20 rondas de entrenamiento.
    4. Evalúa sobre validation.csv después de cada ronda.
    5. Guarda el modelo global final.
    6. Evalúa una sola vez sobre test.csv.
    """

    # Inicializa los coeficientes y el intercepto en cero.
    initial_parameters = [
        np.zeros(NUM_FEATURES, dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    ]

    initial_arrays = ArrayRecord(initial_parameters)

    # Configura la estrategia federada.
    strategy = FedAvg(
        fraction_train=1.0,          # Participan todos los clientes disponibles.
        min_train_nodes=2,           # Requiere dos clientes para entrenar.
        min_available_nodes=2,       # Espera hasta tener dos clientes disponibles.
        fraction_evaluate=0.0,       # Desactiva la evaluación distribuida.
        weighted_by_key="num-examples",  # FedAvg ponderado por cantidad de muestras.
    )

    # Ejecuta las rondas de entrenamiento federado.
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=NUM_ROUNDS,

        # Evalúa el modelo global con validation.csv:
        # - Antes de entrenar (ronda 0).
        # - Después de cada ronda de FedAvg.
        evaluate_fn=evaluate_validation,
    )

    # Recupera los parámetros finales del modelo global.
    weights, bias = result.arrays.to_numpy_ndarrays()

    # Guarda los coeficientes y el intercepto del modelo final.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        MODEL_PATH,
        weights=weights,
        bias=bias,
    )

    print(f"Modelo global guardado en: {MODEL_PATH}")

    # Evalúa una sola vez el modelo global sobre test.
    final_test_metrics = evaluate_test(result.arrays)

    # Genera los gráficos de validación y evaluación final.
    try:
        generate_plots()
        generate_classification_plot()
        generate_final_test_barplot(dict(final_test_metrics))
    except OSError as exc:
        print(f"No se pudieron guardar los gráficos: {exc}")


if __name__ == "__main__":
    main()