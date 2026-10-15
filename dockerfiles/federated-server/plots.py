
"""Genera gráficos de las métricas de validación del entrenamiento federado."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Generación de imágenes sin interfaz gráfica.
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("/app/results")
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "validation_metrics.csv"


def generate_plots() -> None:
    """Guarda gráficos de las métricas acumuladas por ronda federada."""
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics["round"] > 0]   
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plots = {
        "loss": ("Log loss", "loss.png"),
        "accuracy": ("Accuracy", "accuracy.png"),
        "precision": ("Precision - clase potable", "precision.png"),
        "recall": ("Recall - clase potable", "recall.png"),
        "f1": ("F1-score - clase potable", "f1.png"),
        "roc_auc": ("ROC-AUC", "roc_auc.png"),
    }

    for metric, (title, filename) in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(metrics["round"], metrics[metric], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Ronda federada")
        ax.set_ylabel(title)
        ax.set_xticks(metrics["round"][::2])
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=200)
        plt.close(fig)

def generate_classification_plot() -> None:
    """Grafica accuracy, precision, recall y F1 del modelo global por ronda."""
    metrics = pd.read_csv(METRICS_PATH)

    # Excluir la ronda 0 (modelo sin entrenar).
    metrics = metrics[metrics["round"] > 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    for metric in ["accuracy", "precision", "recall", "f1"]:
        ax.plot(
            metrics["round"],
            metrics[metric],
            marker="o",
            markersize=3,
            label=metric.capitalize() if metric != "f1" else "F1-score",
        )

    ax.set_title("Validación del modelo global federado (FedAvg)")
    ax.set_xlabel("Ronda federada")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0.3, 0.6)
    ax.set_xticks(range(1, 21))
    ax.grid(True, alpha=0.3)
    ax.legend()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "classification_metrics.png", dpi=200)
    plt.close(fig)

def generate_final_test_barplot(test_metrics: dict) -> None:
    """
    Genera un gráfico de barras con las métricas finales de test.
    Se muestran accuracy, precision, recall, f1 y roc_auc.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    metric_values = [test_metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(metric_labels, metric_values)

    ax.set_title("Evaluación final en test del modelo global federado")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1)

    for i, value in enumerate(metric_values):
        ax.text(i, value + 0.01, f"{value:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "final_test_metrics_barplot.png", dpi=200)
    plt.close(fig)