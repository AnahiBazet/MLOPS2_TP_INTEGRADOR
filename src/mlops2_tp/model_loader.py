import mlflow


MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "water_potability_model_prod"
MODEL_ALIAS = "champion"


def load_champion_model():
    """
    Load the model registered in MLflow with the Champion alias.

    Returns
    -------
    model
        Loaded MLflow sklearn model.

    version
        Version number of the registered model.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    model = mlflow.sklearn.load_model(
        model_version.source
    )

    version = int(model_version.version)

    return model, version