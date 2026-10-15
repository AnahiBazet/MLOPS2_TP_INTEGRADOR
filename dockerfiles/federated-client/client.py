from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .model import LogisticRegressionModel
from .data import load_data


# Define la aplicación del cliente federado.
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Ejecuta una ronda de entrenamiento local.

    1. Carga los datos exclusivos del laboratorio.
    2. Inicializa el modelo de regresión logística.
    3. Recibe los parámetros del modelo global desde el servidor.
    4. Entrena localmente durante 5 épocas.
    5. Devuelve los parámetros actualizados y la cantidad de muestras.

    El servidor utilizará esta información para aplicar FedAvg.
    """

    # Carga los datos locales del laboratorio.
    X_train, y_train = load_data()

    # Inicializa la regresión logística.
    model = LogisticRegressionModel(
        n_features=X_train.shape[1],
        learning_rate=0.1
    )

    # Extrae los parámetros globales recibidos desde el servidor.
    parameters = msg.content["arrays"].to_numpy_ndarrays()

    # Actualiza el modelo local con los parámetros globales.
    model.set_parameters(parameters)

    # Entrena exclusivamente con los datos del laboratorio.
    model.fit(
        X_train,
        y_train,
        epochs=5
    )

    # Convierte los parámetros actualizados al formato de Flower.
    model_record = ArrayRecord(model.get_parameters())

    # Informa la cantidad de muestras locales.
    # FedAvg utiliza este valor para ponderar los parámetros de cada cliente.
    metric_record = MetricRecord({
        "num-examples": len(y_train)
    })

    # Prepara la respuesta que se enviará al servidor.
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record
    })

    return Message(content=content, reply_to=msg)