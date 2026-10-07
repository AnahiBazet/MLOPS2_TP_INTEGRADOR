import os
from concurrent import futures

import grpc

import scoring_pb2
import scoring_pb2_grpc

from mlops2_tp.model_loader import load_champion_model
from mlops2_tp.inference import (
    load_data_dictionary,
    predict_water_potability,
)


# ============================================================
# Inicialización del modelo y de los datos auxiliares
# ============================================================

# El modelo Champion se carga una única vez cuando arranca
# el servidor gRPC. No se vuelve a cargar en cada predicción.
model, model_version = load_champion_model()

# Se carga también una única vez la información necesaria
# para la inferencia, por ejemplo las medianas utilizadas
# para completar valores faltantes.
data_dict = load_data_dictionary()


# ============================================================
# API Key
# ============================================================

# La API key se obtiene de una variable de entorno.
# En Docker Compose se le pasará la misma API_KEY que usan
# FastAPI y GraphQL.
API_KEY = os.getenv("API_KEY")


def verify_api_key(context):
    """
    Valida la API key enviada por el cliente mediante metadata gRPC.
    """

    # gRPC no utiliza headers HTTP de la misma manera que REST.
    # La información adicional se envía mediante metadata.
    metadata = dict(context.invocation_metadata())

    api_key = metadata.get("x-api-key")

    # Si falta la clave o no coincide con la configurada,
    # se cancela la llamada gRPC.
    if not API_KEY or api_key != API_KEY:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "API key inexistente o inválida.",
        )


# ============================================================
# Conversión del request gRPC al formato usado por inferencia
# ============================================================

def request_to_features(request):
    """
    Convierte el mensaje Features recibido por gRPC en el
    diccionario esperado por predict_water_potability().
    Si algún campo no está presente en el mensaje gRPC, se asigna None.
    """

    return {
        "ph": request.ph if request.HasField("ph") else None,
        "Hardness": request.hardness if request.HasField("hardness") else None,
        "Solids": request.solids if request.HasField("solids") else None,
        "Chloramines": request.chloramines if request.HasField("chloramines") else None,
        "Sulfate": request.sulfate if request.HasField("sulfate") else None,
        "Conductivity": request.conductivity if request.HasField("conductivity") else None,
        "Organic_carbon": request.organic_carbon if request.HasField("organic_carbon") else None,
        "Trihalomethanes": request.trihalomethanes if request.HasField("trihalomethanes") else None,
        "Turbidity": request.turbidity if request.HasField("turbidity") else None,
    }


# ============================================================
# Lógica común de predicción
# ============================================================

def make_prediction(request):
    """
    Ejecuta la inferencia para una muestra y construye
    el mensaje Prediction definido en scoring.proto.
    """

    features = request_to_features(request)

    # Se reutiliza exactamente la misma función de inferencia
    # que utilizan FastAPI y GraphQL.
    prediction, probability = predict_water_potability(
        model=model,
        features=features,
        data_dictionary=data_dict,
    )

    label = "potable" if prediction == 1 else "no potable"

    # Se construye la respuesta gRPC utilizando el mensaje
    # Prediction generado automáticamente desde scoring.proto.
    return scoring_pb2.Prediction(
        prediction=prediction,
        label=label,
        probability=probability,
        model_version=model_version,
    )


# ============================================================
# Implementación del servicio definido en scoring.proto
# ============================================================

class ScoringServicer(scoring_pb2_grpc.ScoringServicer):

    def Predict(self, request, context):
        """
        Método unary:
        recibe una muestra y devuelve una única predicción.
        """

        verify_api_key(context)

        return make_prediction(request)

    def PredictStream(self, request, context):
        """
        Método server-streaming:
        recibe un lote de muestras y devuelve las predicciones
        una por una mediante streaming.
        """

        verify_api_key(context)

        for instance in request.instances:
            yield make_prediction(instance)


# ============================================================
# Inicio del servidor gRPC
# ============================================================

def serve():
    """
    Crea, configura e inicia el servidor gRPC.
    """

    # Se crea un servidor capaz de atender varias llamadas
    # utilizando un pool de threads.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10)
    )

    # Se registra la implementación ScoringServicer
    # dentro del servidor gRPC.
    scoring_pb2_grpc.add_ScoringServicer_to_server(
        ScoringServicer(),
        server,
    )

    # El servidor escucha en el puerto estándar que veníamos
    # utilizando para gRPC en el TP: 50051.
    #
    # "insecure" significa que no estamos utilizando TLS.
    server.add_insecure_port("[::]:50051")

    # Se inicia el servidor.
    server.start()
    
    print("gRPC server running on port 50051", flush=True)

    # Mantiene el proceso en ejecución esperando solicitudes.
    server.wait_for_termination()


# ============================================================
# Punto de entrada
# ============================================================

if __name__ == "__main__":
    serve()