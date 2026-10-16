# Water Potability MLOps

Sistema MLOps para clasificación de potabilidad del agua a partir de características fisicoquímicas.

La solución integra procesamiento de datos, inferencia mediante REST, GraphQL y gRPC, procesamiento de eventos en tiempo real con Kafka y aprendizaje federado con Flower. Además, incorpora mecanismos de seguridad y una infraestructura basada en Airflow, MLflow, MinIO y Docker.

## Índice

- [1. Descripción del proyecto](#1-descripción-del-proyecto)
- [2. Arquitectura del sistema](#2-arquitectura-del-sistema)
- [3. Tecnologías utilizadas](#3-tecnologías-utilizadas)
- [4. Dataset y problema de clasificación](#4-dataset-y-problema-de-clasificación)
- [5. Pipeline de datos en Airflow](#5-pipeline-de-datos-en-airflow)
- [6. Entrenamiento y selección del modelo en Airflow](#6-entrenamiento-y-selección-del-modelo-en-airflow)
- [7. Gestión de experimentos con MLflow](#7-gestión-de-experimentos-con-mlflow)
- [8. Modelo productivo almacenado en MinIO](#8-modelo-productivo-almacenado-en-minio)
- [9. MLOps Water Potability Laboratory: REST, GraphQL y gRPC](#9-mlops-water-potability-laboratory-rest-graphql-y-grpc)
- [10. MLOps Water Potability Streaming](#10-mlops-water-potability-streaming)
- [11. Seguridad](#11-seguridad)
- [12. MLOps Water Potability Federated](#12-mlops-water-potability-federated)
- [13. Instalación y ejecución](#13-instalación-y-ejecución)
- [14. Estructura del proyecto](#14-estructura-del-proyecto)
- [15. Conclusiones y posibles mejoras](#15-conclusiones-y-posibles-mejoras)

## 1. Descripción del proyecto

Water Potability MLOps es un sistema de machine learning orientado a la clasificación de muestras de agua según su potabilidad. A partir de un conjunto de características fisicoquímicas, el modelo determina si una muestra puede clasificarse como potable o no potable.

Implementa un flujo MLOps de punta a punta que contempla la ingesta, preparación y almacenamiento de datos mediante Airflow y MinIO, el entrenamiento y optimización de modelos, el seguimiento de experimentos y versionado mediante MLflow y la selección del modelo productivo para su utilización en distintos escenarios de inferencia.

La solución se organiza en tres componentes principales:

- **MLOps Water Potability Laboratory**: orientado a la inferencia de muestras mediante REST, GraphQL y gRPC.
- **MLOps Water Potability Streaming**: orientado al procesamiento continuo de mediciones mediante Kafka.
- **MLOps Water Potability Federated**: una prueba de concepto de aprendizaje federado con Flower que permite entrenar un modelo global a partir de datos distribuidos entre distintos clientes.

El sistema incorpora además mecanismos de seguridad para controlar el acceso a los servicios y validar los eventos procesados.

La infraestructura se ejecuta mediante contenedores Docker y los distintos servicios se integran y administran utilizando Docker Compose. Airflow se utiliza para la orquestación de pipelines, MLflow para el seguimiento de experimentos y gestión del modelo productivo y MinIO como almacenamiento de datos y artefactos.

## 2. Arquitectura del sistema

El proyecto se organiza en tres arquitecturas complementarias, cada una orientada a un escenario distinto de uso:

- **MLOps Water Potability Laboratory**: inferencia mediante REST, GraphQL y gRPC.
- **MLOps Water Potability Streaming**: procesamiento continuo de eventos mediante Kafka.
- **MLOps Water Potability Federated**: entrenamiento federado con Flower sobre datos distribuidos entre distintos clientes.

Todas están desplegadas con Docker Compose.

### 2.1 MLOps Water Potability Laboratory

La solución se implementa como una arquitectura modular compuesta por servicios de orquestación, procesamiento de datos, entrenamiento, almacenamiento, seguimiento de experimentos, inferencia y despliegue en contenedores.

![Arquitectura del sistema Water Potability Laboratory](Arquitectura%20Water%20Potability%20Laboratory%20v3.png)

Componentes principales:

- **Apache Airflow**: orquestación de los pipelines de datos y entrenamiento.
- **MinIO**: almacenamiento de datasets y artefactos.
- **MLflow**: seguimiento de experimentos y Model Registry.
- **PostgreSQL**: persistencia de metadatos de Airflow y MLflow.
- **Capa de inferencia**: carga el modelo productivo y ejecuta predicciones.
- **REST, GraphQL y gRPC**: servicios de inferencia protegidos mediante API Key.

### 2.2 MLOps Water Potability Streaming

Arquitectura orientada al procesamiento en tiempo real de mediciones provenientes de estaciones de monitoreo.

![Arquitectura del sistema Water Potability Streaming](Arquitectura%20Water%20Potability%20Streaming%20v2.png)

Componentes principales:

- **Productores**: generan y envían eventos de medición.
- **Apache Kafka**: recibe los eventos en el tópico `water.measurements`, dividido en dos particiones.
- **Consumidor**: valida los eventos y ejecuta la inferencia.
- **Capa de inferencia**: reutiliza el modelo productivo.
- **MinIO**: almacena las predicciones generadas en streaming.
- **Airflow y MLflow**: gestionan el entrenamiento y versionado del modelo utilizado para inferencia.
- **PostgreSQL**: persistencia de metadatos de Airflow y MLflow.

### 2.3 MLOps Water Potability Federated

Prueba de concepto de aprendizaje federado con Flower ejecutada mediante contenedores de Docker, orientada al entrenamiento colaborativo sin centralizar los datos de los clientes.

![Arquitectura del sistema Water Potability Federated](Arquitectura%20Water%20Potability%20Federated%20v2.png)

Componentes principales:

- **Federated Init**: prepara y distribuye los datos utilizados por clientes y servidor.
- **Laboratorio A y Laboratorio B**: realizan entrenamiento local sobre sus propios datos.
- **ClientApp / SuperExec Client**: ejecuta el entrenamiento local en cada cliente.
- **SuperNode**: gestiona la comunicación federada de cada cliente.
- **SuperLink**: coordina la infraestructura Flower.
- **ServerApp / SuperExec Server**: ejecuta FedAvg, coordina las rondas y evalúa el modelo global (métricas y gráficos).

## 3. Tecnologías utilizadas

El proyecto integra distintas tecnologías para cubrir el procesamiento de datos, entrenamiento, evaluación y gestión de modelos, almacenamiento, despliegue, inferencia, streaming, seguridad y aprendizaje federado.

| Tecnología | Uso en el proyecto |
|---|---|
| **Python** | Desarrollo de pipelines, entrenamiento, inferencia, APIs, procesamiento de streaming y aprendizaje federado. |
| **Apache Airflow** | Orquestación de los pipelines de procesamiento de datos y entrenamiento del modelo. |
| **MLflow** | Seguimiento de experimentos, registro de métricas y parámetros y gestión de versiones del modelo mediante Model Registry. |
| **Optuna** | Búsqueda y optimización de hiperparámetros. |
| **imbalanced-learn / SMOTE** | Balanceo de la clase minoritaria durante el entrenamiento. |
| **MinIO** | Almacenamiento compatible con S3 para datasets, información de preprocesamiento, artefactos de MLflow, predicciones de streaming y eventos rechazados almacenados en la DLQ. |
| **PostgreSQL** | Backend de persistencia utilizado por Airflow y MLflow. |
| **Apache Kafka** | Plataforma de mensajería utilizada para el procesamiento de eventos de medición en tiempo real mediante el tópico `water.measurements`, dividido en dos particiones. |
| **Flower** | Framework utilizado para implementar el aprendizaje federado, coordinando clientes, servidor y entrenamiento del modelo global. |
| **Docker** | Contenerización de los distintos componentes del sistema. |
| **Docker Compose** | Orquestación e integración de los servicios que conforman la infraestructura. |
| **FastAPI** | Exposición de la API REST de inferencia. |
| **Strawberry GraphQL** | Implementación de la API GraphQL. |
| **gRPC** | Comunicación para inferencia entre sistemas, incluyendo llamadas unary y server streaming. |
| **Protocol Buffers** | Definición de mensajes y servicios utilizados por gRPC. |
| **API Key** | Control de acceso aplicado a los servicios REST, GraphQL y gRPC. |
| **HMAC-SHA256** | Firma y verificación de eventos de streaming para validar su autenticidad e integridad. |
| **Pydantic** | Validación del esquema de los eventos procesados en streaming. |
| **Docker Logs** | Registro de auditoría de eventos aceptados y rechazados durante el procesamiento de streaming. |
| **uv** | Gestión del entorno Python y de las dependencias del proyecto. |
| **Git / GitHub** | Control de versiones y alojamiento del código fuente. |

## 4. Dataset y problema de clasificación

El proyecto utiliza el dataset **Water Potability**, disponible en [Kaggle]( https://www.kaggle.com/datasets/adityakadiwal/water-potability) (versión 3), que contiene mediciones fisicoquímicas de distintas muestras de agua. El objetivo es resolver un problema de **clasificación binaria**, determinando si una muestra de agua es potable o no potable.

La variable objetivo es `Potability`, cuyos valores representan:

- `0`: agua no potable.
- `1`: agua potable.

Para realizar la clasificación se utilizan las siguientes variables:

| Variable | Descripción |
|---|---|
| `ph` | Nivel de pH del agua. |
| `Hardness` | Dureza del agua. |
| `Solids` | Cantidad de sólidos disueltos. |
| `Chloramines` | Concentración de cloraminas. |
| `Sulfate` | Concentración de sulfatos. |
| `Conductivity` | Conductividad eléctrica del agua. |
| `Organic_carbon` | Cantidad de carbono orgánico. |
| `Trihalomethanes` | Concentración de trihalometanos. |
| `Turbidity` | Nivel de turbidez del agua. |

El dataset presenta valores faltantes en algunas de sus variables y un desbalance entre las clases, características que son consideradas posteriormente durante las etapas de preprocesamiento y entrenamiento.

Los datos son obtenidos desde Kaggle y posteriormente procesados por el pipeline de datos implementado con Apache Airflow.

## 5. Pipeline de datos en Airflow

El procesamiento de datos se orquesta mediante el DAG de Apache Airflow `process_etl_water_potability`, compuesto por tres etapas principales: obtención de los datos, separación del dataset e imputación de valores faltantes.

```text
obtain_original_data
        ↓
   split_dataset
        ↓
impute_missing_values
```
* Obtención de los datos: el dataset **Water Potability** se descarga desde Kaggle y se almacena sin modificaciones en MinIO, dentro del bucket `data`.
* Separación del dataset: se eliminan registros duplicados y observaciones con valores nulos en la variable objetivo. Luego se separan las variables predictoras de la variable objetivo y se generan los conjuntos de entrenamiento y prueba, utilizando una separación estratificada para conservar la proporción de clases.
* Imputación de valores faltantes: se imputan utilizando la mediana calculada exclusivamente sobre el conjunto de entrenamiento. Las mismas medianas se utilizan para transformar el conjunto de prueba. Finalmente, los datasets procesados se almacenan en MinIO y se genera un archivo `data.json` con información del preprocesamiento, incluyendo las variables utilizadas, las medianas calculadas, el tamaño del conjunto de prueba y la semilla empleada.

### 5.1 Organización de los datos en MinIO

| Etapa | Ruta en MinIO | Contenido |
|---|---|---|
| Datos originales | `data/raw/` | Dataset original descargado desde Kaggle: `water_potability.csv`. |
| Datos intermedios | `data/interim/` | Variables predictoras de entrenamiento y prueba antes de la imputación. |
| Datos finales | `data/final/` | Conjuntos de entrenamiento y prueba definitivos, incluyendo variables predictoras imputadas y variable objetivo. |
| Metadatos | `data/data_info/` | Archivo `data.json` con información del preprocesamiento y los parámetros utilizados. |

## 6. Entrenamiento y selección del modelo en Airflow

El entrenamiento se orquesta mediante el DAG `train_water_potability_model` de **Airflow**, que automatiza la búsqueda de hiperparámetros, el entrenamiento final de la mejor configuración, la evaluación del modelo, su registro en **MLflow** y la elección del Champion.

```text
search_hyperparameters
        ↓
 select_best_model
        ↓
  train_best_model
        ↓
   evaluate_model
        ↓
   register_model
        ↓
    set_champion
  ```

* Búsqueda de hiperparámetros: se utiliza Optuna para comparar modelos `DecisionTreeClassifier` y `RandomForestClassifier` mediante 80 trials. Cada configuración se evalúa utilizando validación cruzada estratificada de 5 folds y F1-score como métrica de optimización. Durante la validación, la imputación por mediana y el balanceo mediante `SMOTE` se aplican dentro de un pipeline, utilizando únicamente los datos de entrenamiento correspondientes a cada fold.

* Selección del mejor modelo: se selecciona la configuración con mayor F1-score medio obtenido en los 5 folds de la validación cruzada. Se conserva el tipo de modelo y los hiperparámetros elegidos por Optuna.

* Entrenamiento: la mejor configuración encontrada por Optuna se entrena nuevamente utilizando el conjunto completo de entrenamiento ya imputado por el pipeline de datos. Antes del ajuste del modelo, se aplica `SMOTE` para balancear las clases.

* Evaluación: el modelo final se evalúa sobre el conjunto de prueba utilizando precision, recall, F1-score y accuracy, con un umbral de clasificación de `0.5`.

* Registro y selección del Champion: el modelo se registra en **MLflow Model Registry** bajo el nombre `water_potability_model_prod`, generándose una nueva versión numerada en cada ejecución exitosa. Posteriormente, la nueva versión recibe el alias `champion` únicamente si supera el F1-score de test del Champion actual. En la primera ejecución, la primera versión registrada se asigna automáticamente como Champion.

## 7. Gestión de experimentos con MLflow

**MLflow** se utiliza para realizar el seguimiento de los experimentos de entrenamiento y mantener la trazabilidad de los modelos generados.

Durante la búsqueda de hiperparámetros, cada trial ejecutado por Optuna se registra como un run en **MLflow**, almacenando el tipo de modelo, los hiperparámetros evaluados y el F1-score obtenido mediante validación cruzada. Estos runs son hijos de un run principal que registra la mejor configuración encontrada por Optuna.

Posteriormente, el entrenamiento final del modelo seleccionado se registra en un run diferente, donde se almacenan los hiperparámetros utilizados y las métricas obtenidas sobre los conjuntos de entrenamiento y prueba, incluyendo precision, recall, F1-score y accuracy.

La información de tracking de MLflow se almacena en **PostgreSQL**, mientras que los artefactos asociados a los experimentos y modelos se almacenan en **MinIO**.

## 8. Modelo productivo almacenado en MinIO

El modelo productivo corresponde a la versión registrada en **MLflow Model Registry** que posee el alias `champion`.

Actualmente, el modelo seleccionado es un `RandomForestClassifier`, elegido a partir del proceso de optimización y evaluación descripto en las secciones anteriores.

Al iniciar los servicios de inferencia, se consulta **MLflow Model Registry** utilizando el nombre registrado `water_potability_model_prod` y el alias `champion` para identificar la versión productiva. A partir de esa versión, MLflow permite resolver la ubicación del artefacto asociado, que se encuentra almacenado físicamente en **MinIO**.

El modelo y su número de versión quedan disponibles para ser utilizados por las APIs REST, GraphQL y gRPC y por el consumidor de Kafka.

## 9. MLOps Water Potability Laboratory: REST, GraphQL y gRPC

El modelo productivo se expone mediante tres mecanismos de comunicación: una API REST desarrollada con **FastAPI**, una API **GraphQL** y un servicio **gRPC**.

Los tres servicios utilizan el mismo modelo identificado mediante el alias `champion` en **MLflow** y aplican la misma lógica de preparación de datos e inferencia. Cada API expone una forma diferente de consumir el modelo, permitiendo comparar distintos mecanismos de comunicación sobre una misma solución de machine learning.

### 9.1 REST - FastAPI

La API REST se implementa con **FastAPI** y expone un endpoint de predicción que recibe las características fisicoquímicas de una muestra de agua y determina si es o no agua potable.

Este servicio es utilizado por un sistema de gestión de información de laboratorio para consultar la potabilidad de una muestra una vez obtenidos sus valores fisicoquímicos. Por ejemplo, un científico o técnico de laboratorio carga los resultados de una prueba en una aplicación y esta aplicación envía los datos a la API REST para obtener una predicción.

La API se encuentra disponible en el puerto `8800` y expone los siguientes endpoints:

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Endpoint principal de la API. |
| `GET` | `/docs` | Interfaz Swagger para consultar y probar la API de forma interactiva. |
| `GET` | `/health` | Permite verificar el estado del servicio y la disponibilidad del modelo productivo. |
| `POST` | `/v1/predict` | Realiza una predicción de potabilidad a partir de las características de una muestra de agua. |

En el endpoint de inferencia `POST /v1/predict`, la solicitud contempla las nueve variables utilizadas por el modelo. Cuando alguna medición no se encuentra disponible, puede enviarse con valor `null`. Luego, ese valor se completa utilizando la mediana calculada sobre el conjunto de entrenamiento.

La respuesta incluye la clase predicha, su etiqueta, la probabilidad estimada y la versión del modelo productivo utilizada para realizar la inferencia.

La API puede probarse desde Swagger accediendo a:

`http://localhost:8800/docs`

También puede verificarse el estado del servicio en Powershell mediante:

```powershell
curl.exe http://localhost:8800/health
```

Para realizar una predicción desde PowerShell puede utilizarse:

```powershell
curl.exe -X POST "http://localhost:8800/v1/predict" `
  -H "Content-Type: application/json" `
  -H "X-API-KEY: $env:API_KEY" `
  -d '{
    \"ph\": 7.0,
    \"Hardness\": 200.0,
    \"Solids\": 20000.0,
    \"Chloramines\": 7.0,
    \"Sulfate\": 330.0,
    \"Conductivity\": 420.0,
    \"Organic_carbon\": 14.0,
    \"Trihalomethanes\": 70.0,
    \"Turbidity\": 4.0
  }'
```

 El acceso al endpoint de predicción se encuentra protegido mediante una API Key enviada en el header `X-API-KEY`. PowerShell obtiene su valor desde la variable de entorno `API_KEY` mediante `$env:API_KEY` y lo incorpora al header de la solicitud.

### 9.2 GraphQL

La API GraphQL se implementa con **Strawberry** sobre **FastAPI** y permite realizar consultas de predicción sobre el mismo modelo productivo utilizado por la API REST.

Este servicio se utiliza en un dashboard de un laboratorio que necesite mayor flexibilidad sobre la información devuelta. Por ejemplo, una interfaz puede solicitar únicamente la etiqueta de la clasificación (potable / no potable) y la probabilidad de que la muestra sea potable, sin necesidad de recuperar otros campos de la respuesta.

El servicio se encuentra disponible en el puerto `8801` y expone los siguientes endpoints:

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Endpoint principal del servicio. |
| `GET` | `/graphql` | Interfaz GraphiQL de Strawberry para consultar y probar el servicio de forma interactiva. |
| `GET` | `/health` | Permite verificar el estado del servicio y la versión del modelo productivo. |
| `POST` | `/graphql` | Ejecuta consultas GraphQL sobre el modelo de potabilidad. |


Las consultas se realizan mediante el campo `predict`, que recibe las características fisicoquímicas de la muestra. Los argumentos admiten valores nulos, que son imputados durante la inferencia utilizando las medianas calculadas sobre el conjunto de entrenamiento.

La respuesta permite seleccionar únicamente los campos que se desean recuperar, a diferencia de la API REST, cuya estructura de respuesta es fija. Entre los disponibles se encuentran:

- `prediction`: clase predicha, `1` para potable y `0` para no potable.
- `label`: etiqueta asociada a la clase predicha, `potable` o `no potable`.
- `probability`: probabilidad estimada de que la muestra sea potable.
- `modelVersion`: versión del modelo productivo utilizada para realizar la inferencia.

Una vez iniciado el servicio, la interfaz GraphiQL de Strawberry se encuentra disponible en:

`http://localhost:8801/graphql`

Se incluye una consulta de ejemplo en:

`dockerfiles/graphql/example_query.graphql`

Para ejecutarla debe agregarse el siguiente header:

```json
{
  "X-API-KEY": "<API_KEY>"
}
```

El valor de `<API_KEY>` debe coincidir con la variable `API_KEY` definida en el archivo `.env`.

También puede verificarse el estado del servicio en PowerShell mediante:

```powershell
curl.exe http://localhost:8801/health
```

Puede ejecutarse una consulta GraphQL desde PowerShell utilizando `curl.exe`. En este ejemplo se solicita únicamente `label` y `probability`:

```powershell
curl.exe -X POST "http://localhost:8801/graphql" `
  -H "Content-Type: application/json" `
  -H "X-API-KEY: $env:API_KEY" `
  -d '{
    \"query\": \"query { predict(ph: 7.0, hardness: 200.0, solids: 20000.0, chloramines: 7.0, sulfate: 330.0, conductivity: 420.0, organicCarbon: 14.0, trihalomethanes: 70.0, turbidity: 4.0) { label probability } }\"
  }'
  ```

El acceso a la operación `predict` se encuentra protegido mediante una API Key enviada en el header `X-API-KEY`. PowerShell obtiene su valor desde la variable de entorno `API_KEY` mediante `$env:API_KEY` y lo incorpora al header de la solicitud.

### 9.3 gRPC

El servicio gRPC se implementa con **gRPC** y **Protocol Buffers** y permite realizar predicciones con el mismo modelo productivo de las APIs REST y GraphQL.

Este servicio se utiliza para la comunicación automática entre equipos de laboratorio y el servicio de inferencia. Un analizador automático toma muestras de agua, obtiene sus mediciones fisicoquímicas y las envía al servicio gRPC para obtener la predicción de potabilidad sin intervención manual. El método `Predict` procesa una muestra individual, mientras que `PredictStream` recibe un lote de muestras y devuelve sus predicciones de forma secuencial mediante streaming.

El servicio se encuentra disponible en el puerto `50051` y expone los siguientes métodos:

| Método | Tipo | Descripción |
|---|---|---|
| `Predict` | Unary | Recibe una muestra y devuelve una única predicción. |
| `PredictStream` | Server-streaming | Recibe un conjunto de muestras y devuelve las predicciones de forma secuencial mediante un stream. |

La estructura de los mensajes y los métodos disponibles se define en el protocolo:

`dockerfiles/grpc/scoring.proto`

El mensaje `Features` contiene las nueve características fisicoquímicas utilizadas por el modelo. Los campos son opcionales y cuando una medición no se encuentra disponible, el valor faltante se imputa durante la inferencia utilizando las medianas calculadas sobre el conjunto de entrenamiento.

La respuesta `Prediction` incluye:

- `prediction`: clase predicha, `1` para potable y `0` para no potable.
- `label`: etiqueta asociada a la clase predicha, `potable` o `no potable`.
- `probability`: probabilidad estimada de que la muestra sea potable.
- `model_version`: versión del modelo productivo utilizada para realizar la inferencia.

A partir del archivo `scoring.proto` se generan los stubs de Python necesarios para implementar el cliente y el servidor gRPC:

- `scoring_pb2.py`
- `scoring_pb2_grpc.py`

Si se modifica el contrato definido en `scoring.proto`, los stubs deben regenerarse. Desde la carpeta raíz puede ejecutarse:

```powershell
python -m grpc_tools.protoc `
  -Idockerfiles/grpc `
  --python_out=dockerfiles/grpc `
  --grpc_python_out=dockerfiles/grpc `
  dockerfiles/grpc/scoring.proto
```

El proyecto incluye un cliente gRPC de ejemplo ubicado en un contenedor de forma independiente del servidor. Este cliente permite probar una predicción individual mediante `Predict` y el procesamiento de múltiples muestras mediante `PredictStream`.

El cliente se ejecuta mediante el perfil `test` de Docker Compose:

```powershell
docker compose --profile all --profile test run --rm --no-deps grpc-client
```

El acceso al servicio gRPC se encuentra protegido mediante una API Key enviada como metadata con la clave `x-api-key`. La API Key debe coincidir con la variable de entorno `API_KEY` configurada para el proyecto. El cliente obtiene este valor desde el entorno y lo incorpora automáticamente a la metadata de cada solicitud.

Si se desea eliminar la imagen del cliente gRPC:

```powershell
docker image rm mlops2_tp_integrador-grpc-client
```

### 9.4 Pruebas en los servicios de inferencia

El proyecto incluye un script unificado para verificar el funcionamiento de los tres servicios de inferencia implementados: REST, GraphQL y gRPC.

Las pruebas se ejecutan utilizando dos casos almacenados en:

- `src/mlops2_tp/test_cases/potable.json`
- `src/mlops2_tp/test_cases/non_potable.json`

Ambos casos corresponden a muestras del conjunto de test y contienen las nueve características fisicoquímicas requeridas por el modelo.

El script de prueba se encuentra en:

`src/mlops2_tp/test_all_api.py`

Desde la raíz del proyecto puede ejecutarse mediante:

```powershell
python src/mlops2_tp/test_all_api.py
```

El script envía las mismas muestras a los tres servicios de inferencia y permite verificar que utilizan el mismo modelo productivo y devuelven resultados consistentes.

Para cada caso se muestran:

- `prediction`: clase predicha, `1` para potable y `0` para no potable.
- `label`: etiqueta asociada a la clase predicha, `potable` o `no potable`.
- `probability`: probabilidad estimada de que la muestra sea potable.
- `model_version/modelVersion`: versión del modelo productivo utilizada para realizar la inferencia (`model_version` en REST y gRPC, `modelVersion` en GraphQL).

## 10. MLOps Water Potability Streaming

### 10.1 Procesamiento de eventos con Kafka

El componente de streaming permite procesar mediciones de calidad de agua de forma continua mediante Apache Kafka.

El productor genera eventos en formato JSON a partir de las mediciones de las estaciones `station_A` y `station_B` y los publica en el tópico: `water.measurements`

El tópico utiliza dos particiones:

- `partition 0`: recibe los eventos de `station_A`.
- `partition 1`: recibe los eventos de `station_B`.

El consumidor Kafka permanece suscripto al tópico y procesa los mensajes recibidos de manera continua. Para cada evento válido, recupera las características fisicoquímicas, ejecuta la inferencia utilizando el modelo productivo y almacena el resultado en MinIO.

Las predicciones se guardan en el bucket: `s3://inference/` organizadas por estación y fecha.

El consumidor utiliza el mismo modelo productivo identificado mediante el alias `champion` en MLflow que los servicios REST, GraphQL y gRPC.

El procesamiento utiliza confirmación manual de offsets. El offset se confirma después de que el mensaje fue procesado y persistido, ya sea como una predicción válida o como un evento rechazado almacenado en la DLQ.

### 10.2 Pruebas del servicio de streaming

Se verificaron los siguientes escenarios:

- Procesamiento correcto de eventos firmados, con inferencia, almacenamiento en MinIO y registro de auditoría.
- Rechazo de un evento con `ph = 15` y firma HMAC válida.
- Rechazo de un evento con datos válidos y firma HMAC incorrecta.
- Almacenamiento de eventos rechazados en MinIO.
- Activación de una alerta ante cinco rechazos en menos de 60 segundos.

Las pruebas permiten verificar el funcionamiento del pipeline de streaming, desde la recepción de mediciones y la ejecución de inferencias hasta el almacenamiento de resultados en MinIO. También permiten comprobar los mecanismos de validación, autenticidad, auditoría y alertas de seguridad implementados.

**Prueba de procesamiento de eventos válidos**

Para verificar el funcionamiento del pipeline de streaming, se ejecuta el productor Kafka, que envía las 656 mediciones del conjunto de test al tópico `water.measurements`, distribuyéndolas entre las estaciones `station_A` y `station_B`.

```powershell
docker compose --profile all --profile test run --rm --no-deps kafka-producer python producer.py
```

El consumidor verifica la firma HMAC y valida los datos mediante Pydantic antes de ejecutar la inferencia con el modelo productivo.

Los resultados se almacenan en MinIO, organizados por estación y fecha. Los logs del consumidor permiten verificar las predicciones y la información de auditoría asociada a cada evento.

**Ejecución de las pruebas de seguridad**

El proyecto incluye dos scripts que permiten verificar el funcionamiento de los controles de seguridad implementados en el consumidor Kafka. Ambos generan eventos de prueba que son enviados al tópico `water.measurements` para comprobar su validación y posterior tratamiento.

**Prueba 1. Validación de datos con Pydantic**

El script `dockerfiles/kafka-producer/test_invalid_event.py` envía un evento con una firma HMAC válida, pero con un valor de `ph = 15`, fuera del rango permitido.

```powershell
docker compose --profile all --profile test run --rm --no-deps kafka-producer python test_invalid_event.py
```

El consumidor verifica correctamente la firma HMAC, pero rechaza el evento durante la validación con Pydantic. El mensaje rechazado se almacena en la DLQ de MinIO junto con el motivo del rechazo.

**Prueba 2. Verificación de firma HMAC-SHA256**

El script `dockerfiles/kafka-producer/test_invalid_signature.py` envía un evento con datos válidos, pero con una firma HMAC incorrecta.

```powershell
docker compose --profile all --profile test run --rm --no-deps kafka-producer python test_invalid_signature.py
```

El consumidor detecta la firma inválida y rechaza el mensaje antes de ejecutar la validación con Pydantic y la inferencia. El evento también se almacena en la DLQ de MinIO.

**Consulta de resultados**

Los resultados de ambas pruebas pueden verificarse mediante los logs del consumidor:

```powershell
docker compose logs --tail=20 kafka-consumer
```

Los eventos rechazados se almacenan en `s3://inference/dlq/`, donde pueden consultarse sus datos originales y los motivos del rechazo.

**Prueba 3. Alertas de seguridad**

Para comprobar el mecanismo de alertas se ejecuta cinco veces la prueba de firma inválida, dentro de un intervalo de 60 segundos:

```powershell
1..5 | ForEach-Object {
    docker compose --profile all --profile test run --rm --no-deps kafka-producer python test_invalid_signature.py
}
```

Al alcanzar cinco eventos rechazados dentro de la ventana deslizante, el consumidor genera el siguiente mensaje en los logs de Docker:

```text
ALERTA DE SEGURIDAD | 5 eventos rechazados en los últimos 60 segundos
```

Estas pruebas permiten verificar el funcionamiento de la validación de datos, la autenticación mediante HMAC-SHA256, el almacenamiento de eventos rechazados y la generación de alertas de seguridad.

## 11. Seguridad

### 11.1 Seguridad en los servicios REST, GraphQL y gRPC

Los tres servicios de inferencia utilizan la misma API Key para controlar el acceso a las predicciones. La clave se configura mediante la variable de entorno `API_KEY`, definida en el archivo `.env`, evitando incorporarla directamente en el código fuente.

En las APIs REST y GraphQL, la API Key se envía mediante el header: `X-API-KEY`

En gRPC se envía como metadata utilizando la clave: `x-api-key`

Cada servicio valida la clave recibida antes de ejecutar la inferencia. Si la API Key no se encuentra presente o no coincide con la configurada en el entorno, la solicitud es rechazada.

### 11.2 Seguridad en el servicio de streaming

El pipeline de inferencia mediante Apache Kafka incorpora controles de seguridad. La siguiente figura resume el flujo de seguridad implementado, incluyendo los controles de autenticidad e integridad, validación de eventos, auditoría, cuarentena de mensajes rechazados y generación de alertas.

![Diagrama de seguridad en streaming Kafka](Diagrama%20de%20seguridad%20en%20streaming%20Kafka.png)

**1. Validación de eventos con Pydantic**

El consumidor utiliza Pydantic para validar la estructura de los mensajes, los tipos de datos, los identificadores, las fechas y los rangos permitidos para las nueve variables del modelo de potabilidad del agua. Los eventos que no cumplen el esquema son rechazados antes de ejecutar la inferencia.

**2. Autenticidad e integridad mediante HMAC-SHA256**

El productor genera una firma HMAC-SHA256 utilizando una clave secreta compartida, configurada mediante variables de entorno. La firma se transmite en los headers de Kafka y el consumidor la verifica antes de procesar el evento.

Los mensajes con firmas ausentes o inválidas son rechazados.

**3. Cuarentena de eventos rechazados (DLQ en MinIO)**

Los eventos rechazados por HMAC o Pydantic se almacenan en el bucket `s3://inference/dlq/` de MinIO.

Cada registro incluye el mensaje original, el motivo del rechazo, el tópico, la partición, el offset y la fecha de rechazo.

**4. Auditoría y trazabilidad**

El consumidor registra los eventos procesados correctamente en los logs de Docker, incluyendo:

- Identificador del evento (`event_id`).
- Estación de origen declarada (`station_id`).
- Timestamp del evento en UTC.
- Hash SHA-256 del mensaje original.
- Partición y offset de Kafka.
- Predicción generada.
- Ubicación del resultado almacenado en MinIO.

Los rechazos también quedan registrados con su motivo y ubicación en la DLQ.

**5. Alertas de seguridad mediante ventana deslizante**

Se implementó una alerta basada en una ventana temporal deslizante de 60 segundos.

Cuando se acumulan cinco eventos rechazados dentro de la ventana, el consumidor genera una alerta en los logs de Docker.

Se contabilizan tanto los rechazos por firmas HMAC inválidas como los errores de validación de Pydantic.

**6. Aislamiento de la comunicación Kafka**

El broker Kafka se comunica con los servicios del pipeline mediante la red interna de Docker, sin publicar su puerto de comunicación en el host.

## 12. MLOps Water Potability Federated

Se implementa una prueba de concepto de **aprendizaje federado** utilizando **Flower 1.38.0**, con el objetivo de simular el entrenamiento colaborativo de un modelo de clasificación de potabilidad del agua entre dos laboratorios independientes.

A diferencia del entrenamiento centralizado implementado con Airflow, donde los datos se procesan y utilizan en un pipeline común, el aprendizaje federado permite que cada laboratorio entrene localmente utilizando sus propias muestras. Los participantes intercambian los parámetros aprendidos con un servidor coordinador, sin transmitir los datasets de entrenamiento durante las rondas federadas.

Esta implementación es independiente del modelo productivo administrado por MLflow y utilizado por las APIs de inferencia y streaming.

### 12.1 Arquitectura federada

La solución utiliza la arquitectura de aplicaciones de Flower, ejecutada mediante contenedores Docker.

Los componentes principales son:

- **SuperLink:** administra la comunicación y coordinación de la infraestructura federada.
- **SuperNode:** representa a cada laboratorio y establece su conexión con SuperLink.
- **SuperExec:** ejecuta las aplicaciones federadas dentro de los contenedores.
- **ClientApp:** implementa el entrenamiento local de cada laboratorio.
- **ServerApp:** coordina las rondas federadas, aplica FedAvg y realiza la evaluación centralizada.

Se simulan dos participantes, denominados **Laboratorio A** y **Laboratorio B**, que utilizan el mismo código de entrenamiento, pero acceden a datasets diferentes mediante volúmenes Docker independientes.

El servidor distribuye los parámetros del modelo global a ambos laboratorios. Cada participante entrena localmente y devuelve sus parámetros actualizados junto con la cantidad de registros utilizados. Posteriormente, el servidor aplica FedAvg para obtener los nuevos parámetros globales.

### 12.2 Preparación y distribución de datos

La preparación de datos se realiza mediante el servicio Docker `federated-init`, definido en `docker-compose.yaml`.

El script `dockerfiles/federated-init/prepare_data.py` recupera el dataset original desde `s3://data/raw/water_potability.csv`, almacenado en MinIO.

El procesamiento incluye eliminación de duplicados, eliminación de registros sin variable objetivo, separación estratificada, imputación de valores faltantes mediante la mediana y estandarización de las variables predictoras.

La imputación y estandarización se ajustan exclusivamente sobre el conjunto de entrenamiento, aplicando posteriormente las mismas transformaciones a validación y test para evitar fuga de información.

La distribución utilizada es:

| Conjunto | Porcentaje | Destino |
|---|---|---|
| Entrenamiento | 70 % | Laboratorios A y B |
| Validación | 10 % | Servidor Flower |
| Test | 20 % | Servidor Flower |

El conjunto de entrenamiento se divide aleatoriamente entre los dos laboratorios mediante una partición estratificada, obteniendo distribuciones aproximadamente IID.

Los archivos generados son:

`dockerfiles/federated-client/data/client_a/train.csv`

`dockerfiles/federated-client/data/client_b/train.csv`

`dockerfiles/federated-server/data/validation.csv`

`dockerfiles/federated-server/data/test.csv`

Cada laboratorio accede exclusivamente a su partición de entrenamiento. Los archivos de validación y test quedan disponibles únicamente para el servidor de evaluación.

**Alcance:** la preparación y partición inicial de los datos se realiza centralizadamente para simular los laboratorios. El preprocesamiento federado queda fuera del alcance de esta prueba de concepto.

### 12.3 Modelo y estrategia de entrenamiento

El modelo utilizado es una **regresión logística binaria implementada con NumPy**, cuyos parámetros consisten en nueve coeficientes y un intercepto.

El entrenamiento local utiliza descenso de gradiente por lotes completos y una función sigmoide para obtener la probabilidad estimada de que una muestra de agua sea potable.

La arquitectura federada se configura de la siguiente manera:

| Parámetro | Valor |
|---|---|
| Framework | Flower 1.38.0 |
| Modelo | Regresión logística binaria |
| Estrategia de agregación | FedAvg |
| Cantidad de laboratorios | 2 |
| Rondas federadas | 20 |
| Épocas locales por ronda | 5 |
| Learning rate | 0.01 |
| Umbral de clasificación | 0.5 |
| Evaluación | Centralizada |
| Métrica principal | F1-score, clase positiva = 1 |

Cada ronda federada comprende la distribución de los parámetros globales, el entrenamiento local en ambos laboratorios y la agregación ponderada mediante FedAvg.

La ponderación considera la cantidad de muestras utilizadas por cada cliente, de manera que los parámetros de un laboratorio con más registros tengan mayor influencia sobre el modelo global.

Durante las 20 rondas se utilizan los mismos datasets locales. No es necesario que ingresen nuevos registros para ejecutar una nueva ronda federada.

### 12.4 Evaluación centralizada

La evaluación se realiza mediante `dockerfiles/federated-server/evaluate.py`.

No se utiliza evaluación distribuida en los clientes, ya que el servidor dispone de conjuntos independientes de validación y test. Para ello, la estrategia FedAvg se configura con `fraction_evaluate=0.0`.

Se implementan dos instancias de evaluación:

**Evaluación de validación:** se ejecuta sobre `validation.csv` antes del entrenamiento (ronda 0) y después de cada ronda federada. Permite observar la evolución del rendimiento del modelo global.

**Evaluación final:** se ejecuta una única vez sobre `test.csv`, al finalizar las 20 rondas, para medir el rendimiento del modelo resultante.

Las métricas calculadas son accuracy, precision, recall, F1-score y log loss.

Para precision, recall y F1 se considera la clase positiva `Potability = 1`, correspondiente a agua potable.

### 12.5 Persistencia de resultados

Al finalizar el entrenamiento, el servidor guarda los siguientes archivos:

| Archivo | Contenido |
|---|---|
| `global_model.npz` | Coeficientes e intercepto de la regresión logística federada final. |
| `validation_metrics.csv` | Evolución de las métricas de validación durante las rondas federadas. |
| `test_metrics.csv` | Métricas obtenidas en la evaluación final del modelo global. |

Estos artefactos se almacenan en:

`dockerfiles/federated-server/results/`

El archivo de métricas de validación permite analizar y visualizar el comportamiento del modelo global a lo largo del entrenamiento.

Esta prueba de concepto no registra todavía sus experimentos en MLflow ni utiliza el modelo federado como modelo productivo de inferencia.

## 13. Instalación y ejecución

### 13.1 Instalación y ejecución de Laboratory y Streaming

#### Requisitos previos

Para ejecutar el proyecto se requiere tener instalados:

- **Git** para el control de versionado.
- **Docker Desktop** con Docker Compose habilitado.
- **uv** para la gestión del entorno y las dependencias de Python.

#### Clonar el repositorio

Desde una terminal:

```powershell
git clone https://github.com/AnahiBazet/MLOPS2_TP_INTEGRADOR.git
cd MLOPS2_TP_INTEGRADOR
```

#### Configuración del entorno Python

Desde la raíz del proyecto se crea el entorno virtual y se instalan las dependencias definidas en `pyproject.toml` y `uv.lock` mediante:

```powershell
uv sync
```

En Windows, el entorno virtual puede activarse mediante:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Configuración de variables de entorno

En la raíz del proyecto debe existir un archivo `.env` con las variables de entorno utilizadas por los distintos servicios. A modo de ejemplo, se cuenta con el archivo `.env.example`. 

Allí se deben ingresar:

- `API_KEY`: clave compartida por las APIs REST, GraphQL y gRPC.
- `KAGGLE_API_TOKEN`: token utilizado por el pipeline ETL para descargar el dataset desde Kaggle.
- `KAFKA_HMAC_SECRET`: clave secreta compartida utilizada para firmar y verificar los eventos del pipeline de streaming mediante HMAC-SHA256.

El archivo `.env` también permite configurar puertos y credenciales utilizadas por los servicios de infraestructura.

Las credenciales y claves reales no deben incorporarse directamente al código fuente ni versionarse en el repositorio.

#### Inicio de la infraestructura

En una instalación inicial deben iniciarse primero los servicios necesarios para ejecutar los pipelines de datos y entrenamiento, sin levantar todavía las APIs de inferencia.

Desde la raíz del proyecto:

```powershell
docker compose --profile airflow --profile mlflow up -d --build
```

Para verificar el estado de los contenedores:

```powershell
docker compose ps
```

Los servicios principales para esta etapa quedan disponibles en:

| Servicio | Acceso |
|---|---|
| Airflow | `http://localhost:8080` |
| MLflow | `http://localhost:5001` |
| MinIO Console | `http://localhost:9001` |

#### Ejecución del pipeline de datos

Una vez iniciado Airflow, se accede a:

`http://localhost:8080`

En una instalación inicial debe ejecutarse primero el DAG:

`process_etl_water_potability`

Este pipeline ya fue explicado anteriormente en la sección `Pipeline de datos en Airflow`. El DAG debe finalizar correctamente antes de iniciar el entrenamiento.

#### Ejecución del pipeline de entrenamiento

Una vez finalizado el pipeline de datos debe ejecutarse el DAG:

`train_water_potability_model`

Este pipeline fue detallado anteriormente en la sección `Entrenamiento y selección del modelo en Airflow`.

#### Inicio de los servicios principales

Una vez finalizados correctamente ambos DAGs, pueden iniciarse los servicios de Laboratory y Streaming, junto con la infraestructura necesaria, mediante:

```powershell
docker compose --profile all up -d --build
```

Para verificar el estado de los contenedores:

```powershell
docker compose ps
```

Una vez iniciados, los principales servicios quedan disponibles en:

| Servicio | Acceso |
|---|---|
| Airflow | `http://localhost:8080` |
| MLflow | `http://localhost:5001` |
| MinIO API | `http://localhost:9000` |
| MinIO Console | `http://localhost:9001` |
| FastAPI REST | `http://localhost:8800` |
| Swagger | `http://localhost:8800/docs` |
| GraphQL / GraphiQL | `http://localhost:8801/graphql` |
| gRPC | `localhost:50051` |
| Kafka | `kafka:9092` dentro de la red interna de Docker |
| Kafka Consumer | Servicio interno, sin interfaz ni puerto expuesto al host |
| PostgreSQL | Servicio interno utilizado por Airflow y MLflow |

gRPC no expone una interfaz web, sino que recibe conexiones de clientes gRPC a través del puerto `50051`. Kafka y el consumidor de streaming se ejecutan como servicios internos de Docker Compose. El broker Kafka no publica su puerto de comunicación en el host y es accedido por los componentes del pipeline mediante la red interna de Docker.

#### Ejecuciones posteriores

Si las imágenes ya fueron construidas y no se realizaron cambios que requieran reconstruirlas, los servicios pueden iniciarse mediante:

```powershell
docker compose --profile all up -d
```

Si se modifican los Dockerfiles, las dependencias instaladas dentro de las imágenes o archivos copiados durante la construcción, deben reconstruirse mediante:

```powershell
docker compose --profile all up -d --build
```

#### Detener los servicios

Para detener y eliminar los contenedores conservando los volúmenes e imágenes:

```powershell
docker compose --profile all down
```

Los datos persistidos en los volúmenes de PostgreSQL y MinIO se mantienen disponibles para posteriores ejecuciones.

En cambio, si se desea eliminar los contenedores, imágenes y volúmenes:

```powershell
docker compose --profile all --profile test down -v --rmi all
```

### 13.2 Instalación y ejecución de Federated

El entrenamiento federado se ejecuta de manera independiente del pipeline centralizado de Airflow. Utiliza Flower 1.38.0, Docker Compose y un entorno Python específico para el comando de ejecución de Flower.

Los archivos de configuración y los módulos Python de la implementación federada ya forman parte del proyecto. No es necesario crearlos manualmente.

**Requisito previo:** el archivo `water_potability.csv` debe encontrarse disponible en `s3://data/raw/` dentro de MinIO, como resultado de la etapa de obtención de datos del pipeline ETL.

**1. Crear el entorno independiente de Flower**

Desde PowerShell, ubicarse en la raíz del proyecto:

```powershell
uv venv .venv-flower --python 3.11
```

Instalar Flower 1.38.0:

```powershell
uv pip install --python .venv-flower "flwr==1.38.0"
```

Verificar la versión instalada:

```powershell
.\.venv-flower\Scripts\flwr.exe --version
```

El resultado esperado es `Flower version: 1.38.0`.

Se utiliza un entorno separado porque la versión de Flower seleccionada requiere una versión de `protobuf` incompatible con la dependencia `grpcio-tools` del entorno principal del integrador.

No es necesario activar este entorno virtual.

**2. Validar la configuración de Docker Compose**

Desde la raíz del proyecto:

```powershell
docker compose --profile federated config --quiet
```

Si el archivo de configuración es válido, el comando finaliza sin errores.

**3. Construir la imagen de SuperExec**

```powershell
docker compose --profile federated build fl-superexec-client-a
```

Este comando construye la imagen `federated_superexec`, utilizada por los dos clientes y el servidor de ejecución de Flower.

SuperLink y SuperNode utilizan imágenes oficiales de Flower 1.38.0.

**4. Iniciar la infraestructura federada**

```powershell
docker compose --profile federated up -d
```

Este comando inicia los servicios del perfil `federated`, incluyendo MinIO, `federated-init`, SuperLink, los dos SuperNodes y los tres SuperExec.

El servicio `federated-init` ejecuta la preparación de datos y genera los conjuntos de entrenamiento, validación y test.

Para comprobar el estado de los contenedores:

```powershell
docker compose --profile federated ps -a
```

Los servicios de Flower deben encontrarse en estado `Up`. El servicio `federated-init` debe figurar como `Exited (0)`, indicando que terminó exitosamente.

Para consultar los logs:

```powershell
docker compose --profile federated logs --tail=50 fl-superlink fl-supernode-a fl-supernode-b fl-superexec-client-a fl-superexec-client-b fl-superexec-server
```

**5. Configurar la conexión de Flower con SuperLink**

El comando `flwr run` se ejecuta desde Windows y se conecta al SuperLink disponible mediante Docker en `127.0.0.1:18000`.

Primero se consultan las conexiones configuradas en Flower:

```powershell
.\.venv-flower\Scripts\flwr.exe config list
```

El archivo de configuración se encuentra en el directorio personal del usuario, bajo `.flwr/config.toml`.

Puede abrirse desde PowerShell con:

```powershell
code "$HOME\.flwr\config.toml"
```

Agregar la siguiente conexión, conservando las configuraciones existentes:

```toml
[superlink.water-potability]
address = "127.0.0.1:18000"
insecure = true
```

La propiedad `insecure = true` habilita una conexión sin TLS, exclusivamente para la ejecución local.

Verificar que Flower reconozca la conexión:

```powershell
.\.venv-flower\Scripts\flwr.exe config list
```

En la salida debe aparecer `water-potability` entre las conexiones disponibles.

Esta configuración solamente debe realizarse una vez por máquina, salvo que cambien los parámetros de conexión.

**6. Preparar el paquete de la aplicación Flower**

El proyecto incluye el script `prepare_app.ps1`, encargado de preparar el paquete Python utilizado por Flower.

Desde la raíz:

```powershell
.\dockerfiles\federated-app\prepare_app.ps1
```

El script copia los módulos de `federated-client` y `federated-server` a la carpeta generada `water_federated`, dentro de `dockerfiles/federated-app/`.

Los archivos originales permanecen en sus respectivas ubicaciones.

El archivo `dockerfiles/federated-app/pyproject.toml` define la aplicación federada y registra `ClientApp` y `ServerApp`.

El script debe ejecutarse nuevamente cuando se modifique alguno de los módulos Python que componen la aplicación.

**7. Ejecutar el entrenamiento federado**

Desde la raíz, cambiar al directorio de la aplicación Flower:

```powershell
cd dockerfiles\federated-app
```

Ejecutar:

```powershell
..\..\.venv-flower\Scripts\flwr.exe run . water-potability --stream
```

El comando empaqueta y envía la aplicación al SuperLink, inicia el entrenamiento federado y muestra sus logs en la terminal.

Se ejecutan 20 rondas de FedAvg, con dos clientes y cinco épocas locales por ronda.

Durante la ejecución se muestran las métricas de validación del modelo global por ronda. Al finalizar las 20 rondas, se realiza una única evaluación sobre el conjunto de test y se almacenan los resultados finales.

**8. Verificar los resultados**

Los resultados generados se almacenan en:

`dockerfiles/federated-server/results/`

Los archivos esperados son:

```text
global_model.npz
validation_metrics.csv
test_metrics.csv
classification_metrics.png
final_test_metrics_barplot.png
```

`validation_metrics.csv` contiene las métricas registradas durante las rondas federadas y `test_metrics.csv` contiene los resultados de la evaluación final.

`global_model.npz` almacena los parámetros del modelo global y permite recuperarlos posteriormente utilizando NumPy.

`classification_metrics.png` corresponde a la evolución de accuracy, precision, recall y F1 sobre validación por ronda, mientras que `final_test_metrics_barplot.png` representa las métricas de la evaluación final sobre test.

**9. Ejecuciones posteriores**

Una vez creados los entornos y construidas las imágenes, no es necesario repetir su instalación o construcción si no se modificaron las dependencias ni los Dockerfiles.

Para iniciar nuevamente la infraestructura:

```powershell
docker compose --profile federated up -d
```

Para volver a ejecutar el entrenamiento, desde `dockerfiles/federated-app/`:

```powershell
..\..\.venv-flower\Scripts\flwr.exe run . water-potability --stream
```

Si se modificó el código Python de Flower, debe ejecutarse previamente `prepare_app.ps1` desde la raíz del proyecto.

**Nota:** la ejecución federada no sustituye el entrenamiento centralizado del integrador ni modifica el modelo Champion registrado en MLflow.

## 14. Estructura del proyecto

La estructura principal del proyecto es la siguiente:

```text
MLOPS2_TP_INTEGRADOR/
├── .venv/
├── .venv-flower/
├── airflow/
├── dockerfiles/
├── docs/
├── notebook_example/
├── src/
│   └── mlops2_tp/
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── .python-version
├── XX.png
├── docker-compose.yaml
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

Los principales componentes del proyecto son:

- `.venv/`: entorno virtual principal de Python utilizado por el proyecto.
- `.venv-flower/`: entorno virtual independiente utilizado para ejecutar Flower 1.38.0 y evitar conflictos de dependencias con el entorno principal.
- `airflow/`: contiene los DAGs, archivos de configuración y logs asociados a Airflow.
- `dockerfiles/`: contiene la configuración, Dockerfiles y código específico de los distintos servicios desplegados mediante Docker Compose, incluyendo Laboratory, Streaming y Federated.
- `docs/`: contiene la presentación y el documento de la demo del proyecto.
- `notebook_example/`: contiene la notebook utilizada durante la experimentación inicial.
- `src/mlops2_tp/`: contiene la lógica compartida de inferencia, carga del modelo y scripts de prueba de los servicios.
- `.env`: contiene las variables de entorno utilizadas por los distintos servicios. No debe versionarse.
- `.env.example`: contiene una plantilla de las variables de entorno necesarias para ejecutar el proyecto.
- `.gitattributes` y `.gitignore`: contienen la configuración del repositorio Git y las reglas de exclusión de archivos.
- `.python-version`: define la versión de Python utilizada por el entorno principal del proyecto.
- `XX.png`: representa los diagramas de arquitectura y seguridad utilizados en la documentación del proyecto.
- `docker-compose.yaml`: define y orquesta los servicios de infraestructura, inferencia, streaming y aprendizaje federado.
- `LICENSE`: licencia del proyecto.
- `pyproject.toml` y `uv.lock`: definen las dependencias y permiten reproducir el entorno principal administrado con `uv`.
- `README.md`: contiene la documentación principal del proyecto.

## 15. Conclusiones y posibles mejoras

El desarrollo permitió comprobar que una solución de machine learning productiva requiere mucho más que el entrenamiento de un modelo. La integración de orquestación, versionado, almacenamiento, distintos mecanismos de inferencia, procesamiento en tiempo real y controles de seguridad permitió construir un flujo más cercano a un escenario real de MLOps.

La comparación entre REST, GraphQL y gRPC mostró que un mismo modelo puede ser expuesto mediante diferentes mecanismos de comunicación según las necesidades del consumidor, manteniendo una lógica de inferencia común.

La incorporación de Kafka permitió extender la solución hacia un esquema de procesamiento continuo, donde además de la predicción resultan necesarios mecanismos de validación, trazabilidad y tratamiento de eventos rechazados.

Por otra parte, la prueba de concepto con Flower permitió explorar un enfoque diferente al entrenamiento centralizado. El aprendizaje federado mostró la posibilidad de entrenar un modelo global a partir de datos distribuidos sin intercambiar los datasets locales durante las rondas de entrenamiento.

En conjunto, el proyecto permitió analizar no solo el desempeño del modelo, sino también decisiones de arquitectura, operación, seguridad y despliegue necesarias para llevar una solución de machine learning a un entorno más cercano a producción.

Hay varias posibles mejoras, pero los próximos pasos recomendados son:

- Integrar el entrenamiento federado con MLflow para registrar experimentos, métricas y artefactos.
- Evaluar la incorporación del modelo federado al flujo productivo mediante un proceso de promoción orquestado con Airflow.

