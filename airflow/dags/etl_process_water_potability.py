import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Water Potability Data
"""


default_args = {
    'owner': "Anahi Bazet",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_water_potability",
    description="ETL process for water potability.",
    doc_md=markdown_text,
    tags=["ETL", "Water Potability"],
    default_args=default_args,
    catchup=False,
)
def process_etl_water_potability():
    
    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=[
                    "kagglehub",
                    "awswrangler==3.17.1",
                    "pandas"],
    system_site_packages=True
    )

    def get_data():
        """
        Download the Water Potability dataset from Kaggle
        and save the raw CSV in MinIO.
        """
        import kagglehub
        import pandas as pd
        import awswrangler as wr

        # Download version 3 of the dataset from Kaggle.
        file_path = kagglehub.dataset_download(
            "adityakadiwal/water-potability/versions/3",
            path="water_potability.csv"
        )

        # Load the downloaded CSV.
        dataframe = pd.read_csv(file_path)

        # Save the raw dataset in MinIO
        data_path = "s3://data/raw/water_potability.csv"

        wr.s3.to_csv(
            df=dataframe,
            path=data_path,
            index=False
        )

    @task.virtualenv(
        task_id="split_dataset",
        requirements=[
            "awswrangler==3.17.1",
            "scikit-learn==1.9.1"
        ],
        system_site_packages=True
    )

    def split_dataset():
        """
        Split the Water Potability dataset into
        training and testing sets.
        """

        import awswrangler as wr

        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(
                df=df,
                path=path,
                index=False
            )

        # Dataset original
        data_original_path = "s3://data/raw/water_potability.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # ---------------------------------------------------------
        # Limpieza previa al split
        # ---------------------------------------------------------

        # Eliminar duplicados
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        # La variable target no debe ser imputada.
        # Si hubiera algún target nulo, eliminamos esa observación.
        target_col = Variable.get("target_col_water")
        dataset.dropna(subset=[target_col], inplace=True)

        # ---------------------------------------------------------
        # Separación X / y
        # ---------------------------------------------------------

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        test_size = float(Variable.get("test_size_water"))

        random_seed = int(Variable.get("random_seed_water"))

        # ---------------------------------------------------------
        # Split train / test
        # ---------------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_seed
        )

        # ---------------------------------------------------------
        # Guardado
        # ---------------------------------------------------------

        save_to_csv(
            X_train,
            "s3://data/interim/train/water_X_train.csv"
        )

        save_to_csv(
            X_test,
            "s3://data/interim/test/water_X_test.csv"
        )

        save_to_csv(
            y_train,
            "s3://data/final/train/water_y_train.csv"
        )

        save_to_csv(
            y_test,
            "s3://data/final/test/water_y_test.csv"
        )

    @task.virtualenv(
        task_id="impute_missing_values",
        requirements=[
            "awswrangler==3.17.1",
            "scikit-learn==1.9.1",
            "boto3"
        ],
        system_site_packages=True
    )
    def impute_missing_values():
        """
        Impute missing numerical values using the median calculated
        exclusively from the training dataset.
        """

        import json
        import datetime
        import boto3
        import awswrangler as wr
        import pandas as pd

        from sklearn.impute import SimpleImputer
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(
                df=df,
                path=path,
                index=False
            )

        # ---------------------------------------------------------
        # Cargar train y test
        # ---------------------------------------------------------

        X_train = wr.s3.read_csv(
            "s3://data/interim/train/water_X_train.csv"
        )

        X_test = wr.s3.read_csv(
            "s3://data/interim/test/water_X_test.csv"
        )

        # ---------------------------------------------------------
        # Imputación
        # ---------------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        # Las medianas se calculan SOLO con train
        X_train_imputed = imputer.fit_transform(X_train)

        # Test usa las medianas calculadas con train
        X_test_imputed = imputer.transform(X_test)

        X_train_imputed = pd.DataFrame(
            X_train_imputed,
            columns=X_train.columns
        )

        X_test_imputed = pd.DataFrame(
            X_test_imputed,
            columns=X_test.columns
        )

        # ---------------------------------------------------------
        # Guardar datasets finales
        # ---------------------------------------------------------

        save_to_csv(
            X_train_imputed,
            "s3://data/final/train/water_X_train.csv"
        )

        save_to_csv(
            X_test_imputed,
            "s3://data/final/test/water_X_test.csv"
        )

        # ---------------------------------------------------------
        # Guardar información del preprocesamiento
        # ---------------------------------------------------------

        target_col = Variable.get("target_col_water")
        test_size = float(Variable.get("test_size_water"))

        median_values = {
            column: float(median)
            for column, median in zip(
                X_train.columns,
                imputer.statistics_
            )
        }

        random_seed = int(Variable.get("random_seed_water"))

        data_dict = {
            "dataset": "Water Potability",
            "target_col": target_col,
            "columns": X_train.columns.to_list(),
            "imputation_strategy": "median",
            "median_values": median_values,
            "test_size": test_size,
            "random_state": random_seed,
            "train_observations": X_train.shape[0],
            "test_observations": X_test.shape[0],
            "date": datetime.datetime.today().strftime(
                "%Y/%m/%d-%H:%M:%S"
            )
        }

        client = boto3.client("s3")

        client.put_object(
            Bucket="data",
            Key="data_info/data.json",
            Body=json.dumps(data_dict, indent=2)
        )

    get_data() >> split_dataset() >> impute_missing_values()

dag = process_etl_water_potability()