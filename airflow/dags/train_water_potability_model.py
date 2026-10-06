import datetime

from airflow.decorators import dag, task  # pyright: ignore[reportMissingImports]

markdown_text = """
### Initial Training - Water Potability

This DAG performs the initial training of the Water Potability model.

It:
- Searches hyperparameters using Optuna.
- Compares Decision Tree and Random Forest.
- Uses 5-fold cross-validation.
- Optimizes F1-score.
- Trains the best configuration using the complete training dataset.
- Evaluates the final model on the test dataset.
- Registers the model in MLflow.
- Assigns the Champion alias.
"""


default_args = {
    "owner": "Anahi Bazet",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
    "dagrun_timeout": datetime.timedelta(minutes=30),
}


@dag(
    dag_id="train_water_potability_model",
    description="Initial training of the Water Potability classification model.",
    doc_md=markdown_text,
    tags=["Training", "Water Potability", "MLflow", "Optuna"],
    default_args=default_args,
    schedule=None,
    catchup=False,
    params={
        "n_trials": 80
    },
    )
def train_water_potability_model():

    # =========================================================
    # 1. BÚSQUEDA DE HIPERPARÁMETROS
    # =========================================================

    @task.virtualenv(
        task_id="search_hyperparameters",
        requirements=[
            "awswrangler==3.17.1",
            "scikit-learn==1.9.1",
            "mlflow==3.16.1",
            "optuna",
            "boto3",
            "imbalanced-learn",
        ],
        system_site_packages=True,
    )
    def search_hyperparameters(n_trials):
        """
        Search the best model and hyperparameters using Optuna.

        Only the training dataset is used.
        Each trial is evaluated using 5-fold cross-validation
        and mean F1-score.
        """

        import datetime

        import awswrangler as wr
        import mlflow
        import optuna

        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, make_scorer
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.tree import DecisionTreeClassifier

        mlflow.set_tracking_uri("http://mlflow:5000")

        from airflow.models import Variable
        
        random_seed = int(
            Variable.get("random_seed_water")
        )

        # -----------------------------------------------------
        # Cargar TRAIN
        # -----------------------------------------------------

        X_train = wr.s3.read_csv(
            "s3://data/final/train/water_X_train.csv"
        )

        y_train = wr.s3.read_csv(
            "s3://data/final/train/water_y_train.csv"
        )

        y_train = y_train.to_numpy().ravel()

        # -----------------------------------------------------
        # Experimento MLflow
        # -----------------------------------------------------

        experiment = mlflow.set_experiment(
            "Water Potability"
        )

        # -----------------------------------------------------
        # Cross Validation reproducible
        # -----------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_seed
        )

        f1_scorer = make_scorer(
            f1_score,
            zero_division=0
        )

        # -----------------------------------------------------
        # Función objetivo de Optuna
        # -----------------------------------------------------

        def objective(trial):

            classifier_name = trial.suggest_categorical(
                "classifier",
                [
                    "DecisionTreeClassifier",
                    "RandomForestClassifier",
                ],
            )

            if classifier_name == "DecisionTreeClassifier":

                max_depth = trial.suggest_categorical(
                    "tree_max_depth",
                    [2, 3, 6]
                )

                max_features = trial.suggest_categorical(
                    "tree_max_features",
                    [6, 9]
                )

                min_samples_leaf = trial.suggest_categorical(
                    "tree_min_samples_leaf",
                    [3, 5]
                )

                criterion = trial.suggest_categorical(
                    "tree_criterion",
                    ["gini", "entropy"]
                )

                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=random_seed
                )

                params = {
                    "model": "DecisionTreeClassifier",
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "criterion": criterion,
                    "random_state": random_seed,
                }

            else:

                n_estimators = trial.suggest_categorical(
                    "rf_n_estimators",
                    [100, 500]
                )

                max_depth = trial.suggest_categorical(
                    "rf_max_depth",
                    [6, None]
                )

                max_features = trial.suggest_categorical(
                    "rf_max_features",
                    [6, 9]
                )

                min_samples_leaf = trial.suggest_categorical(
                    "rf_min_samples_leaf",
                    [3, 9]
                )

                criterion = trial.suggest_categorical(
                    "rf_criterion",
                    ["gini", "entropy"]
                )

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    random_state=random_seed
                )

                params = {
                    "model": "RandomForestClassifier",
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "max_features": max_features,
                    "min_samples_leaf": min_samples_leaf,
                    "criterion": criterion,
                    "random_state": random_seed,
                }

            pipeline = Pipeline([
                (
                    "oversampler",
                    SMOTE(
                        random_state=random_seed
                    )
                ),
                (
                    "classifier",
                    model
                ),
            ])
              
            # Cada trial queda registrado como child run
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=f"Trial_{trial.number}",
                nested=True,
            ):

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=f1_scorer,
                    n_jobs=-1,
                )

                mean_f1 = scores.mean()

                mlflow.log_params(params)

                mlflow.log_metric(
                    "cv_f1",
                    mean_f1
                )

            return mean_f1

        # -----------------------------------------------------
        # Run padre de la búsqueda
        # -----------------------------------------------------

        run_name = (
            "Optuna_search_"
            + datetime.datetime.today().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        ) as parent_run:

            study = optuna.create_study(
                direction="maximize"
            )

            study.optimize(
                objective,
                n_trials=int(n_trials)
            )

            mlflow.log_params(
                study.best_params
            )

            mlflow.log_metric(
                "best_cv_f1",
                study.best_value
            )

            mlflow.set_tags(
                {
                    "project": "Water Potability",
                    "optimizer_engine": "optuna",
                    "optimization_metric": "f1",
                    "model_family": "sklearn",
                }
            )

            search_run_id = parent_run.info.run_id

        return {
            "best_params": study.best_params,
            "best_cv_f1": float(study.best_value),
            "best_trial_number": study.best_trial.number,
            "search_run_id": search_run_id,
            "experiment_id": experiment.experiment_id,
        }

    # =========================================================
    # 2. SELECCIÓN DEL MEJOR MODELO
    # =========================================================

    @task(task_id="select_best_model")
    def select_best_model(search_results):
        """
        Select the best model/configuration found by Optuna.
        """

        print(
            "Best model:",
            search_results["best_params"]["classifier"]
        )

        print(
            "Best CV F1:",
            search_results["best_cv_f1"]
        )

        print(
            "Best parameters:",
            search_results["best_params"]
        )

        return search_results

    # =========================================================
    # 3. ENTRENAMIENTO FINAL
    # =========================================================

    @task.virtualenv(
        task_id="train_best_model",
        requirements=[
            "awswrangler==3.17.1",
            "scikit-learn==1.9.1",
            "mlflow==3.16.1",
            "boto3",
            "imbalanced-learn",
        ],
        system_site_packages=True,
    )
    def train_best_model(selected_model):
        """
        Train the best configuration using the complete training dataset.
        """

        import datetime

        import awswrangler as wr
        import mlflow

        from mlflow.models import infer_signature
        from imblearn.over_sampling import SMOTE

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        from airflow.models import Variable

        random_seed = int(
            Variable.get("random_seed_water")
        )

        mlflow.set_tracking_uri(
            "http://mlflow:5000"
        )

        X_train = wr.s3.read_csv(
            "s3://data/final/train/water_X_train.csv"
        )

        y_train = wr.s3.read_csv(
            "s3://data/final/train/water_y_train.csv"
        )

        y_train = y_train.to_numpy().ravel()

        best_params = selected_model["best_params"]

        # -----------------------------------------------------
        # Crear modelo ganador
        # -----------------------------------------------------

        if (
            best_params["classifier"]
            == "DecisionTreeClassifier"
        ):

           model = DecisionTreeClassifier(
                max_depth=best_params[
                    "tree_max_depth"
                ],
                max_features=best_params[
                    "tree_max_features"
                ],
                min_samples_leaf=best_params[
                    "tree_min_samples_leaf"
                ],
                criterion=best_params[
                    "tree_criterion"
                ],
                random_state=random_seed,
        )

        else:

            model = RandomForestClassifier(
                max_depth=best_params["rf_max_depth"],
                n_estimators=best_params["rf_n_estimators"],
                max_features=best_params["rf_max_features"],
                min_samples_leaf=best_params["rf_min_samples_leaf"],
                criterion=best_params["rf_criterion"],
                random_state=random_seed,
            )

        # -----------------------------------------------------
        # Entrenar nuevamente con TODO train
        # -----------------------------------------------------

        oversampler = SMOTE(
            random_state=random_seed
        )

        X_train_resampled, y_train_resampled = (
            oversampler.fit_resample(
                X_train,
                y_train
            )
        )
        
        model.fit(
            X_train_resampled,
            y_train_resampled
        )

        # -----------------------------------------------------
        # Métricas sobre TRAIN completo
        # -----------------------------------------------------

        positive_class_index = list(
            model.classes_
        ).index(1)

        train_probabilities = model.predict_proba(
            X_train
        )[:, positive_class_index]

        y_train_pred = (
            train_probabilities >= 0.5
        ).astype(int)

        train_precision = precision_score(
            y_train,
            y_train_pred,
            zero_division=0
        )

        train_recall = recall_score(
            y_train,
            y_train_pred,
            zero_division=0
        )

        train_f1 = f1_score(
            y_train,
            y_train_pred,
            zero_division=0
        )

        train_accuracy = accuracy_score(
            y_train,
            y_train_pred
        )

        experiment = mlflow.set_experiment(
            "Water Potability"
        )

        run_name = (
            "Final_model_"
            + datetime.datetime.today().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        ) as run:

            mlflow.log_params(
                model.get_params()
            )

            mlflow.log_param(
                "model",
                type(model).__name__
            )

            mlflow.log_param(
                "classification_threshold",
                0.5
            )

            # Trazabilidad con la búsqueda de Optuna
            mlflow.set_tags({
                "optuna_best_trial": selected_model["best_trial_number"],
                "optuna_search_run_id": selected_model["search_run_id"],
            })

            mlflow.log_metric(
                "best_cv_f1",
                selected_model[
                    "best_cv_f1"
                ]
            )

            mlflow.log_metrics({
                "train_precision": float(train_precision),
                "train_recall": float(train_recall),
                "train_f1": float(train_f1),
                "train_accuracy": float(train_accuracy),
            })

            signature = infer_signature(
                X_train,
                model.predict(X_train)
            )

            model_info = mlflow.sklearn.log_model(
                name="model",
                sk_model=model,
                signature=signature,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            run_id = run.info.run_id

        return {
            "model_uri": model_info.model_uri,
            "run_id": run_id,
            "best_params": best_params,
            "best_cv_f1": selected_model["best_cv_f1"],
            "train_precision": float(train_precision),
            "train_recall": float(train_recall),
            "train_f1": float(train_f1),
            "train_accuracy": float(train_accuracy),
        }

    # =========================================================
    # 4. EVALUACIÓN EN TEST
    # =========================================================

    @task.virtualenv(
        task_id="evaluate_model",
        requirements=[
            "awswrangler==3.17.1",
            "scikit-learn==1.9.1",
            "mlflow==3.16.1",
            "boto3",
        ],
        system_site_packages=True,
    )
    def evaluate_model(training_result):
        """
        Evaluate the final model using the test dataset.

        F1 is the main metric.
        The classification threshold remains fixed at 0.5.
        """

        import awswrangler as wr
        import mlflow

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        mlflow.set_tracking_uri(
            "http://mlflow:5000"
        )

        X_test = wr.s3.read_csv(
            "s3://data/final/test/water_X_test.csv"
        )

        y_test = wr.s3.read_csv(
            "s3://data/final/test/water_y_test.csv"
        )

        y_test = y_test.to_numpy().ravel()

        model = mlflow.sklearn.load_model(
            training_result["model_uri"]
        )

        # -----------------------------------------------------
        # Threshold fijo = 0.5
        # -----------------------------------------------------

        positive_class_index = list(
            model.classes_
        ).index(1)

        probabilities = model.predict_proba(
            X_test
        )[:, positive_class_index]

        y_pred = (
            probabilities >= 0.5
        ).astype(int)

        # -----------------------------------------------------
        # Métricas finales
        # -----------------------------------------------------

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        metrics = {
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_accuracy": float(accuracy),
        }

        # Guardamos las métricas en el mismo run
        # donde se registró el modelo final.
        with mlflow.start_run(
            run_id=training_result["run_id"]
        ):
            mlflow.log_metrics(metrics)

        print(
            f"Precision: {precision:.4f}"
        )
        print(
            f"Recall:    {recall:.4f}"
        )
        print(
            f"F1:        {f1:.4f}"
        )
        print(
            f"Accuracy:  {accuracy:.4f}"
        )

        return {
            **training_result,
            **metrics,
        }

    # =========================================================
    # 5. REGISTRO DEL MODELO EN MLFLOW
    # =========================================================

    @task.virtualenv(
        task_id="register_model",
        requirements=[
            "mlflow==3.16.1",
            "boto3",
        ],
        system_site_packages=True,
    )
    def register_model(evaluation_result):
        """
        Register the final model in the MLflow Model Registry.
        """

        import mlflow

        mlflow.set_tracking_uri(
            "http://mlflow:5000"
        )

        client = mlflow.MlflowClient()

        model_name = (
            "water_potability_model_prod"
        )

        # Crear Registered Model si todavía no existe
        try:
            client.get_registered_model(
                model_name
            )
        except Exception:
            client.create_registered_model(
                name=model_name,
                description=(
                    "Classifier for Water Potability"
                ),
            )

        tags = {
            key: str(value)
            for key, value
            in evaluation_result[
                "best_params"
            ].items()
        }

        tags["balancing_strategy"] = "SMOTE"

        tags["best_cv_f1"] = str(
            evaluation_result["best_cv_f1"]
        )

        tags["train_precision"] = str(
            evaluation_result["train_precision"]
        )

        tags["train_recall"] = str(
            evaluation_result["train_recall"]
        )

        tags["train_f1"] = str(
            evaluation_result["train_f1"]
        )

        tags["train_accuracy"] = str(
            evaluation_result["train_accuracy"]
        )

        tags["test_precision"] = str(
            evaluation_result[
                "test_precision"
            ]
        )

        tags["test_recall"] = str(
            evaluation_result[
                "test_recall"
            ]
        )

        tags["test_f1"] = str(
            evaluation_result[
                "test_f1"
            ]
        )

        tags["test_accuracy"] = str(
            evaluation_result[
                "test_accuracy"
            ]
        )

        result = client.create_model_version(
            name=model_name,
            source=evaluation_result[
                "model_uri"
            ],
            run_id=evaluation_result[
                "run_id"
            ],
            tags=tags,
        )

        return {
            "model_name": model_name,
            "version": result.version,
        }

    # =========================================================
    # 6. SET CHAMPION
    # =========================================================

    @task.virtualenv(
        task_id="set_champion",
        requirements=[
            "mlflow==3.16.1",
        ],
        system_site_packages=True,
    )
    def set_champion(
        registered_model,
        evaluation_result
    ):
        """
        Compare the new model with the current Champion.

        If no Champion exists yet, assign the new model as Champion.
        Otherwise, promote the new model only if its test F1 is higher.
        """

        import mlflow

        mlflow.set_tracking_uri(
            "http://mlflow:5000"
        )

        client = mlflow.MlflowClient()

        model_name = registered_model["model_name"]
        new_version = registered_model["version"]

        new_test_f1 = float(
            evaluation_result["test_f1"]
        )

        try:
            # -----------------------------------------------------
            # Obtener Champion actual
            # -----------------------------------------------------

            current_champion = client.get_model_version_by_alias(
                model_name,
                "champion"
            )

            current_champion_version = current_champion.version

            current_champion_test_f1 = float(
                current_champion.tags["test_f1"]
            )

            print(
                "Current Champion version:",
                current_champion_version
            )

            print(
                "Current Champion test F1:",
                current_champion_test_f1
            )

            print(
                "New model version:",
                new_version
            )

            print(
                "New model test F1:",
                new_test_f1
            )

            # -----------------------------------------------------
            # Comparar modelo nuevo vs Champion
            # -----------------------------------------------------

            if new_test_f1 > current_champion_test_f1:

                client.set_registered_model_alias(
                    model_name,
                    "champion",
                    new_version,
                )

                print(
                    "New Champion:",
                    model_name,
                    "version",
                    new_version,
                )

            else:

                print(
                    "Champion remains:",
                    model_name,
                    "version",
                    current_champion_version,
                )

        except Exception:

            # -----------------------------------------------------
            # Primera corrida: todavía no existe Champion
            # -----------------------------------------------------

            client.set_registered_model_alias(
                model_name,
                "champion",
                new_version,
            )

            print(
                "No previous Champion found."
            )

            print(
                "Initial Champion:",
                model_name,
                "version",
                new_version,
            )

    # =========================================================
    # PIPELINE
    # =========================================================

    search_results = search_hyperparameters(
        "{{ params.n_trials }}"
    )

    selected_model = select_best_model(
        search_results
    )

    training_result = train_best_model(
        selected_model
    )

    evaluation_result = evaluate_model(
        training_result
    )

    registered_model = register_model(
        evaluation_result
    )

    set_champion(
        registered_model,
        evaluation_result
    )


dag = train_water_potability_model()