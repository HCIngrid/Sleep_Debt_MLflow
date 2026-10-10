import argparse
import tempfile
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "sleep_debt_category"
CLASS_ORDER = [
    "Optimal Recovery",
    "Mild Deficit",
    "Moderate Debt",
    "Severe Sleep Debt",
]


def positive_float(value):
    parsed_value = float(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser mayor que cero.")
    return parsed_value


def positive_integer(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("El valor debe ser mayor que cero.")
    return parsed_value


def parse_args():
    default_data_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "bedtime_screentime_sleep_debt.csv"
    )
    parser = argparse.ArgumentParser(
        description="Entrena y registra una regresión logística para clasificar la deuda de sueño."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help="Ruta del archivo CSV de entrada.",
    )
    parser.add_argument(
        "--C",
        "--c",
        dest="C",
        type=positive_float,
        default=1.0,
        help="Inverso de la fuerza de regularización. Valor predeterminado: 1.0.",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        default="lbfgs",
        help="Algoritmo de optimización. Valor predeterminado: lbfgs.",
    )
    parser.add_argument(
        "--max-iter",
        type=positive_integer,
        default=1000,
        help="Número máximo de iteraciones. Valor predeterminado: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Semilla para la partición y el modelo. Valor predeterminado: 777.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
        help="Dirección del servidor de MLflow Tracking.",
    )
    parser.add_argument(
        "--experiment-name",
        default="Logistic Regression Sleep Debt",
        help="Nombre del experimento de MLflow.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre opcional de la run de MLflow.",
    )
    return parser.parse_args()


def load_data(filepath):
    return pd.read_csv(filepath)


def clean_data(data):
    data = data.drop(["user_id", "next_day_fatigue_score"], axis=1)
    data["blue_light_filter_active"] = pd.Categorical(
        data["blue_light_filter_active"],
        categories=[0, 1],
    )
    data[TARGET] = pd.Categorical(
        data[TARGET],
        categories=CLASS_ORDER,
        ordered=True,
    )
    return data


def split_data(data, target, val_size=0.15, test_size=0.15, random_state=777):
    X = data.drop(columns=[target])
    y = data[target].astype(str)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_preprocessor(data, target):
    features = data.drop(columns=[target])
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(
        include=["str", "category"]
    ).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical_features,
            ),
        ]
    )


def train_model(
    X_train,
    y_train,
    preprocessor,
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=777,
):
    model = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def calculate_metrics(y_real, y_pred):
    return {
        "accuracy": accuracy_score(y_real, y_pred),
        "recall_macro": recall_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_real,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "gmean": geometric_mean_score(
            y_real,
            y_pred,
            average="multiclass",
        ),
    }


def save_confusion_matrix(y_real, y_pred, split_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_pred,
        labels=CLASS_ORDER,
        cmap="Blues",
        xticks_rotation=30,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusión - {split_name}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def evaluate_model(model, X_data, y_data, split_name, artifact_directory):
    y_pred = model.predict(X_data)
    metrics = calculate_metrics(y_data, y_pred)
    mlflow.log_metrics(
        {
            f"{split_name}_{metric_name}": metric_value
            for metric_name, metric_value in metrics.items()
        }
    )
    predictions_path = artifact_directory / f"predictions_{split_name}.csv"
    pd.DataFrame(
        {
            "Valor_Real": y_data.to_numpy(),
            "Prediccion": y_pred,
        }
    ).to_csv(predictions_path, index=False)
    mlflow.log_artifact(str(predictions_path), artifact_path="predictions")
    confusion_matrix_path = artifact_directory / f"confusion_matrix_{split_name}.png"
    save_confusion_matrix(
        y_data,
        y_pred,
        split_name,
        confusion_matrix_path,
    )
    mlflow.log_artifact(
        str(confusion_matrix_path),
        artifact_path="confusion_matrix",
    )
    return metrics


def configure_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow.set_experiment(experiment_name)


def run_experiment(
    data,
    target,
    data_path,
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=777,
    val_size=0.15,
    test_size=0.15,
    run_name=None,
):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        data,
        target,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = create_preprocessor(data, target)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "C": C,
                "solver": solver,
                "max_iter": max_iter,
                "random_state": random_state,
                "validation_size": val_size,
                "test_size": test_size,
                "data_path": str(data_path),
            }
        )
        validation_model = train_model(
            X_train,
            y_train,
            preprocessor,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_directory = Path(temporary_directory)
            train_metrics = evaluate_model(
                validation_model,
                X_train,
                y_train,
                "train",
                artifact_directory,
            )
            validation_metrics = evaluate_model(
                validation_model,
                X_val,
                y_val,
                "validation",
                artifact_directory,
            )
            X_train_final = pd.concat([X_train, X_val])
            y_train_final = pd.concat([y_train, y_val])
            final_model = train_model(
                X_train_final,
                y_train_final,
                preprocessor,
                C=C,
                solver=solver,
                max_iter=max_iter,
                random_state=random_state,
            )
            test_metrics = evaluate_model(
                final_model,
                X_test,
                y_test,
                "test",
                artifact_directory,
            )
        signature = infer_signature(
            X_train_final,
            final_model.predict(X_train_final),
        )
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            signature=signature,
            input_example=X_train_final.head(5),
        )
    return {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }


def main():
    args = parse_args()
    data_path = args.data_path.resolve()
    data = clean_data(load_data(data_path))
    configure_mlflow(args.tracking_uri, args.experiment_name)
    run_experiment(
        data,
        TARGET,
        data_path,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        random_state=args.seed,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
