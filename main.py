from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
DEFAULT_TARGET = "is_tp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline de Machine Learning para el dataset eco-acoustic."
    )
    parser.add_argument("--train", default="eco_acoustic_train.csv", help="CSV de entrenamiento.")
    parser.add_argument("--test", default="eco_acoustic_test.csv", help="CSV de prueba.")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["is_tp", "species_id", "songtype_id"],
        help="Variable objetivo a predecir.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Carpeta para resultados.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Activa una busqueda pequena de hiperparametros para el mejor modelo base.",
    )
    return parser.parse_args()


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    mel_columns = [column for column in df.columns if column.startswith("mel_")]
    optional_categorical = [
        column
        for column in ["species_id", "songtype_id", "is_tp"]
        if column in df.columns and column != target
    ]
    return optional_categorical + mel_columns


def print_dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> None:
    print("\n=== Resumen del dataset ===")
    print(f"Train: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
    print(f"Test : {test_df.shape[0]} filas, {test_df.shape[1]} columnas")
    print(f"Target: {target}")
    print("\nDistribucion del target en train:")
    print(train_df[target].value_counts().sort_index())
    missing = train_df.isna().sum().sum() + test_df.isna().sum().sum()
    print(f"\nValores faltantes totales train+test: {missing}")


def build_models() -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, str, Pipeline]:
    rows = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nEntrenando {name}...")
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_valid, predictions),
                "f1_macro": f1_score(y_valid, predictions, average="macro"),
                "f1_weighted": f1_score(y_valid, predictions, average="weighted"),
            }
        )
        fitted_models[name] = model

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["f1_macro", "accuracy"], ascending=False
    )
    best_name = str(metrics_df.iloc[0]["model"])
    return metrics_df, best_name, fitted_models[best_name]


def tune_model(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
) -> Pipeline:
    if "svm" in str(model.named_steps["model"]).lower():
        params = {
            "model__C": [0.5, 1.0, 2.0, 5.0],
            "model__gamma": ["scale", 0.01, 0.1],
        }
    elif isinstance(model.named_steps["model"], RandomForestClassifier):
        params = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        params = {}

    if not params:
        print("\nEl modelo ganador no tiene grilla configurada; se mantiene la version base.")
        return model

    min_class_count = int(y_train.value_counts().min())
    cv_splits = max(2, min(5, min_class_count))
    scoring = "f1_macro" if target != "is_tp" else "f1"

    print(f"\nAjustando hiperparametros con {cv_splits}-fold CV...")
    search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    print(f"Mejores parametros: {search.best_params_}")
    print(f"Mejor score CV: {search.best_score_:.4f}")
    return search.best_estimator_


def save_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    best_model: Pipeline,
    feature_columns: list[str],
    test_df: pd.DataFrame,
    target: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.csv"
    model_path = output_dir / "best_model.joblib"
    predictions_path = output_dir / "test_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    joblib.dump(best_model, model_path)

    prediction_df = pd.DataFrame(
        {
            "recording_id": test_df["recording_id"],
            f"predicted_{target}": best_model.predict(test_df[feature_columns]),
        }
    )
    prediction_df.to_csv(predictions_path, index=False)

    print("\n=== Archivos generados ===")
    print(metrics_path)
    print(model_path)
    print(predictions_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    train_df, test_df = load_data(args.train, args.test)
    if args.target not in train_df.columns:
        raise ValueError(f"La columna target '{args.target}' no existe en train.")

    feature_columns = get_feature_columns(train_df, args.target)
    if not feature_columns:
        raise ValueError("No se encontraron columnas de caracteristicas.")

    print_dataset_summary(train_df, test_df, args.target)
    print(f"\nCaracteristicas usadas ({len(feature_columns)}): {feature_columns[:8]}...")

    x = train_df[feature_columns]
    y = train_df[args.target]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    metrics_df, best_name, best_model = evaluate_models(
        build_models(), x_train, x_valid, y_train, y_valid
    )

    print("\n=== Comparacion de modelos ===")
    print(metrics_df.to_string(index=False))
    print(f"\nMejor modelo base: {best_name}")

    if args.tune:
        best_model = tune_model(best_model, x_train, y_train, args.target)
        tuned_predictions = best_model.predict(x_valid)
        print("\n=== Evaluacion del modelo ajustado en validacion ===")
        print(classification_report(y_valid, tuned_predictions, zero_division=0))
        print("Matriz de confusion:")
        print(confusion_matrix(y_valid, tuned_predictions))

    print("\nReentrenando el mejor modelo con todo el train...")
    best_model.fit(x, y)

    save_outputs(output_dir, metrics_df, best_model, feature_columns, test_df, args.target)


if __name__ == "__main__":
    main()
