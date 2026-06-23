from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# Configuracion general del proyecto
# ============================================================
# El objetivo principal del enunciado es clasificar especies (`species_id`)
# usando el espacio vectorial de 64 caracteristicas Mel: mel_0 ... mel_63.
RANDOM_STATE = 42
DEFAULT_TARGET = "species_id"

SPECIES_NAMES = {
    10: "Leptodactylus discodactylus",
    12: "Osteocephalus taurinus",
    17: "Chiroxiphia lineata",
    18: "Saltator grossus",
    23: "Pheucticus chrysopeplus",
}

CONFIDENCE_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.40


@dataclass
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    models: Path


@dataclass
class ClassifierCandidate:
    family: str
    name: str
    model: Any
    metrics: dict[str, float | str]
    predictions: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def parse_args() -> argparse.Namespace:
    # Parametros que se pueden cambiar desde consola sin editar el codigo.
    parser = argparse.ArgumentParser(
        description="Pipeline de Machine Learning para clasificacion eco-acustica."
    )
    parser.add_argument("--train", default="eco_acoustic_train.csv", help="CSV de entrenamiento.")
    parser.add_argument("--test", default="eco_acoustic_test.csv", help="CSV de prueba.")
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["species_id", "songtype_id", "is_tp"],
        help="Variable objetivo. Para el proyecto se recomienda species_id.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Carpeta de resultados.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporcion de validacion.")
    parser.add_argument("--epochs", type=int, default=80, help="Epocas maximas para los MLP.")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamano de batch para los MLP.")
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Incluye songtype_id/is_tp como variables. Por defecto se usa solo X=mel_0..mel_63.",
    )
    parser.add_argument("--skip-geometry", action="store_true", help="Omite PCA y t-SNE.")
    parser.add_argument("--skip-clustering", action="store_true", help="Omite DBSCAN y GMM.")
    parser.add_argument("--skip-mlp", action="store_true", help="Omite redes MLP implementadas con NumPy.")
    return parser.parse_args()


def configure_plots() -> None:
    # Tamano de fuente >= 14 para evitar penalizaciones en las figuras del informe.
    plt.rcParams.update(
        {
            "figure.figsize": (10, 7),
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )


def prepare_output_paths(output_dir: str) -> OutputPaths:
    # Crea la estructura outputs/: tablas, figuras y modelos.
    root = Path(output_dir)
    paths = OutputPaths(
        root=root,
        figures=root / "figures",
        tables=root / "tables",
        models=root / "models",
    )
    for path in [paths.root, paths.figures, paths.tables, paths.models]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Paso 1: lectura de CSV de entrenamiento y prueba.
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def get_mel_columns(df: pd.DataFrame) -> list[str]:
    # Selecciona solo las variables numericas del espacio X in R64.
    mel_columns = [column for column in df.columns if column.startswith("mel_")]
    return sorted(mel_columns, key=lambda column: int(column.split("_")[1]))


def get_feature_columns(df: pd.DataFrame, target: str, include_metadata: bool) -> list[str]:
    # Por defecto se usan solo mel_0..mel_63, como pide el enunciado.
    feature_columns = get_mel_columns(df)

    if include_metadata:
        metadata_columns = [
            column
            for column in ["songtype_id", "is_tp", "species_id"]
            if column in df.columns and column != target
        ]
        feature_columns = metadata_columns + feature_columns

    return feature_columns


def validate_inputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
) -> None:
    # Validaciones basicas antes de entrenar modelos.
    if target not in train_df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en train.")

    missing_test_columns = [column for column in feature_columns if column not in test_df.columns]
    if missing_test_columns:
        raise ValueError(f"Faltan columnas en test: {missing_test_columns}")

    if len(get_mel_columns(train_df)) != 64:
        raise ValueError("El proyecto espera exactamente 64 columnas mel_0..mel_63.")


def species_label(value: Any) -> str:
    if value in SPECIES_NAMES:
        return f"{value} - {SPECIES_NAMES[value]}"
    return str(value)


def save_dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    paths: OutputPaths,
) -> None:
    # Paso 2: resumen inicial del dataset y distribucion de clases.
    summary = pd.DataFrame(
        [
            {"partition": "train", "rows": len(train_df), "columns": train_df.shape[1]},
            {"partition": "test", "rows": len(test_df), "columns": test_df.shape[1]},
        ]
    )
    summary["target"] = target
    summary["features_used"] = len(feature_columns)
    summary["missing_values"] = [
        int(train_df.isna().sum().sum()),
        int(test_df.isna().sum().sum()),
    ]
    summary.to_csv(paths.tables / "dataset_summary.csv", index=False)

    distribution = (
        train_df[target]
        .value_counts()
        .sort_index()
        .rename_axis(target)
        .reset_index(name="count")
    )
    distribution["percentage"] = distribution["count"] / distribution["count"].sum()
    distribution["label"] = distribution[target].map(species_label)
    distribution.to_csv(paths.tables / "target_distribution.csv", index=False)

    plt.figure()
    plt.bar(distribution["label"], distribution["count"], color="#4c78a8")
    plt.title(f"Distribucion de clases: {target}")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(paths.figures / "target_distribution.png", dpi=160)
    plt.close()

    print("\n=== Resumen del dataset ===")
    print(summary.to_string(index=False))
    print("\nDistribucion del target:")
    print(distribution[[target, "count", "percentage"]].to_string(index=False))


def plot_projection(
    embedding: np.ndarray,
    labels: pd.Series | np.ndarray,
    title: str,
    output_path: Path,
    target: str,
) -> None:
    # Grafico 2D usado para PCA y t-SNE.
    labels_array = np.asarray(labels)
    unique_labels = sorted(pd.unique(labels_array))
    cmap = plt.get_cmap("tab10")

    plt.figure()
    for index, label in enumerate(unique_labels):
        mask = labels_array == label
        legend_label = species_label(label) if target == "species_id" else str(label)
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=28,
            alpha=0.75,
            color=cmap(index % 10),
            label=legend_label,
        )
    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_dimensionality_analysis(
    x_scaled: np.ndarray,
    y: pd.Series,
    target: str,
    paths: OutputPaths,
) -> pd.DataFrame:
    # Paso 3: exploracion geometrica y reduccion de dimensionalidad.
    rows: list[dict[str, Any]] = []

    print("\n=== Reduccion dimensional ===")

    # Algoritmo PCA: metodo lineal. Sirve para medir varianza global retenida.
    start = time.perf_counter()
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_embedding = pca.fit_transform(x_scaled)
    elapsed = time.perf_counter() - start
    pca_trustworthiness = trustworthiness(x_scaled, pca_embedding, n_neighbors=10)
    pca_variance = float(np.sum(pca.explained_variance_ratio_))

    rows.append(
        {
            "method": "PCA",
            "seconds": elapsed,
            "explained_variance_ratio": pca_variance,
            "trustworthiness_10nn": pca_trustworthiness,
        }
    )
    plot_projection(
        pca_embedding,
        y,
        "PCA 2D del espacio Mel",
        paths.figures / "pca_projection.png",
        target,
    )

    # Algoritmo t-SNE: metodo no lineal. Sirve para visualizar vecindarios locales.
    perplexity = min(30, max(5, (len(x_scaled) - 1) // 3))
    start = time.perf_counter()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=750,
        random_state=RANDOM_STATE,
    )
    tsne_embedding = tsne.fit_transform(x_scaled)
    elapsed = time.perf_counter() - start
    tsne_trustworthiness = trustworthiness(x_scaled, tsne_embedding, n_neighbors=10)

    rows.append(
        {
            "method": "t-SNE",
            "seconds": elapsed,
            "explained_variance_ratio": np.nan,
            "trustworthiness_10nn": tsne_trustworthiness,
        }
    )
    plot_projection(
        tsne_embedding,
        y,
        "t-SNE 2D del espacio Mel",
        paths.figures / "tsne_projection.png",
        target,
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(paths.tables / "dimensionality_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))
    return metrics_df


def valid_cluster_count(labels: np.ndarray) -> int:
    # En DBSCAN, la etiqueta -1 significa ruido y no cuenta como cluster.
    unique_labels = set(labels)
    unique_labels.discard(-1)
    return len(unique_labels)


def score_clustering(x_cluster: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    # Metricas internas: no usan las etiquetas reales, solo la geometria de los datos.
    cluster_count = valid_cluster_count(labels)
    non_noise_mask = labels != -1
    noise_ratio = float(np.mean(labels == -1))

    scores = {
        "n_clusters": cluster_count,
        "noise_ratio": noise_ratio,
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }

    if cluster_count < 2 or np.sum(non_noise_mask) <= cluster_count:
        return scores

    x_valid = x_cluster[non_noise_mask]
    labels_valid = labels[non_noise_mask]
    scores["silhouette"] = float(silhouette_score(x_valid, labels_valid))
    scores["calinski_harabasz"] = float(calinski_harabasz_score(x_valid, labels_valid))
    scores["davies_bouldin"] = float(davies_bouldin_score(x_valid, labels_valid))
    return scores


def plot_cluster_projection(
    pca_embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    # Proyecta los clusters a 2D con PCA para poder visualizarlos.
    unique_labels = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20")

    plt.figure()
    for index, label in enumerate(unique_labels):
        mask = labels == label
        label_name = "ruido" if label == -1 else f"cluster {label}"
        color = "#888888" if label == -1 else cmap(index % 20)
        plt.scatter(
            pca_embedding[mask, 0],
            pca_embedding[mask, 1],
            s=24,
            alpha=0.70,
            color=color,
            label=label_name,
        )
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_clustering_analysis(x_scaled: np.ndarray, paths: OutputPaths) -> pd.DataFrame:
    # Paso 4: mineria de patrones no supervisada con GMM y DBSCAN.
    print("\n=== Clustering no supervisado ===")

    # Se reduce primero con PCA al 95% de varianza para estabilizar clustering.
    pca_for_clustering = PCA(n_components=0.95, random_state=RANDOM_STATE)
    x_cluster = pca_for_clustering.fit_transform(x_scaled)
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(x_scaled)

    rows: list[dict[str, Any]] = []

    # Algoritmo GMM: clustering probabilistico con distinto numero de componentes.
    max_k = min(8, len(x_cluster) - 1)
    for n_components in range(2, max_k + 1):
        start = time.perf_counter()
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=5,
            random_state=RANDOM_STATE,
        )
        labels = gmm.fit_predict(x_cluster)
        elapsed = time.perf_counter() - start
        scores = score_clustering(x_cluster, labels)
        rows.append(
            {
                "algorithm": "GMM",
                "hyperparameters": f"n_components={n_components}",
                "seconds": elapsed,
                "bic": float(gmm.bic(x_cluster)),
                **scores,
            }
        )

    # Algoritmo DBSCAN: clustering por densidad. eps se estima con vecinos cercanos.
    min_samples = max(5, int(math.sqrt(x_cluster.shape[1] * 4)))
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(x_cluster)
    distances, _ = neighbors.kneighbors(x_cluster)
    kth_neighbor_distances = distances[:, -1]
    eps_candidates = np.quantile(kth_neighbor_distances, [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    eps_candidates = sorted({round(float(eps), 4) for eps in eps_candidates if eps > 0})

    for eps in eps_candidates:
        start = time.perf_counter()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(x_cluster)
        elapsed = time.perf_counter() - start
        scores = score_clustering(x_cluster, labels)
        rows.append(
            {
                "algorithm": "DBSCAN",
                "hyperparameters": f"eps={eps}, min_samples={min_samples}",
                "seconds": elapsed,
                "bic": np.nan,
                **scores,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(paths.tables / "clustering_metrics.csv", index=False)

    # Se guarda una figura para el mejor GMM y el mejor DBSCAN segun Silhouette.
    for algorithm in ["GMM", "DBSCAN"]:
        subset = metrics_df[metrics_df["algorithm"] == algorithm].dropna(subset=["silhouette"])
        if subset.empty:
            continue

        best_row = subset.sort_values(["silhouette", "calinski_harabasz"], ascending=False).iloc[0]
        if algorithm == "GMM":
            n_components = int(str(best_row["hyperparameters"]).split("=")[1])
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                n_init=5,
                random_state=RANDOM_STATE,
            )
            labels = model.fit_predict(x_cluster)
        else:
            parts = str(best_row["hyperparameters"]).replace("eps=", "").replace(" min_samples=", "").split(",")
            eps = float(parts[0])
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_cluster)

        plot_cluster_projection(
            pca_2d,
            labels,
            f"{algorithm}: mejor segmentacion segun Silhouette",
            paths.figures / f"{algorithm.lower()}_clusters.png",
        )

    print(metrics_df.to_string(index=False))
    return metrics_df


def build_ensemble_models() -> dict[str, Pipeline]:
    # Paso 5A: modelos de ensamble para comparar contra el MLP.
    return {
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.06,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
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
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def predict_proba_from_sklearn(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.named_steps["model"].classes_)
    return probabilities, classes


def evaluate_ensemble_models(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> list[ClassifierCandidate]:
    # Entrena cada ensamble y calcula metricas sobre validacion.
    candidates: list[ClassifierCandidate] = []

    print("\n=== Modelos de ensamble ===")
    for name, model in build_ensemble_models().items():
        print(f"Entrenando {name}...")
        start = time.perf_counter()
        fitted_model = clone(model).fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = fitted_model.predict(x_valid)
        probabilities, classes = predict_proba_from_sklearn(fitted_model, x_valid)
        predict_seconds = time.perf_counter() - start

        metrics = {
            "family": "ensemble",
            "model": name,
            "accuracy": accuracy_score(y_valid, predictions),
            "f1_macro": f1_score(y_valid, predictions, average="macro"),
            "f1_weighted": f1_score(y_valid, predictions, average="weighted"),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_sample": 1000 * predict_seconds / len(x_valid),
        }
        candidates.append(
            ClassifierCandidate(
                family="ensemble",
                name=name,
                model=fitted_model,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                classes=classes,
            )
        )

    return candidates


def encode_labels(y_train: pd.Series, y_valid: pd.Series | None = None) -> tuple[LabelEncoder, np.ndarray, np.ndarray | None]:
    # Convierte etiquetas originales, por ejemplo 10/12/17/18/23, a indices 0..K-1.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_valid_encoded = None if y_valid is None else encoder.transform(y_valid)
    return encoder, y_train_encoded, y_valid_encoded


def compute_class_weights(encoded_labels: np.ndarray) -> np.ndarray:
    # Pesos para compensar el desbalance entre especies.
    classes, counts = np.unique(encoded_labels, return_counts=True)
    weights = len(encoded_labels) / (len(classes) * counts)
    class_weights = np.ones(len(classes), dtype=np.float32)
    class_weights[classes] = weights.astype(np.float32)
    return class_weights


class NumpyMLPClassifier:
    # Paso 5B: red neuronal MLP implementada con NumPy.
    # Topologia por defecto: entrada -> 128 -> 64 -> salida.
    # La funcion de perdida es Cross-Entropy ponderada por clase.
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        variant: str,
        hidden_layers: tuple[int, int] = (128, 64),
        dropout_rate: float = 0.30,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if variant not in {"plain", "dropout_then_batchnorm", "batchnorm_then_dropout"}:
            raise ValueError(f"Variante MLP desconocida: {variant}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.variant = variant
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = np.random.default_rng(random_state)

        self.params: dict[str, np.ndarray] = {}
        self.running_means: dict[int, np.ndarray] = {}
        self.running_vars: dict[int, np.ndarray] = {}
        self.adam_m: dict[str, np.ndarray] = {}
        self.adam_v: dict[str, np.ndarray] = {}
        self.adam_step = 0
        self._initialize_parameters()

    @property
    def hidden_count(self) -> int:
        return len(self.hidden_layers)

    @property
    def output_layer(self) -> int:
        return self.hidden_count

    def _initialize_parameters(self) -> None:
        # Inicializacion He para capas con activacion ReLU.
        layer_dims = [self.input_dim, *self.hidden_layers, self.output_dim]

        for layer in range(len(layer_dims) - 1):
            fan_in = layer_dims[layer]
            fan_out = layer_dims[layer + 1]
            scale = np.sqrt(2.0 / fan_in)
            self.params[f"W{layer}"] = self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)
            self.params[f"b{layer}"] = np.zeros(fan_out, dtype=np.float32)

            if layer < self.hidden_count and self.variant != "plain":
                self.params[f"gamma{layer}"] = np.ones(fan_out, dtype=np.float32)
                self.params[f"beta{layer}"] = np.zeros(fan_out, dtype=np.float32)
                self.running_means[layer] = np.zeros(fan_out, dtype=np.float32)
                self.running_vars[layer] = np.ones(fan_out, dtype=np.float32)

        for name, value in self.params.items():
            self.adam_m[name] = np.zeros_like(value)
            self.adam_v[name] = np.zeros_like(value)

    def _batchnorm_forward(
        self,
        x: np.ndarray,
        layer: int,
        training: bool,
        momentum: float = 0.90,
        eps: float = 1e-5,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # Batch Normalization estabiliza las activaciones durante entrenamiento.
        gamma = self.params[f"gamma{layer}"]
        beta = self.params[f"beta{layer}"]

        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_means[layer] = momentum * self.running_means[layer] + (1 - momentum) * mean
            self.running_vars[layer] = momentum * self.running_vars[layer] + (1 - momentum) * var
        else:
            mean = self.running_means[layer]
            var = self.running_vars[layer]

        x_centered = x - mean
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = x_centered * inv_std
        output = gamma * x_hat + beta
        cache = {"x_hat": x_hat, "inv_std": inv_std, "gamma": gamma}
        return output, cache

    def _batchnorm_backward(
        self,
        dout: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_hat = cache["x_hat"]
        inv_std = cache["inv_std"]
        gamma = cache["gamma"]
        batch_size = dout.shape[0]

        dbeta = dout.sum(axis=0)
        dgamma = np.sum(dout * x_hat, axis=0)
        dx_hat = dout * gamma
        dx = (
            inv_std
            / batch_size
            * (
                batch_size * dx_hat
                - dx_hat.sum(axis=0)
                - x_hat * np.sum(dx_hat * x_hat, axis=0)
            )
        )
        return dx, dgamma, dbeta

    def _dropout_forward(self, x: np.ndarray, training: bool) -> tuple[np.ndarray, np.ndarray | None]:
        # Dropout apaga neuronas al azar para reducir sobreajuste.
        if not training or self.dropout_rate <= 0:
            return x, None

        keep_probability = 1.0 - self.dropout_rate
        mask = (self.rng.random(x.shape) < keep_probability).astype(np.float32) / keep_probability
        return x * mask, mask

    def _dropout_backward(self, dout: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return dout
        return dout * mask

    def _forward(
        self,
        x: np.ndarray,
        training: bool,
    ) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, np.ndarray]]:
        # Propagacion hacia adelante. La variante cambia el orden Dropout/BatchNorm.
        activations = x
        hidden_caches: list[dict[str, Any]] = []

        for layer in range(self.hidden_count):
            z = activations @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            cache: dict[str, Any] = {"layer": layer, "a_prev": activations, "z": z}

            if self.variant == "plain":
                activations = np.maximum(0.0, z)
            elif self.variant == "batchnorm_then_dropout":
                normalized, bn_cache = self._batchnorm_forward(z, layer, training)
                relu_output = np.maximum(0.0, normalized)
                activations, dropout_mask = self._dropout_forward(relu_output, training)
                cache.update({"normalized": normalized, "bn_cache": bn_cache, "dropout_mask": dropout_mask})
            else:
                relu_output = np.maximum(0.0, z)
                dropped, dropout_mask = self._dropout_forward(relu_output, training)
                activations, bn_cache = self._batchnorm_forward(dropped, layer, training)
                cache.update({"dropout_mask": dropout_mask, "bn_cache": bn_cache})

            hidden_caches.append(cache)

        output_layer = self.output_layer
        logits = activations @ self.params[f"W{output_layer}"] + self.params[f"b{output_layer}"]
        output_cache = {"a_prev": activations}
        return logits, hidden_caches, output_cache

    def _loss_and_probabilities(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        class_weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        # Cross-Entropy multiclase + regularizacion L2 sobre los pesos.
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        sample_weights = class_weights[y]
        negative_log_likelihood = -np.log(probabilities[np.arange(len(y)), y] + 1e-12)
        data_loss = float(np.mean(sample_weights * negative_log_likelihood))

        l2_loss = 0.0
        for layer in range(self.output_layer + 1):
            l2_loss += float(np.sum(self.params[f"W{layer}"] ** 2))

        return data_loss + 0.5 * self.weight_decay * l2_loss, probabilities

    def _backward(
        self,
        probabilities: np.ndarray,
        y: np.ndarray,
        class_weights: np.ndarray,
        hidden_caches: list[dict[str, Any]],
        output_cache: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        # Backpropagation: calcula gradientes para actualizar la red.
        grads: dict[str, np.ndarray] = {}
        batch_size = len(y)
        dlogits = probabilities.copy()
        dlogits[np.arange(batch_size), y] -= 1.0
        dlogits *= class_weights[y][:, None] / batch_size

        output_layer = self.output_layer
        grads[f"W{output_layer}"] = (
            output_cache["a_prev"].T @ dlogits + self.weight_decay * self.params[f"W{output_layer}"]
        )
        grads[f"b{output_layer}"] = dlogits.sum(axis=0)
        dactivation = dlogits @ self.params[f"W{output_layer}"].T

        for cache in reversed(hidden_caches):
            layer = int(cache["layer"])

            if self.variant == "plain":
                dz = dactivation * (cache["z"] > 0)
            elif self.variant == "batchnorm_then_dropout":
                dactivation = self._dropout_backward(dactivation, cache["dropout_mask"])
                dnormalized = dactivation * (cache["normalized"] > 0)
                dz, dgamma, dbeta = self._batchnorm_backward(dnormalized, cache["bn_cache"])
                grads[f"gamma{layer}"] = dgamma
                grads[f"beta{layer}"] = dbeta
            else:
                ddropped, dgamma, dbeta = self._batchnorm_backward(dactivation, cache["bn_cache"])
                grads[f"gamma{layer}"] = dgamma
                grads[f"beta{layer}"] = dbeta
                drelu = self._dropout_backward(ddropped, cache["dropout_mask"])
                dz = drelu * (cache["z"] > 0)

            grads[f"W{layer}"] = cache["a_prev"].T @ dz + self.weight_decay * self.params[f"W{layer}"]
            grads[f"b{layer}"] = dz.sum(axis=0)
            dactivation = dz @ self.params[f"W{layer}"].T

        return grads

    def _adam_update(self, grads: dict[str, np.ndarray]) -> None:
        # Optimizador Adam: actualiza parametros con momentos de primer y segundo orden.
        beta1 = 0.90
        beta2 = 0.999
        eps = 1e-8
        self.adam_step += 1

        for name, grad in grads.items():
            self.adam_m[name] = beta1 * self.adam_m[name] + (1 - beta1) * grad
            self.adam_v[name] = beta2 * self.adam_v[name] + (1 - beta2) * (grad**2)
            corrected_m = self.adam_m[name] / (1 - beta1**self.adam_step)
            corrected_v = self.adam_v[name] / (1 - beta2**self.adam_step)
            self.params[name] -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + eps)

    def _train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        class_weights: np.ndarray,
    ) -> float:
        indices = self.rng.permutation(len(x))
        losses = []

        for start in range(0, len(x), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]
            logits, hidden_caches, output_cache = self._forward(batch_x, training=True)
            loss, probabilities = self._loss_and_probabilities(logits, batch_y, class_weights)
            grads = self._backward(probabilities, batch_y, class_weights, hidden_caches, output_cache)
            self._adam_update(grads)
            losses.append(loss)

        return float(np.mean(losses))

    def _evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        class_weights: np.ndarray,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        probabilities = self.predict_proba(x)
        predictions = np.argmax(probabilities, axis=1)
        logits, _, _ = self._forward(x, training=False)
        loss, _ = self._loss_and_probabilities(logits, y, class_weights)
        f1 = f1_score(y, predictions, average="macro")
        return loss, f1, predictions, probabilities

    def get_state(self) -> dict[str, Any]:
        return {
            "params": {name: value.copy() for name, value in self.params.items()},
            "running_means": {layer: value.copy() for layer, value in self.running_means.items()},
            "running_vars": {layer: value.copy() for layer, value in self.running_vars.items()},
            "adam_m": {name: value.copy() for name, value in self.adam_m.items()},
            "adam_v": {name: value.copy() for name, value in self.adam_v.items()},
            "adam_step": self.adam_step,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.params = {name: value.copy() for name, value in state["params"].items()}
        self.running_means = {layer: value.copy() for layer, value in state["running_means"].items()}
        self.running_vars = {layer: value.copy() for layer, value in state["running_vars"].items()}
        self.adam_m = {name: value.copy() for name, value in state["adam_m"].items()}
        self.adam_v = {name: value.copy() for name, value in state["adam_v"].items()}
        self.adam_step = int(state["adam_step"])

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        epochs: int,
        batch_size: int,
        class_weights: np.ndarray,
    ) -> tuple[pd.DataFrame, int]:
        # Entrena por epocas y guarda la mejor version segun F1 macro en validacion.
        history: list[dict[str, float | int]] = []
        best_state = self.get_state()
        best_f1 = -np.inf
        best_epoch = 1
        patience = 15
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(x_train, y_train, batch_size, class_weights)
            valid_loss, valid_f1, _, _ = self._evaluate(x_valid, y_valid, class_weights)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_f1_macro": valid_f1,
                }
            )

            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_epoch = epoch
                best_state = self.get_state()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        self.set_state(best_state)
        return pd.DataFrame(history), best_epoch

    def fit_full(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        class_weights: np.ndarray,
    ) -> None:
        for _ in range(max(1, epochs)):
            self._train_epoch(x, y, batch_size, class_weights)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _, _ = self._forward(x, training=False)
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def train_mlp_variant(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    variant: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    # Entrena una variante especifica del MLP: plain, dropout_then_batchnorm, etc.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_valid_scaled = scaler.transform(x_valid).astype(np.float32)
    encoder, y_train_encoded, y_valid_encoded = encode_labels(y_train, y_valid)
    class_weights = compute_class_weights(y_train_encoded)

    model = NumpyMLPClassifier(
        input_dim=x_train_scaled.shape[1],
        output_dim=len(encoder.classes_),
        variant=variant,
    )
    history, best_epoch = model.fit(
        x_train=x_train_scaled,
        y_train=y_train_encoded,
        x_valid=x_valid_scaled,
        y_valid=y_valid_encoded,
        epochs=epochs,
        batch_size=batch_size,
        class_weights=class_weights,
    )

    probabilities = model.predict_proba(x_valid_scaled)
    encoded_predictions = np.argmax(probabilities, axis=1)
    predictions = encoder.inverse_transform(encoded_predictions)

    return {
        "variant": variant,
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "history": history,
        "best_epoch": best_epoch,
        "predictions": predictions,
        "probabilities": probabilities,
        "classes": encoder.classes_,
    }


def train_mlp_on_full_data(
    x: pd.DataFrame,
    y: pd.Series,
    variant: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    # Reentrena el mejor MLP con todos los datos de train antes de predecir test.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    encoder, y_encoded, _ = encode_labels(y)
    class_weights = compute_class_weights(y_encoded)

    model = NumpyMLPClassifier(
        input_dim=x_scaled.shape[1],
        output_dim=len(encoder.classes_),
        variant=variant,
    )
    model.fit_full(
        x=x_scaled,
        y=y_encoded,
        epochs=epochs,
        batch_size=batch_size,
        class_weights=class_weights,
    )

    return {"model": model, "scaler": scaler, "encoder": encoder, "variant": variant}


def plot_mlp_histories(histories: dict[str, pd.DataFrame], paths: OutputPaths) -> None:
    # Figuras solicitadas: curvas de Loss y F1 por epoca.
    plt.figure()
    for variant, history in histories.items():
        plt.plot(history["epoch"], history["valid_loss"], label=f"{variant} - valid")
    plt.title("Curvas de perdida de MLP")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths.figures / "mlp_loss_curves.png", dpi=160)
    plt.close()

    plt.figure()
    for variant, history in histories.items():
        plt.plot(history["epoch"], history["valid_f1_macro"], label=variant)
    plt.title("F1 macro de validacion en MLP")
    plt.xlabel("Epoca")
    plt.ylabel("F1 macro")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(paths.figures / "mlp_f1_curves.png", dpi=160)
    plt.close()


def evaluate_mlp_models(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    epochs: int,
    batch_size: int,
    paths: OutputPaths,
) -> list[ClassifierCandidate]:
    # Compara tres configuraciones de regularizacion de la red MLP.
    variants = ["plain", "dropout_then_batchnorm", "batchnorm_then_dropout"]
    histories: dict[str, pd.DataFrame] = {}
    candidates: list[ClassifierCandidate] = []

    print("\n=== Redes neuronales MLP ===")
    for variant in variants:
        print(f"Entrenando MLP: {variant}...")
        start = time.perf_counter()
        result = train_mlp_variant(
            x_train=x_train,
            x_valid=x_valid,
            y_train=y_train,
            y_valid=y_valid,
            variant=variant,
            epochs=epochs,
            batch_size=batch_size,
        )
        fit_seconds = time.perf_counter() - start
        histories[variant] = result["history"]
        result["history"].to_csv(paths.tables / f"mlp_history_{variant}.csv", index=False)

        start = time.perf_counter()
        x_valid_scaled = result["scaler"].transform(x_valid).astype(np.float32)
        probabilities = result["model"].predict_proba(x_valid_scaled)
        encoded_predictions = np.argmax(probabilities, axis=1)
        predictions = result["encoder"].inverse_transform(encoded_predictions)
        predict_seconds = time.perf_counter() - start

        metrics = {
            "family": "mlp",
            "model": variant,
            "accuracy": accuracy_score(y_valid, predictions),
            "f1_macro": f1_score(y_valid, predictions, average="macro"),
            "f1_weighted": f1_score(y_valid, predictions, average="weighted"),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_sample": 1000 * predict_seconds / len(x_valid),
            "best_epoch": result["best_epoch"],
        }
        candidates.append(
            ClassifierCandidate(
                family="mlp",
                name=variant,
                model=result,
                metrics=metrics,
                predictions=predictions,
                probabilities=probabilities,
                classes=result["classes"],
            )
        )

    plot_mlp_histories(histories, paths)
    return candidates


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    # Matriz de confusion para analizar que clases se confunden entre si.
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def select_best_classifier(candidates: list[ClassifierCandidate]) -> ClassifierCandidate:
    # El mejor modelo se elige primero por F1 macro y luego por accuracy.
    if not candidates:
        raise RuntimeError("No se entreno ningun clasificador.")

    return sorted(
        candidates,
        key=lambda item: (float(item.metrics["f1_macro"]), float(item.metrics["accuracy"])),
        reverse=True,
    )[0]


def confidence_zone(confidence: float) -> str:
    # Paso 6: politica de mitigacion de riesgo basada en probabilidad.
    if confidence >= CONFIDENCE_THRESHOLD:
        return "confianza"
    if confidence >= REVIEW_THRESHOLD:
        return "incertidumbre"
    return "rechazo"


def build_prediction_table(
    ids: pd.Series,
    target: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    id_column: str = "recording_id",
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    # Crea la tabla final con prediccion, probabilidad y zona operativa.
    confidence = probabilities.max(axis=1)
    prediction_df = pd.DataFrame(
        {
            id_column: ids,
            f"predicted_{target}": predictions,
            "confidence": confidence,
            "decision_zone": [confidence_zone(value) for value in confidence],
        }
    )

    if y_true is not None:
        prediction_df.insert(1, f"true_{target}", y_true.to_numpy())

    for index, class_value in enumerate(classes):
        prediction_df[f"prob_{class_value}"] = probabilities[:, index]

    return prediction_df


def save_policy_summary(prediction_df: pd.DataFrame, paths: OutputPaths, filename: str) -> None:
    summary = (
        prediction_df["decision_zone"]
        .value_counts()
        .rename_axis("decision_zone")
        .reset_index(name="count")
    )
    summary["percentage"] = summary["count"] / summary["count"].sum()
    summary.to_csv(paths.tables / filename, index=False)


def final_fit_and_predict(
    best: ClassifierCandidate,
    x_full: pd.DataFrame,
    y_full: pd.Series,
    x_test: pd.DataFrame,
    epochs: int,
    batch_size: int,
    paths: OutputPaths,
    feature_columns: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Reentrena el mejor clasificador con todo train y genera predicciones de test.
    if best.family == "ensemble":
        model = clone(build_ensemble_models()[best.name]).fit(x_full, y_full)
        probabilities, classes = predict_proba_from_sklearn(model, x_test)
        predictions = classes[np.argmax(probabilities, axis=1)]
        joblib.dump(model, paths.models / "best_model.joblib")
        metadata = {
            "family": best.family,
            "model": best.name,
            "target": target,
            "feature_columns": feature_columns,
        }
        (paths.models / "best_model_metadata.json").write_text(json.dumps(metadata, indent=2))
        return predictions, probabilities, classes

    best_epoch = int(best.metrics.get("best_epoch", epochs))
    bundle = train_mlp_on_full_data(
        x=x_full,
        y=y_full,
        variant=best.name,
        epochs=best_epoch,
        batch_size=batch_size,
    )
    x_test_scaled = bundle["scaler"].transform(x_test).astype(np.float32)
    probabilities = bundle["model"].predict_proba(x_test_scaled)
    classes = bundle["encoder"].classes_
    predictions = classes[np.argmax(probabilities, axis=1)]

    joblib.dump(
        {
            "model": bundle["model"],
            "scaler": bundle["scaler"],
            "label_encoder": bundle["encoder"],
            "variant": best.name,
            "target": target,
            "feature_columns": feature_columns,
        },
        paths.models / "best_mlp_model.joblib",
    )
    return predictions, probabilities, classes


def save_classification_outputs(
    candidates: list[ClassifierCandidate],
    best: ClassifierCandidate,
    x_full: pd.DataFrame,
    y_full: pd.Series,
    x_test: pd.DataFrame,
    test_df: pd.DataFrame,
    y_valid: pd.Series,
    target: str,
    feature_columns: list[str],
    epochs: int,
    batch_size: int,
    paths: OutputPaths,
) -> None:
    # Guarda metricas, reportes, matrices de confusion, modelo final y predicciones.
    metrics_df = pd.DataFrame([candidate.metrics for candidate in candidates]).sort_values(
        by=["f1_macro", "accuracy"],
        ascending=False,
    )
    metrics_df.to_csv(paths.tables / "classification_metrics.csv", index=False)

    print("\n=== Comparacion de clasificadores ===")
    print(metrics_df.to_string(index=False))
    print(f"\nMejor clasificador: {best.family}/{best.name}")

    save_confusion_matrix(
        y_valid,
        best.predictions,
        labels=best.classes,
        title=f"Matriz de confusion - validacion ({best.name})",
        output_path=paths.figures / "best_validation_confusion_matrix.png",
    )

    report = classification_report(y_valid, best.predictions, zero_division=0)
    (paths.tables / "best_validation_classification_report.txt").write_text(report)

    validation_predictions = build_prediction_table(
        ids=pd.Series(y_valid.index, name="validation_index"),
        target=target,
        predictions=best.predictions,
        probabilities=best.probabilities,
        classes=best.classes,
        id_column="validation_index",
        y_true=y_valid.reset_index(drop=True),
    )
    validation_predictions.to_csv(paths.tables / "validation_predictions_with_policy.csv", index=False)
    save_policy_summary(validation_predictions, paths, "validation_policy_summary.csv")

    test_predictions, test_probabilities, test_classes = final_fit_and_predict(
        best=best,
        x_full=x_full,
        y_full=y_full,
        x_test=x_test,
        epochs=epochs,
        batch_size=batch_size,
        paths=paths,
        feature_columns=feature_columns,
        target=target,
    )

    y_test = test_df[target] if target in test_df.columns else None
    prediction_table = build_prediction_table(
        ids=test_df["recording_id"],
        target=target,
        predictions=test_predictions,
        probabilities=test_probabilities,
        classes=test_classes,
        y_true=y_test,
    )
    prediction_table.to_csv(paths.root / "test_predictions.csv", index=False)
    save_policy_summary(prediction_table, paths, "test_policy_summary.csv")

    if y_test is not None:
        test_report = classification_report(y_test, test_predictions, zero_division=0)
        (paths.tables / "test_classification_report.txt").write_text(test_report)
        test_metrics = pd.DataFrame(
            [
                {
                    "accuracy": accuracy_score(y_test, test_predictions),
                    "f1_macro": f1_score(y_test, test_predictions, average="macro"),
                    "f1_weighted": f1_score(y_test, test_predictions, average="weighted"),
                }
            ]
        )
        test_metrics.to_csv(paths.tables / "test_metrics.csv", index=False)
        save_confusion_matrix(
            y_test,
            test_predictions,
            labels=test_classes,
            title=f"Matriz de confusion - test ({best.name})",
            output_path=paths.figures / "test_confusion_matrix.png",
        )


def main() -> None:
    # Orquestador principal: ejecuta todas las etapas en orden.
    args = parse_args()
    configure_plots()
    paths = prepare_output_paths(args.output_dir)

    train_df, test_df = load_data(args.train, args.test)
    feature_columns = get_feature_columns(train_df, args.target, args.include_metadata)
    validate_inputs(train_df, test_df, args.target, feature_columns)

    save_dataset_summary(train_df, test_df, args.target, feature_columns, paths)
    print(f"\nCaracteristicas usadas ({len(feature_columns)}): {feature_columns[:8]} ...")

    x = train_df[feature_columns]
    y = train_df[args.target]
    x_test = test_df[feature_columns]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler_for_analysis = StandardScaler()
    x_scaled = scaler_for_analysis.fit_transform(x)

    # Bloque de reduccion dimensional: PCA vs t-SNE.
    if not args.skip_geometry:
        run_dimensionality_analysis(x_scaled, y, args.target, paths)

    # Bloque de clustering: GMM vs DBSCAN.
    if not args.skip_clustering:
        run_clustering_analysis(x_scaled, paths)

    # Bloque de clasificacion: ensambles y MLP.
    candidates = evaluate_ensemble_models(x_train, x_valid, y_train, y_valid)

    if not args.skip_mlp:
        candidates.extend(
            evaluate_mlp_models(
                x_train=x_train,
                x_valid=x_valid,
                y_train=y_train,
                y_valid=y_valid,
                epochs=args.epochs,
                batch_size=args.batch_size,
                paths=paths,
            )
        )

    best = select_best_classifier(candidates)
    save_classification_outputs(
        candidates=candidates,
        best=best,
        x_full=x,
        y_full=y,
        x_test=x_test,
        test_df=test_df,
        y_valid=y_valid,
        target=args.target,
        feature_columns=feature_columns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        paths=paths,
    )

    print("\n=== Archivos principales generados ===")
    for output_path in [
        paths.tables / "classification_metrics.csv",
        paths.tables / "dimensionality_metrics.csv",
        paths.tables / "clustering_metrics.csv",
        paths.root / "test_predictions.csv",
        paths.figures,
    ]:
        if output_path.exists():
            print(output_path)


if __name__ == "__main__":
    main()
