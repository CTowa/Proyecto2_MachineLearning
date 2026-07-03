from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# Configuracion de la interfaz Streamlit
# ============================================================
# Mapa con P2_ML.pdf:
# - 3.1: interfaz opcional Streamlit dentro de la arquitectura del pipeline.
# - 3.5: simulacion de inferencia y politicas de umbrales probabilisticos.
ROOT = Path(__file__).parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
TEST_PATH = ROOT / "eco_acoustic_test.csv"
PIPELINE_SCRIPT = "main_adjusted.py"
PIPELINE_COMMAND = f"python {PIPELINE_SCRIPT}"

SPECIES_NAMES = {
    10: "Leptodactylus discodactylus",
    12: "Osteocephalus taurinus",
    17: "Chiroxiphia lineata",
    18: "Saltator grossus",
    23: "Pheucticus chrysopeplus",
}

ZONE_LABELS = {
    "confianza": "Zona de confianza",
    "incertidumbre": "Zona de incertidumbre",
    "rechazo": "Zona de rechazo",
}

ZONE_MESSAGES = {
    "confianza": "Clasificacion automatica: la probabilidad supera el umbral operativo.",
    "incertidumbre": "Clasificacion asistida: el registro debe revisarse por un experto.",
    "rechazo": "Descarte automatico: la confianza es baja y se mitiga el riesgo de falsos positivos.",
}

ZONE_COLORS = {
    "confianza": "#0f7b45",
    "incertidumbre": "#946200",
    "rechazo": "#b42318",
}

PROJECT_CHECKLIST = [
    {
        "section": "3.1",
        "item": "Carga de datos y espacio vectorial X en R64",
        "status": "Implementado",
        "evidence": "main_adjusted.py valida y usa mel_0 a mel_63 como variables de entrada.",
    },
    {
        "section": "3.2",
        "item": "Reduccion dimensional lineal y no lineal",
        "status": "Implementado",
        "evidence": "PCA y t-SNE generan metricas, tiempos y proyecciones 2D.",
    },
    {
        "section": "3.3",
        "item": "Clustering con validacion interna",
        "status": "Implementado",
        "evidence": "GMM y DBSCAN se comparan con metricas internas, curva BIC y curva k-distancia.",
    },
    {
        "section": "3.4",
        "item": "MLP, regularizacion y comparacion contra ensambles",
        "status": "Implementado",
        "evidence": "El MLP compara variantes con train/valid loss, F1 macro y matrices normalizadas.",
    },
    {
        "section": "3.5",
        "item": "MLOps, costo de inferencia y politica de decision",
        "status": "Implementado",
        "evidence": "Se guardan tiempos, predicciones, probabilidades y zonas de confianza.",
    },
    {
        "section": "3.6",
        "item": "Contribution statement del equipo",
        "status": "Pendiente en informe",
        "evidence": "Debe agregarse como tabla final en el documento escrito; no depende del codigo.",
    },
]

FIGURE_GROUPS = {
    "Exploracion del dataset": [
        (
            "Distribucion de clases",
            "target_distribution.png",
            "Verifica el balance de especies antes de entrenar. Cuando una especie tiene menos muestras, la metrica F1 macro es mas representativa que el accuracy global.",
        ),
    ],
    "Reduccion dimensional": [
        (
            "PCA 2D",
            "pca_projection.png",
            "Resume el espacio Mel en dos componentes lineales para revisar separacion global y cuanta varianza explica cada componente.",
        ),
        (
            "t-SNE 2D",
            "tsne_projection.png",
            "Visualiza vecindarios no lineales del espacio Mel. Sirve para inspeccionar grupos locales, aunque no mide varianza explicada.",
        ),
    ],
    "Clustering no supervisado": [
        (
            "Curva BIC GMM",
            "gmm_bic_curve.png",
            "Justifica cuantos componentes se probaron en el modelo probabilistico GMM. Un BIC menor sugiere mejor balance entre ajuste y complejidad.",
        ),
        (
            "Curva k-distancia DBSCAN",
            "dbscan_k_distance.png",
            "Muestra la escala de distancias usada para proponer valores de eps. El cambio de pendiente revela por que eps no se eligio arbitrariamente.",
        ),
        (
            "GMM clusters",
            "gmm_clusters.png",
            "Muestra la mejor segmentacion probabilistica segun el coeficiente de Silhouette. Clusters compactos y separados indican estructura acustica recuperable.",
        ),
        (
            "DBSCAN clusters",
            "dbscan_clusters.png",
            "Muestra agrupaciones por densidad y posibles registros de ruido. Cuando hay mucho ruido, la densidad no separa bien todas las especies.",
        ),
    ],
    "Clasificacion supervisada": [
        (
            "Curvas de perdida MLP",
            "mlp_loss_curves.png",
            "Compara la estabilidad del entrenamiento entre variantes de MLP. Una curva que desciende y se estabiliza indica aprendizaje sin oscilaciones fuertes.",
        ),
        (
            "F1 macro MLP",
            "mlp_f1_curves.png",
            "Evalua en que epoca y con que variante mejora el rendimiento promedio por clase. La variante con mayor F1 macro balancea mejor todas las especies.",
        ),
        (
            "Matriz de confusion en validacion",
            "best_validation_confusion_matrix.png",
            "Identifica los errores por especie dentro del conjunto de validacion. Las celdas fuera de la diagonal revelan que especies el modelo confunde entre si.",
        ),
        (
            "Matriz de validacion normalizada",
            "best_validation_confusion_matrix_normalized.png",
            "Compara errores por clase sin que domine la cantidad de muestras. Cada fila suma 1 y la diagonal muestra el recall por especie.",
        ),
        (
            "Matriz de confusion en test",
            "test_confusion_matrix.png",
            "Resume los errores finales sobre datos que el modelo nunca vio durante el entrenamiento. Una diagonal dominante indica buena generalizacion.",
        ),
        (
            "Matriz de test normalizada",
            "test_confusion_matrix_normalized.png",
            "Evalua el desempeno final clase por clase en escala relativa. Es la vista mas justa cuando las clases no tienen el mismo numero de muestras.",
        ),
    ],
}


def species_label(value: int | str) -> str:
    try:
        species_id = int(value)
    except (TypeError, ValueError):
        return str(value)

    name = SPECIES_NAMES.get(species_id, "especie no catalogada")
    return f"{species_id} - {name}"


def configure_page() -> None:
    st.set_page_config(
        page_title="Inferencia Eco-Acustica",
        page_icon=None,
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .app-hero {
            background: linear-gradient(135deg, #f5f7fb 0%, #edf4f2 100%);
            border: 1px solid #dce5e2;
            border-radius: 8px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1rem;
        }
        .app-hero h1 {
            margin: 0 0 0.35rem 0;
            color: #183c40;
            font-size: 2rem;
            letter-spacing: 0;
        }
        .app-hero p {
            margin: 0;
            color: #40514e;
            font-size: 1rem;
        }
        .decision-box {
            border-radius: 8px;
            padding: 1rem;
            color: white;
            font-weight: 600;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .section-explanation {
            color: #40514e;
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e5e9e8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "run_metadata": output_dir / "pipeline_run_metadata.json",
        "predictions": output_dir / "test_predictions.csv",
        "dataset_summary": output_dir / "tables" / "dataset_summary.csv",
        "target_distribution": output_dir / "tables" / "target_distribution.csv",
        "classification_metrics": output_dir / "tables" / "classification_metrics.csv",
        "dimensionality_metrics": output_dir / "tables" / "dimensionality_metrics.csv",
        "clustering_metrics": output_dir / "tables" / "clustering_metrics.csv",
        "test_metrics": output_dir / "tables" / "test_metrics.csv",
        "test_policy_summary": output_dir / "tables" / "test_policy_summary.csv",
        "best_params": output_dir / "tables" / "best_params.json",
        "calibrated_thresholds": output_dir / "tables" / "calibrated_thresholds.json",
        "figures": output_dir / "figures",
    }


def render_hero() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>Dashboard eco-acustico</h1>
            <p>
                Visualizacion de resultados del pipeline de clasificacion: exploracion de datos,
                reduccion dimensional, clustering, clasificacion supervisada y politica de decision
                con umbrales probabilisticos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _has_true_label(prediction_df: pd.DataFrame) -> bool:
    return "true_species_id" in prediction_df.columns


def require_outputs(paths: dict[str, Path]) -> bool:
    missing = [
        label
        for label, path in {
            "metadata del pipeline": paths["run_metadata"],
            "predicciones": paths["predictions"],
        }.items()
        if not path.exists()
    ]
    if missing:
        st.warning("Faltan archivos necesarios: " + ", ".join(missing) + ".")
        st.markdown(
            "La interfaz lee resultados ya calculados por el pipeline para evitar reentrenar modelos desde Streamlit. "
            "Ejecuta el pipeline primero y luego vuelve a abrir esta aplicacion."
        )
        st.code(PIPELINE_COMMAND, language="bash")
        return False

    metadata = load_json(paths["run_metadata"])
    producer = metadata.get("producer_script")
    if producer != PIPELINE_SCRIPT:
        st.warning(
            f"La carpeta seleccionada fue generada por `{producer}`. "
            f"Esta app espera salidas de `{PIPELINE_SCRIPT}`."
        )
        st.code(PIPELINE_COMMAND, language="bash")
        return False

    return True


def probability_columns(prediction_df: pd.DataFrame) -> list[str]:
    return [column for column in prediction_df.columns if column.startswith("prob_")]


def selected_probability_table(row: pd.Series) -> pd.DataFrame:
    rows = []
    for column in probability_columns(row.to_frame().T):
        species_id = column.replace("prob_", "")
        rows.append(
            {
                "species_id": species_id,
                "species": species_label(species_id),
                "probability": float(row[column]),
            }
        )

    return pd.DataFrame(rows).sort_values("probability", ascending=False)


def selected_mel_table(test_row: pd.Series) -> pd.DataFrame:
    mel_columns = sorted(
        [column for column in test_row.index if column.startswith("mel_")],
        key=lambda column: int(column.split("_")[1]),
    )
    return pd.DataFrame(
        {
            "mel_feature": mel_columns,
            "value": [float(test_row[column]) for column in mel_columns],
        }
    )


def format_percentage(value: float | int | str) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


def format_decimal(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def render_dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    options = {
        "use_container_width": True,
        "hide_index": True,
    }
    if height is not None:
        options["height"] = height

    st.dataframe(df, **options)


def decision_box(zone: str, confidence: float) -> None:
    color = ZONE_COLORS.get(zone, "#333333")
    label = ZONE_LABELS.get(zone, zone)
    message = ZONE_MESSAGES.get(zone, "")
    st.markdown(
        f"""
        <div class="decision-box" style="background:{color};">
            {label}: {confidence:.2%}<br>
            <span style="font-weight:400;">{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(prediction_df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Escenarios")

    zone_options = ["todas", *sorted(prediction_df["decision_zone"].dropna().unique())]
    zone_display = {"todas": "Todas las zonas", **ZONE_LABELS}
    selected_zone = st.sidebar.selectbox(
        "Filtrar por zona",
        zone_options,
        format_func=lambda x: zone_display.get(x, x),
    )

    filtered_df = prediction_df.copy()
    if selected_zone != "todas":
        filtered_df = filtered_df[filtered_df["decision_zone"] == selected_zone]

    if filtered_df.empty:
        st.sidebar.warning("No hay registros para ese filtro.")
        return prediction_df

    st.sidebar.caption("Distribucion filtrada")
    for zone in ["confianza", "incertidumbre", "rechazo"]:
        count = filtered_df["decision_zone"].value_counts().get(zone, 0)
        st.sidebar.write(f"{ZONE_LABELS[zone]}: {count}")
    st.sidebar.caption(f"Registros disponibles: {len(filtered_df)}")
    return filtered_df


def render_project_overview(
    paths: dict[str, Path],
    prediction_df: pd.DataFrame | None = None,
) -> None:

    st.markdown(
        "Pipeline de clasificacion eco-acustica que procesa **64 coeficientes Mel** "
        "de grabaciones de aves y anfibios. Aplica reduccion dimensional (PCA, t-SNE), "
        "clustering no supervisado (GMM, DBSCAN), clasificacion supervisada con MLP y "
        "ensambles, y una politica de decision basada en umbrales de confianza calibrados. "
        "Esta seccion resume el desempeno global del modelo sobre los datos de prueba."
    )

    if prediction_df is None:
        return

    confidence_mean = prediction_df["confidence"].mean()
    zone_counts = prediction_df["decision_zone"].value_counts()
    automatic_ratio = zone_counts.get("confianza", 0) / len(prediction_df)
    review_ratio = zone_counts.get("incertidumbre", 0) / len(prediction_df)
    reject_ratio = zone_counts.get("rechazo", 0) / len(prediction_df)

    # === Metricas principales ===
    st.subheader("Metricas principales")
    cols = st.columns(4)
    cols[0].metric("Registros en test", f"{len(prediction_df):,}")
    cols[1].metric("Confianza promedio", format_percentage(confidence_mean))
    cols[2].metric("Decision automatica", format_percentage(automatic_ratio))

    if _has_true_label(prediction_df):
        correct = (prediction_df["true_species_id"] == prediction_df["predicted_species_id"]).sum()
        cols[3].metric("Aciertos", f"{correct:,} / {len(prediction_df):,}")
    else:
        cols[3].metric("Especies detectadas", f"{prediction_df['predicted_species_id'].nunique()}")

    st.divider()

    # === Distribucion por zona ===
    st.subheader("Distribucion por zona operativa")
    st.markdown(
        "Las predicciones se clasifican en tres zonas segun la confianza del modelo. "
        "Esto permite decidir que registros se aceptan automaticamente, cuales requieren "
        "revision de un experto y cuales se descartan para evitar falsos positivos."
    )
    zone_data = pd.DataFrame({
        "Zona": ["Confianza", "Incertidumbre", "Rechazo"],
        "Cantidad": [
            zone_counts.get("confianza", 0),
            zone_counts.get("incertidumbre", 0),
            zone_counts.get("rechazo", 0),
        ],
    })
    zone_data["Porcentaje"] = zone_data["Cantidad"].apply(lambda x: f"{x / len(prediction_df):.1%}")

    render_dataframe(zone_data[["Zona", "Cantidad", "Porcentaje"]])

    chart_col, summary_col = st.columns([1, 1])
    with chart_col:
        st.bar_chart(zone_data.set_index("Zona")["Cantidad"], horizontal=True)
    with summary_col:
        st.markdown("**Interpretacion**")
        if automatic_ratio > 0.5:
            st.success(f"{format_percentage(automatic_ratio)} de las predicciones se aceptan automaticamente.")
        else:
            st.warning(f"Solo {format_percentage(automatic_ratio)} son automaticas; la mayoria requiere revision.")
        if reject_ratio > 0.05:
            st.error(f"{format_percentage(reject_ratio)} de registros son rechazados (confianza baja).")
        if review_ratio > 0.3:
            st.info(f"{format_percentage(review_ratio)} requieren revision experta.")


    # === Rendimiento por especie ===
    st.subheader("Rendimiento por especie")
    st.markdown(
        "Desglose de confianza promedio por cada especie que el modelo predice. "
        "Las especies con mayor confianza son aquellas que el modelo reconoce con mas seguridad."
    )
    species_stats = prediction_df.groupby("predicted_species_id").agg(
        Registros=("confidence", "count"),
        Confianza_promedio=("confidence", "mean"),
    ).reset_index()
    species_stats["Especie"] = species_stats["predicted_species_id"].apply(species_label)
    species_stats["Confianza_promedio"] = species_stats["Confianza_promedio"].apply(format_percentage)
    render_dataframe(species_stats[["Especie", "Registros", "Confianza_promedio"]])

    st.divider()

    # === Conclusiones rapidas ===
    st.subheader("Conclusiones rapidas")
    conclusions = []

    if _has_true_label(prediction_df):
        acc = correct / len(prediction_df)
        if acc > 0.7:
            conclusions.append(f"**Precision global de {format_percentage(acc)}**: el modelo generaliza bien.")
        elif acc > 0.4:
            conclusions.append(f"**Precision global de {format_percentage(acc)}**: rendimiento moderado, mejorable con mas datos.")
        else:
            conclusions.append(f"**Precision global de {format_percentage(acc)}**: el modelo presenta dificultades, se recomienda revisar features.")

    if automatic_ratio > 0.5:
        conclusions.append(f"Alta autonomia: **{format_percentage(automatic_ratio)}** de decisiones son automaticas.")
    else:
        conclusions.append(f"Baja autonomia: solo **{format_percentage(automatic_ratio)}** son automaticas. El sistema prioriza la revision humana.")

    if reject_ratio > 0.05:
        conclusions.append(f"Mitigacion de riesgos: **{format_percentage(reject_ratio)}** de registros son rechazados para evitar falsos positivos.")

    species_distribution = prediction_df["predicted_species_id"].value_counts()
    most_common = species_distribution.idxmax()
    conclusions.append(f"Especie mas predicha: **{species_label(most_common)}** ({species_distribution.max()} registros).")

    species_count = prediction_df["predicted_species_id"].nunique()
    conclusions.append(f"El modelo distingue entre **{species_count} especies** distintas.")

    for c in conclusions:
        st.write(f"- {c}")


def render_inference_tab(prediction_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    filtered_df = render_sidebar(prediction_df)
    recording_ids = filtered_df["recording_id"].astype(str).tolist()
    selected_recording = st.sidebar.selectbox("Seleccionar recording_id", recording_ids)

    prediction_row = prediction_df[
        prediction_df["recording_id"].astype(str) == selected_recording
    ].iloc[0]
    test_row = test_df[test_df["recording_id"].astype(str) == selected_recording].iloc[0]

    predicted_species = prediction_row["predicted_species_id"]
    true_species = prediction_row.get("true_species_id", test_row.get("species_id", "N/A"))
    confidence = float(prediction_row["confidence"])
    zone = str(prediction_row["decision_zone"])

    st.subheader("Simulador de inferencia")
    st.markdown(
        "Selecciona un registro de prueba desde la barra lateral para revisar como el modelo "
        "lo clasifica. La prediccion se acepta automaticamente, se envia a revision experta "
        "o se rechaza segun la confianza maxima entre todas las especies candidatas."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Recording ID", selected_recording)
    metric_columns[1].metric("Especie real", species_label(true_species))
    metric_columns[2].metric("Prediccion", species_label(predicted_species))
    metric_columns[3].metric("Confianza", f"{confidence:.2%}")

    decision_box(zone, confidence)

    probability_df = selected_probability_table(prediction_row)
    top_probability = probability_df.iloc[0]
    second_probability = probability_df.iloc[1] if len(probability_df) > 1 else None
    probability_margin = (
        top_probability["probability"] - second_probability["probability"]
        if second_probability is not None
        else top_probability["probability"]
    )

    st.markdown("#### Lectura de la prediccion")
    st.write(
        "El modelo asigna la mayor probabilidad a "
        f"**{top_probability['species']}**. "
        f"La diferencia contra la segunda opcion es **{probability_margin:.2%}**, "
        "lo que ayuda a medir si la decision es clara o requiere revision."
    )

    chart_columns = st.columns([1.05, 0.95])
    with chart_columns[0]:
        st.markdown("#### Probabilidades por especie")
        st.markdown(
            "Cada barra representa la probabilidad que el modelo asigna a cada clase "
            "para este registro. Una barra claramente dominante indica una decision estable; "
            "barras de altura similar senalan incertidumbre."
        )
        st.bar_chart(probability_df.set_index("species")["probability"])
        render_dataframe(probability_df)

    with chart_columns[1]:
        st.markdown("#### Vector Mel del registro")
        st.markdown(
            "Las 64 variables acusticas (coeficientes Mel) que entran al clasificador. "
            "Los picos y valles describen la distribucion de energia en distintas bandas "
            "de frecuencia; no son probabilidades."
        )
        mel_df = selected_mel_table(test_row)
        st.line_chart(mel_df.set_index("mel_feature")["value"])
        render_dataframe(mel_df, height=260)


def render_dataset_tables(paths: dict[str, Path]) -> None:
    st.subheader("Datos y balance")

    if paths["dataset_summary"].exists():
        st.markdown("#### Resumen del dataset")
        st.markdown(
            "Confirma las particiones usadas (entrenamiento, validacion, prueba), "
            "la variable objetivo, la cantidad de features (64 Mel) y si hay valores faltantes."
        )
        render_dataframe(load_csv(paths["dataset_summary"]))

    if paths["target_distribution"].exists():
        st.markdown("#### Distribucion del objetivo")
        st.markdown(
            "Cuantas muestras hay por especie antes de entrenar. Si las clases estan "
            "desbalanceadas, la metrica F1 macro es mas representativa que el accuracy global."
        )
        target_df = load_csv(paths["target_distribution"])
        render_dataframe(target_df)
        if "label" in target_df.columns and "count" in target_df.columns:
            st.bar_chart(target_df.set_index("label")["count"])


def render_metrics_tab(paths: dict[str, Path]) -> None:
    render_dataset_tables(paths)

    if paths["classification_metrics"].exists():
        metrics_df = load_csv(paths["classification_metrics"])
        st.markdown("#### Comparacion de clasificadores")
        st.markdown(
            "Compara el rendimiento de MLP y ensambles (Random Forest, XGBoost) usando "
            "F1 macro y costo computacional. El mejor candidato equilibra calidad predictiva "
            "con velocidad de inferencia para ser viable en produccion."
        )
        render_dataframe(metrics_df)
        best_row = metrics_df.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Mejor familia", str(best_row["family"]))
        cols[1].metric("Mejor modelo", str(best_row["model"]))
        cols[2].metric("F1 macro", f"{best_row['f1_macro']:.4f}")
        cols[3].metric("Inferencia ms/muestra", f"{best_row['predict_ms_per_sample']:.3f}")

    if paths["test_metrics"].exists():
        st.markdown("#### Desempeno en test")
        st.markdown(
            "Rendimiento final del modelo seleccionado sobre el conjunto de prueba. "
            "F1 macro resume el equilibrio entre especies; accuracy global da una idea "
            "general de aciertos."
        )
        render_dataframe(load_csv(paths["test_metrics"]))

    if paths["test_policy_summary"].exists():
        st.markdown("#### Distribucion por zona operativa")
        st.markdown(
            "Cuantas predicciones caen en cada zona operativa. Una proporcion alta en "
            "incertidumbre o rechazo indica que el sistema prioriza la cautela sobre "
            "la automatizacion."
        )
        policy_df = load_csv(paths["test_policy_summary"])
        if "percentage" in policy_df.columns:
            policy_df["percentage_label"] = policy_df["percentage"].map(format_percentage)
        render_dataframe(policy_df)
        st.bar_chart(policy_df.set_index("decision_zone")["count"])

    if paths["dimensionality_metrics"].exists():
        st.markdown("#### PCA vs t-SNE")
        st.markdown(
            "Compara la reduccion lineal (PCA) contra la no lineal (t-SNE) para entender "
            "la geometria del espacio Mel. Una mayor trustworthiness indica que se conservan "
            "mejor los vecinos; PCA ademas reporta la varianza explicada por cada componente."
        )
        dimensionality_df = load_csv(paths["dimensionality_metrics"])
        render_dataframe(dimensionality_df)

    if paths["clustering_metrics"].exists():
        st.markdown("#### GMM vs DBSCAN")
        st.markdown(
            "Evalua si hay estructura no supervisada en los datos antes de usar etiquetas. "
            "Silhouette y Calinski-Harabasz altos indican clusters compactos; Davies-Bouldin "
            "y BIC bajos son mejores."
        )
        clustering_df = load_csv(paths["clustering_metrics"])
        render_dataframe(clustering_df)


def render_figures_tab(paths: dict[str, Path]) -> None:
    st.subheader("Figuras generadas")
    st.markdown(
        "Graficos organizados por etapa del pipeline, listos para usar en el informe. "
        "Cada figura incluye una breve descripcion de su proposito."
    )

    has_any_figure = any(
        (paths["figures"] / filename).exists()
        for group in FIGURE_GROUPS.values()
        for _, filename, _ in group
    )
    if not has_any_figure:
        st.info(f"Aun no hay figuras generadas. Ejecuta `{PIPELINE_COMMAND}`.")
        return

    for group_name, figures in FIGURE_GROUPS.items():
        available = [
            (title, paths["figures"] / filename, caption)
            for title, filename, caption in figures
            if (paths["figures"] / filename).exists()
        ]
        if not available:
            continue

        st.markdown(f"#### {group_name}")
        for index in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (title, path, caption) in zip(cols, available[index : index + 2]):
                with col:
                    st.markdown(f"**{title}**")
                    st.image(str(path), use_container_width=True)
                    st.caption(caption)


def render_predictions_table(prediction_df: pd.DataFrame) -> None:
    st.markdown("#### Predicciones precalculadas")
    st.markdown(
        "Tabla completa con cada registro de prueba, su etiqueta real (si esta disponible), "
        "la prediccion del modelo, la confianza asociada, la zona de decision y las "
        "probabilidades para todas las especies candidatas."
    )

    display_columns = [
        column
        for column in [
            "recording_id",
            "true_species_id",
            "predicted_species_id",
            "confidence",
            "decision_zone",
        ]
        if column in prediction_df.columns
    ]
    display_columns.extend(probability_columns(prediction_df))
    render_dataframe(prediction_df[display_columns], height=430)


def render_test_table(test_df: pd.DataFrame) -> None:
    st.markdown("#### Dataset de prueba")
    st.markdown(
        "Variables originales del conjunto de prueba. Las columnas `mel_0` a `mel_63` "
        "son el vector acustico de 64 coeficientes Mel; `species_id` es la referencia "
        "si existe."
    )
    preview_columns = [
        column
        for column in ["recording_id", "species_id", "songtype_id", "is_tp"]
        if column in test_df.columns
    ]
    mel_columns = sorted(
        [column for column in test_df.columns if column.startswith("mel_")],
        key=lambda column: int(column.split("_")[1]),
    )
    render_dataframe(test_df[preview_columns + mel_columns], height=430)


def render_raw_reports(paths: dict[str, Path]) -> None:
    report_paths = [
        ("Reporte de clasificacion en validacion", paths["classification_metrics"].parent / "best_validation_classification_report.txt"),
        ("Reporte de clasificacion en test", paths["classification_metrics"].parent / "test_classification_report.txt"),
    ]
    available_reports = [(title, path) for title, path in report_paths if path.exists()]
    if not available_reports:
        return

    st.markdown("#### Reportes textuales")
    st.markdown(
        "Reportes detallados con precision, recall y F1 por cada clase. "
        "Permiten identificar que especies se predicen mejor y cuales necesitan "
        "mas datos de entrenamiento."
    )
    for title, path in available_reports:
        with st.expander(title):
            st.text(path.read_text())


def render_metadata_table(metadata: dict[str, Any]) -> None:
    st.markdown("#### Metadata del pipeline")
    st.markdown(
        "Configuracion completa usada por el pipeline para generar las tablas, modelos, "
        "figuras y predicciones. Verifica que el `producer_script` sea el esperado; "
        "si no coincide, los outputs no corresponden a esta aplicacion."
    )
    rows = [
        {"campo": key, "valor": json.dumps(value, ensure_ascii=False)}
        for key, value in metadata.items()
        if key not in {"feature_columns", "best_classifier"}
    ]
    render_dataframe(pd.DataFrame(rows), height=430)


def render_classification_summary(filtered_df: pd.DataFrame) -> None:
    st.markdown("#### Resumen de clasificacion por especie")
    st.markdown(
        "Rendimiento del modelo desglosado por cada especie. Una especie con alta "
        "confianza promedio y muchos aciertos indica que el modelo la reconoce bien; "
        "lo contrario sugiere que requiere mas datos o mejores features."
    )
    if not _has_true_label(filtered_df):
        st.warning("No hay etiquetas reales disponibles para calcular precision por especie.")
        return

    summary = []
    for species_id in sorted(filtered_df["predicted_species_id"].unique()):
        subset = filtered_df[filtered_df["predicted_species_id"] == species_id]
        total = len(subset)
        correct = (subset["true_species_id"] == subset["predicted_species_id"]).sum()
        accuracy = correct / total if total > 0 else 0
        confidence = subset["confidence"].mean()
        summary.append({
            "Especie": species_label(species_id),
            "Registros": total,
            "Aciertos": correct,
            "Precision": format_percentage(accuracy),
            "Confianza promedio": format_percentage(confidence),
        })
    summary_df = pd.DataFrame(summary)
    render_dataframe(summary_df)
    if not summary_df.empty and "Registros" in summary_df.columns:
        st.bar_chart(summary_df.set_index("Especie")["Registros"])


def render_data_tab(
    prediction_df: pd.DataFrame,
    test_df: pd.DataFrame,
    paths: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    st.subheader("Datos usados por la interfaz")
    st.markdown(
        "Explora las tablas generadas por el pipeline: predicciones, rendimiento por "
        "especie, dataset de prueba, reportes textuales y metadata de configuracion."
    )
    table_choice = st.radio(
        "Selecciona una tabla para revisar",
        ["Predicciones", "Resumen por especie", "Dataset de prueba", "Reportes", "Metadata"],
        horizontal=True,
    )

    if table_choice == "Predicciones":
        render_predictions_table(prediction_df)
    elif table_choice == "Resumen por especie":
        render_classification_summary(prediction_df)
    elif table_choice == "Dataset de prueba":
        render_test_table(test_df)
    elif table_choice == "Reportes":
        render_raw_reports(paths)
    else:
        render_metadata_table(metadata)


def render_missing_outputs_page(paths: dict[str, Path]) -> None:
    st.markdown(
        '<div class="section-explanation">',
        unsafe_allow_html=True,
    )
    st.markdown(
        "La aplicacion ya esta configurada para mostrar resultados, pero necesita "
        "que el pipeline genere la carpeta `outputs` con las predicciones, tablas y "
        "figuras. Ejecuta el comando correspondiente y vuelve a cargar esta pagina."
    )
    st.code(PIPELINE_COMMAND, language="bash")
    st.markdown(
        "Para una validacion mas rapida puedes omitir el entrenamiento del MLP "
        "(aunque el informe completo debe incluirlo)."
    )
    st.code(f"{PIPELINE_COMMAND} --skip-mlp", language="bash")


def main() -> None:
    configure_page()

    render_hero()

    output_dir_text = st.sidebar.text_input("Carpeta de outputs", value=str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_text).expanduser()
    paths = get_paths(output_dir)

    if not require_outputs(paths):
        render_missing_outputs_page(paths)
        return

    metadata = load_json(paths["run_metadata"])
    prediction_df = load_csv(paths["predictions"])
    test_df = load_csv(TEST_PATH)

    overview_tab, inference_tab, metrics_tab, figures_tab, data_tab = st.tabs(
        ["Resumen", "Inferencia", "Metricas", "Figuras", "Datos"]
    )

    with overview_tab:
        render_project_overview(paths, prediction_df)

    with inference_tab:
        render_inference_tab(prediction_df, test_df)

    with metrics_tab:
        render_metrics_tab(paths)

    with figures_tab:
        render_figures_tab(paths)

    with data_tab:
        render_data_tab(prediction_df, test_df, paths, metadata)


if __name__ == "__main__":
        main()
