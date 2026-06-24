from __future__ import annotations

from pathlib import Path

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
    # P2_ML.pdf 3.5 - Tres estados operativos segun la probabilidad predictiva.
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
        "evidence": "main.py valida y usa mel_0 a mel_63 como variables de entrada.",
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
        "evidence": "GMM y DBSCAN se comparan con Silhouette, Calinski-Harabasz y Davies-Bouldin.",
    },
    {
        "section": "3.4",
        "item": "MLP, regularizacion y comparacion contra ensambles",
        "status": "Implementado",
        "evidence": "El MLP en NumPy compara variantes con Dropout y BatchNorm, junto a Random Forest y HGB.",
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
            "Verifica el balance de especies antes de entrenar.",
            "Si una especie tiene menos muestras, F1 macro es mas representativo que accuracy.",
        ),
    ],
    "Reduccion dimensional": [
        (
            "PCA 2D",
            "pca_projection.png",
            "Resume el espacio Mel en dos componentes lineales.",
            "Sirve para revisar separacion global y varianza explicada.",
        ),
        (
            "t-SNE 2D",
            "tsne_projection.png",
            "Visualiza vecindarios no lineales del espacio Mel.",
            "Sirve para inspeccionar grupos locales, no para medir varianza.",
        ),
    ],
    "Clustering no supervisado": [
        (
            "GMM clusters",
            "gmm_clusters.png",
            "Muestra la mejor segmentacion probabilistica segun Silhouette.",
            "Clusters compactos y separados sugieren estructura acustica recuperable.",
        ),
        (
            "DBSCAN clusters",
            "dbscan_clusters.png",
            "Muestra agrupaciones por densidad y posibles registros de ruido.",
            "Mucho ruido indica que la densidad no separa bien todas las especies.",
        ),
    ],
    "Clasificacion supervisada": [
        (
            "Curvas de perdida MLP",
            "mlp_loss_curves.png",
            "Compara estabilidad de entrenamiento entre variantes de MLP.",
            "Una curva que baja y se estabiliza sugiere aprendizaje sin oscilacion fuerte.",
        ),
        (
            "F1 macro MLP",
            "mlp_f1_curves.png",
            "Evalua en que epoca y variante mejora el rendimiento por clase.",
            "La variante con mayor F1 macro valida mejor el balance entre especies.",
        ),
        (
            "Matriz de confusion en validacion",
            "best_validation_confusion_matrix.png",
            "Identifica errores por especie en el conjunto de validacion.",
            "Las celdas fuera de la diagonal muestran especies que el modelo confunde.",
        ),
        (
            "Matriz de confusion en test",
            "test_confusion_matrix.png",
            "Resume errores finales sobre datos no usados para entrenar.",
            "Una diagonal dominante indica buena generalizacion del modelo final.",
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
        .small-note {
            color: #555;
            font-size: 0.92rem;
        }
        .context-box {
            border-left: 4px solid #2f6f73;
            background: #f6faf9;
            border-radius: 6px;
            color: #183c40;
            padding: 0.75rem 0.9rem;
            margin: 0.4rem 0 0.8rem 0;
            line-height: 1.45;
        }
        .context-box strong {
            color: #183c40;
        }
        .status-ok {
            color: #0f7b45;
            font-weight: 700;
        }
        .status-pending {
            color: #946200;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_paths(output_dir: Path) -> dict[str, Path]:
    # P2_ML.pdf 3.1 - La interfaz consume las salidas generadas por el pipeline.
    return {
        "predictions": output_dir / "test_predictions.csv",
        "dataset_summary": output_dir / "tables" / "dataset_summary.csv",
        "target_distribution": output_dir / "tables" / "target_distribution.csv",
        "classification_metrics": output_dir / "tables" / "classification_metrics.csv",
        "dimensionality_metrics": output_dir / "tables" / "dimensionality_metrics.csv",
        "clustering_metrics": output_dir / "tables" / "clustering_metrics.csv",
        "test_metrics": output_dir / "tables" / "test_metrics.csv",
        "test_policy_summary": output_dir / "tables" / "test_policy_summary.csv",
        "figures": output_dir / "figures",
    }


def render_hero() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>Dashboard eco-acustico</h1>
            <p>
                Lectura ordenada del pipeline: datos, exploracion, clustering,
                clasificacion, politica de decision e inferencia sobre registros de prueba.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context(purpose: str, interpretation: str) -> None:
    st.markdown(
        f"""
        <div class="context-box">
            <strong>Para que se usa:</strong> {purpose}<br>
            <strong>Como interpretarlo:</strong> {interpretation}
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_outputs(paths: dict[str, Path]) -> bool:
    if paths["predictions"].exists():
        return True

    st.warning("Todavia no hay predicciones generadas en `outputs/test_predictions.csv`.")
    render_context(
        "La interfaz consume archivos ya calculados para evitar reentrenar modelos desde Streamlit.",
        "Ejecuta el pipeline y vuelve a abrir la app; entonces apareceran metricas, figuras y escenarios.",
    )
    st.code("python main.py", language="bash")
    return False


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


def metric_value(df: pd.DataFrame, column: str, default: str = "N/A") -> str:
    if df.empty or column not in df.columns:
        return default
    value = df.iloc[0][column]
    if pd.isna(value):
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_percentage(value: float | int | str) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


def render_dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    options = {
        "use_container_width": True,
        "hide_index": True,
    }
    if height is not None:
        options["height"] = height

    st.dataframe(df, **options)


def decision_box(zone: str, confidence: float) -> None:
    # P2_ML.pdf 3.5 - Visualiza la politica de confianza/incertidumbre/rechazo.
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
    # P2_ML.pdf - Bonus Streamlit: escenarios precargados para simular inferencia.
    st.sidebar.header("Escenarios")
    zone_options = ["todas", *sorted(prediction_df["decision_zone"].dropna().unique())]
    selected_zone = st.sidebar.selectbox("Filtrar por zona", zone_options)

    filtered_df = prediction_df.copy()
    if selected_zone != "todas":
        filtered_df = filtered_df[filtered_df["decision_zone"] == selected_zone]

    if filtered_df.empty:
        st.sidebar.warning("No hay registros para ese filtro.")
        return prediction_df

    zone_counts = filtered_df["decision_zone"].value_counts().to_dict()
    st.sidebar.caption("Distribucion filtrada")
    for zone, count in zone_counts.items():
        st.sidebar.write(f"{ZONE_LABELS.get(zone, zone)}: {count}")
    st.sidebar.caption(f"Registros disponibles: {len(filtered_df)}")
    return filtered_df


def render_project_overview(paths: dict[str, Path], prediction_df: pd.DataFrame | None = None) -> None:
    st.subheader("Estado frente al PDF")
    render_context(
        "Resume que partes del enunciado ya estan cubiertas por el codigo y que queda para el informe.",
        "El pipeline esta cubierto; lo pendiente es agregar la tabla de contribuciones del equipo en el documento escrito.",
    )

    checklist_df = pd.DataFrame(PROJECT_CHECKLIST)
    render_dataframe(checklist_df)

    if prediction_df is not None:
        confidence_mean = prediction_df["confidence"].mean()
        automatic_ratio = (prediction_df["decision_zone"] == "confianza").mean()
        summary_cols = st.columns(3)
        summary_cols[0].metric("Registros en test", f"{len(prediction_df):,}")
        summary_cols[1].metric("Confianza promedio", format_percentage(confidence_mean))
        summary_cols[2].metric("Decision automatica", format_percentage(automatic_ratio))

    missing = [
        label
        for label, path in {
            "resumen del dataset": paths["dataset_summary"],
            "metricas de clasificacion": paths["classification_metrics"],
            "metricas de reduccion": paths["dimensionality_metrics"],
            "metricas de clustering": paths["clustering_metrics"],
        }.items()
        if not path.exists()
    ]
    if missing:
        st.info("Faltan salidas por generar: " + ", ".join(missing) + ".")


def render_inference_tab(prediction_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    # P2_ML.pdf 3.5 - Pantalla principal de inferencia con probabilidades y umbral.
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
    render_context(
        "Permite revisar un registro de prueba y explicar la decision del modelo final.",
        "La prediccion se acepta, revisa o rechaza segun la confianza maxima entre especies.",
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
        render_context(
            "Compara todas las clases candidatas para el registro seleccionado.",
            "Una barra dominante indica decision estable; barras similares indican incertidumbre.",
        )
        st.bar_chart(probability_df.set_index("species")["probability"])
        render_dataframe(probability_df)

    with chart_columns[1]:
        st.markdown("#### Vector Mel del registro")
        render_context(
            "Muestra las 64 variables acusticas que entran al clasificador.",
            "Picos y valles describen la energia por bandas Mel; no son probabilidades.",
        )
        mel_df = selected_mel_table(test_row)
        st.line_chart(mel_df.set_index("mel_feature")["value"])
        render_dataframe(mel_df, height=260)


def render_dataset_tables(paths: dict[str, Path]) -> None:
    st.subheader("Datos y balance")

    if paths["dataset_summary"].exists():
        st.markdown("#### Resumen del dataset")
        render_context(
            "Confirma particiones, variable objetivo, cantidad de features y valores faltantes.",
            "Debe mostrar 64 features cuando se usa el espacio Mel puro pedido por el PDF.",
        )
        render_dataframe(load_csv(paths["dataset_summary"]))

    if paths["target_distribution"].exists():
        st.markdown("#### Distribucion del objetivo")
        render_context(
            "Mide cuantas muestras hay por especie antes de entrenar.",
            "Si las clases estan desbalanceadas, F1 macro debe tener mas peso que accuracy.",
        )
        target_df = load_csv(paths["target_distribution"])
        render_dataframe(target_df)
        if "label" in target_df.columns and "count" in target_df.columns:
            st.bar_chart(target_df.set_index("label")["count"])


def render_metrics_tab(paths: dict[str, Path]) -> None:
    # P2_ML.pdf 3.2-3.4 - Presenta metricas de reduccion, clustering y clasificacion.
    render_dataset_tables(paths)

    if paths["classification_metrics"].exists():
        metrics_df = load_csv(paths["classification_metrics"])
        st.markdown("#### Comparacion de clasificadores")
        render_context(
            "Elige el modelo final comparando MLP y ensambles con calidad y costo computacional.",
            "El mejor candidato es el de mayor F1 macro; los tiempos muestran viabilidad operativa.",
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
        render_context(
            "Verifica el rendimiento final sobre datos de prueba.",
            "F1 macro resume el trato equilibrado entre especies; accuracy resume aciertos globales.",
        )
        render_dataframe(load_csv(paths["test_metrics"]))

    if paths["test_policy_summary"].exists():
        st.markdown("#### Distribucion por zona operativa")
        render_context(
            "Resume cuantas predicciones se aceptan automaticamente, se revisan o se rechazan.",
            "Una proporcion alta en incertidumbre/rechazo indica que el sistema prioriza cautela.",
        )
        policy_df = load_csv(paths["test_policy_summary"])
        if "percentage" in policy_df.columns:
            policy_df["percentage_label"] = policy_df["percentage"].map(format_percentage)
        render_dataframe(policy_df)
        st.bar_chart(policy_df.set_index("decision_zone")["count"])

    if paths["dimensionality_metrics"].exists():
        st.markdown("#### PCA vs t-SNE")
        render_context(
            "Compara reduccion lineal contra no lineal para entender la geometria del espacio Mel.",
            "Mayor trustworthiness conserva mejor vecinos; PCA ademas reporta varianza explicada.",
        )
        dimensionality_df = load_csv(paths["dimensionality_metrics"])
        render_dataframe(dimensionality_df)

    if paths["clustering_metrics"].exists():
        st.markdown("#### GMM vs DBSCAN")
        render_context(
            "Evalua si hay estructura no supervisada antes de usar etiquetas.",
            "Silhouette y Calinski-Harabasz altos son mejores; Davies-Bouldin bajo es mejor.",
        )
        clustering_df = load_csv(paths["clustering_metrics"])
        render_dataframe(clustering_df)


def render_figures_tab(paths: dict[str, Path]) -> None:
    # P2_ML.pdf 3.2-3.4 - Muestra las figuras que se pueden insertar en el informe.
    st.subheader("Figuras generadas")
    render_context(
        "Agrupa los graficos por etapa del pipeline para usarlos directamente en el informe.",
        "Lee primero la finalidad de cada grafico y luego confirma si apoya la conclusion escrita.",
    )

    has_any_figure = any(
        (paths["figures"] / filename).exists()
        for group in FIGURE_GROUPS.values()
        for _, filename, _, _ in group
    )
    if not has_any_figure:
        st.info("Aun no hay figuras generadas. Ejecuta `.venv/bin/python main.py`.")
        return

    for group_name, figures in FIGURE_GROUPS.items():
        available = [
            (title, paths["figures"] / filename, purpose, interpretation)
            for title, filename, purpose, interpretation in figures
            if (paths["figures"] / filename).exists()
        ]
        if not available:
            continue

        st.markdown(f"#### {group_name}")
        for index in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (title, path, purpose, interpretation) in zip(cols, available[index : index + 2]):
                with col:
                    st.markdown(f"##### {title}")
                    render_context(purpose, interpretation)
                    st.image(str(path), use_container_width=True)


def render_predictions_table(prediction_df: pd.DataFrame) -> None:
    st.markdown("#### Predicciones precalculadas")
    render_context(
        "Audita cada registro de prueba con etiqueta real, prediccion, confianza y probabilidades.",
        "Sirve para justificar casos aceptados, casos enviados a revision y posibles errores.",
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
    render_context(
        "Muestra las variables originales usadas por la interfaz y por las predicciones finales.",
        "Las columnas mel_0 a mel_63 son el vector acustico; species_id es la referencia si existe.",
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
    render_context(
        "Complementan las tablas con precision, recall y F1 por clase.",
        "Usalos para explicar que especies se predicen mejor y cuales requieren mas datos.",
    )
    for title, path in available_reports:
        with st.expander(title):
            st.text(path.read_text())


def render_data_tab(prediction_df: pd.DataFrame, test_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    st.subheader("Datos usados por la interfaz")
    table_choice = st.radio(
        "Tabla a revisar",
        ["Predicciones", "Dataset de prueba", "Reportes"],
        horizontal=True,
    )

    if table_choice == "Predicciones":
        render_predictions_table(prediction_df)
    elif table_choice == "Dataset de prueba":
        render_test_table(test_df)
    else:
        render_raw_reports(paths)


def render_missing_outputs_page(paths: dict[str, Path]) -> None:
    render_project_overview(paths)
    st.subheader("Como generar la evidencia")
    st.write(
        "La app ya esta lista para mostrar resultados, pero necesita que el pipeline cree la carpeta `outputs`."
    )
    st.code("python main.py", language="bash")
    st.write(
        "Para una validacion mas rapida puedes omitir el MLP, aunque el informe completo debe incluirlo."
    )
    st.code("python main.py --skip-mlp", language="bash")


def main() -> None:
    configure_page()

    render_hero()

    output_dir_text = st.sidebar.text_input("Carpeta de outputs", value=str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_text).expanduser()
    paths = get_paths(output_dir)

    if not require_outputs(paths):
        render_missing_outputs_page(paths)
        return

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
        render_data_tab(prediction_df, test_df, paths)


if __name__ == "__main__":
    main()
