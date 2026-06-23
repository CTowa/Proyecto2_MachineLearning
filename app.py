from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Configuracion de la interfaz Streamlit
# ============================================================
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
    "confianza": "Clasificacion automatica: la probabilidad supera el umbral operativo.",
    "incertidumbre": "Clasificacion asistida: el registro debe revisarse por un experto.",
    "rechazo": "Descarte automatico: la confianza es baja y se mitiga el riesgo de falsos positivos.",
}

ZONE_COLORS = {
    "confianza": "#0f7b45",
    "incertidumbre": "#946200",
    "rechazo": "#b42318",
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
        .decision-box {
            border-radius: 8px;
            padding: 1rem;
            color: white;
            font-weight: 600;
        }
        .small-note {
            color: #555;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "predictions": output_dir / "test_predictions.csv",
        "classification_metrics": output_dir / "tables" / "classification_metrics.csv",
        "dimensionality_metrics": output_dir / "tables" / "dimensionality_metrics.csv",
        "clustering_metrics": output_dir / "tables" / "clustering_metrics.csv",
        "test_metrics": output_dir / "tables" / "test_metrics.csv",
        "test_policy_summary": output_dir / "tables" / "test_policy_summary.csv",
        "figures": output_dir / "figures",
    }


def require_outputs(paths: dict[str, Path]) -> bool:
    if paths["predictions"].exists():
        return True

    st.error("No se encontro `outputs/test_predictions.csv`.")
    st.info("Primero ejecuta el pipeline completo o rapido para generar las predicciones.")
    st.code(".venv/bin/python main.py", language="bash")
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
    selected_zone = st.sidebar.selectbox("Filtrar por zona", zone_options)

    filtered_df = prediction_df.copy()
    if selected_zone != "todas":
        filtered_df = filtered_df[filtered_df["decision_zone"] == selected_zone]

    if filtered_df.empty:
        st.sidebar.warning("No hay registros para ese filtro.")
        return prediction_df

    st.sidebar.caption(f"Registros disponibles: {len(filtered_df)}")
    return filtered_df


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
        '<p class="small-note">Selecciona un escenario precargado del conjunto de prueba. '
        "La interfaz muestra la prediccion final del pipeline y aplica los umbrales operativos.</p>",
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Recording ID", selected_recording)
    metric_columns[1].metric("Especie real", species_label(true_species))
    metric_columns[2].metric("Prediccion", species_label(predicted_species))
    metric_columns[3].metric("Confianza", f"{confidence:.2%}")

    decision_box(zone, confidence)

    chart_columns = st.columns([1, 1])
    probability_df = selected_probability_table(prediction_row)
    with chart_columns[0]:
        st.markdown("#### Probabilidades por especie")
        st.bar_chart(probability_df.set_index("species")["probability"])
        st.dataframe(probability_df, use_container_width=True, hide_index=True)

    with chart_columns[1]:
        st.markdown("#### Vector Mel del registro")
        mel_df = selected_mel_table(test_row)
        st.line_chart(mel_df.set_index("mel_feature")["value"])
        st.dataframe(mel_df, use_container_width=True, hide_index=True, height=260)


def render_metrics_tab(paths: dict[str, Path]) -> None:
    st.subheader("Metricas del pipeline")

    if paths["classification_metrics"].exists():
        metrics_df = load_csv(paths["classification_metrics"])
        st.markdown("#### Comparacion de clasificadores")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        best_row = metrics_df.iloc[0]
        st.success(
            f"Mejor modelo: {best_row['family']}/{best_row['model']} "
            f"con F1 macro = {best_row['f1_macro']:.4f}"
        )

    if paths["test_metrics"].exists():
        st.markdown("#### Desempeno en test")
        st.dataframe(load_csv(paths["test_metrics"]), use_container_width=True, hide_index=True)

    if paths["test_policy_summary"].exists():
        st.markdown("#### Distribucion por zona operativa")
        policy_df = load_csv(paths["test_policy_summary"])
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
        st.bar_chart(policy_df.set_index("decision_zone")["count"])

    if paths["dimensionality_metrics"].exists():
        st.markdown("#### PCA vs t-SNE")
        st.dataframe(load_csv(paths["dimensionality_metrics"]), use_container_width=True, hide_index=True)

    if paths["clustering_metrics"].exists():
        st.markdown("#### GMM vs DBSCAN")
        st.dataframe(load_csv(paths["clustering_metrics"]), use_container_width=True, hide_index=True)


def render_figures_tab(paths: dict[str, Path]) -> None:
    st.subheader("Figuras generadas")
    figures = [
        ("Distribucion de clases", "target_distribution.png"),
        ("PCA 2D", "pca_projection.png"),
        ("t-SNE 2D", "tsne_projection.png"),
        ("GMM clusters", "gmm_clusters.png"),
        ("DBSCAN clusters", "dbscan_clusters.png"),
        ("Curvas de perdida MLP", "mlp_loss_curves.png"),
        ("F1 macro MLP", "mlp_f1_curves.png"),
        ("Matriz de confusion en validacion", "best_validation_confusion_matrix.png"),
        ("Matriz de confusion en test", "test_confusion_matrix.png"),
    ]

    available = [(title, paths["figures"] / filename) for title, filename in figures if (paths["figures"] / filename).exists()]
    if not available:
        st.info("Aun no hay figuras generadas. Ejecuta `.venv/bin/python main.py`.")
        return

    for index in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, available[index : index + 2]):
            with col:
                st.markdown(f"#### {title}")
                st.image(str(path), use_container_width=True)


def render_data_tab(prediction_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    st.subheader("Datos usados por la interfaz")
    st.markdown("#### Predicciones precalculadas")
    st.dataframe(prediction_df, use_container_width=True, hide_index=True)

    st.markdown("#### Dataset de prueba")
    st.dataframe(test_df, use_container_width=True, hide_index=True)


def main() -> None:
    configure_page()

    st.title("Inferencia Eco-Acustica")
    st.caption("Simulador de decisiones para clasificacion de especies con escenarios precargados.")

    output_dir_text = st.sidebar.text_input("Carpeta de outputs", value=str(DEFAULT_OUTPUT_DIR))
    output_dir = Path(output_dir_text).expanduser()
    paths = get_paths(output_dir)

    if not require_outputs(paths):
        return

    prediction_df = load_csv(paths["predictions"])
    test_df = load_csv(TEST_PATH)

    inference_tab, metrics_tab, figures_tab, data_tab = st.tabs(
        ["Inferencia", "Metricas", "Figuras", "Datos"]
    )

    with inference_tab:
        render_inference_tab(prediction_df, test_df)

    with metrics_tab:
        render_metrics_tab(paths)

    with figures_tab:
        render_figures_tab(paths)

    with data_tab:
        render_data_tab(prediction_df, test_df)


if __name__ == "__main__":
    main()
