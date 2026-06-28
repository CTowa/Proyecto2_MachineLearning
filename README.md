# Proyecto 2 - Machine Learning Eco-Acustico

Este proyecto implementa un pipeline para clasificar señales eco-acusticas usando las 64 variables `mel_0` a `mel_63`.
El MLP se implementa con NumPy para incluir Dropout y Batch Normalization sin depender de frameworks pesados.

## Relacion con las indicaciones del PDF

Esta tabla conecta lo pedido en `P2_ML.pdf` con la parte del proyecto que lo implementa.

| Seccion del PDF | Que pide | Donde se implementa | Salida util |
| --- | --- | --- | --- |
| 3.1 Resumen e introduccion al espacio vectorial | Definir el problema, cargar CSV y trabajar con `X in R64` | `main.py`: `load_data`, `get_mel_columns`, `get_feature_columns`, `save_dataset_summary` | `outputs/tables/dataset_summary.csv`, `outputs/figures/target_distribution.png` |
| 3.1 Arquitectura del pipeline | Mostrar flujo desde CSV hasta inferencia/Streamlit | `main.py`: `main`; `app.py`: interfaz de inferencia | README + app Streamlit |
| 3.2 Reduccion dimensional | Comparar PCA vs metodo no lineal | `main.py`: `run_dimensionality_analysis` | `outputs/tables/dimensionality_metrics.csv`, `outputs/figures/pca_projection.png`, `outputs/figures/tsne_projection.png` |
| 3.2 Reporte cuantitativo | Medir tiempos y preservacion geometrica | `main.py`: columnas `seconds`, `explained_variance_ratio`, `trustworthiness_10nn` | `outputs/tables/dimensionality_metrics.csv` |
| 3.3 Clustering | Comparar dos metodos no supervisados distintos | `main.py`: `run_clustering_analysis` con GMM y DBSCAN | `outputs/tables/clustering_metrics.csv`, `outputs/figures/gmm_clusters.png`, `outputs/figures/dbscan_clusters.png` |
| 3.3 Validacion interna | Usar metricas como Silhouette | `main.py`: `score_clustering` | `outputs/tables/clustering_metrics.csv` |
| 3.4 MLP | Definir topologia, loss y regularizacion | `main.py`: `NumpyMLPClassifier` | `outputs/tables/mlp_history_*.csv`, `outputs/figures/mlp_loss_curves.png` |
| 3.4 Dropout y BatchNorm | Comparar posicion de Dropout y Batch Normalization | `main.py`: variantes `plain`, `dropout_then_batchnorm`, `batchnorm_then_dropout` | `outputs/figures/mlp_loss_curves.png`, `outputs/figures/mlp_f1_curves.png` |
| 3.4 MLP vs ensambles | Comparar con F1 y matrices de confusion | `main.py`: `evaluate_mlp_models`, `evaluate_ensemble_models`, `save_confusion_matrix` | `outputs/tables/classification_metrics.csv`, `outputs/figures/best_validation_confusion_matrix.png` |
| 3.5 MLOps y negocio | Medir costo computacional e inferencia | `main.py`: `fit_seconds`, `predict_ms_per_sample` | `outputs/tables/classification_metrics.csv` |
| 3.5 Umbrales operativos | Aplicar confianza, incertidumbre y rechazo | `main.py`: `confidence_zone`, `build_prediction_table`; `app.py`: `decision_box` | `outputs/test_predictions.csv`, Streamlit |
| Bonus Streamlit | Interfaz informativa con escenarios precargados | `app.py` | `http://localhost:8501` |
| 3.6 Contribution statement | Tabla de coevaluacion del equipo | No corresponde al codigo; debe agregarse en el informe LaTeX | Tabla final del informe |

En `main.py` y `app.py` tambien hay comentarios con referencias como `P2_ML.pdf 3.2` para ubicar rapidamente que parte del codigo responde a cada indicacion.

## Cierre para el informe

El codigo queda enfocado en PCA y t-SNE. No se incluye UMAP porque el proyecto ya contrasta un metodo lineal contra un metodo no lineal de reduccion dimensional, y asi se mantiene el alcance alineado con la implementacion documentada.

Puntos que deben explicarse en el informe:

- La variable objetivo principal es `species_id`, porque representa la especie a clasificar.
- Las variables de entrada principales son solo `mel_0` a `mel_63`, es decir, `X in R64`.
- `songtype_id` e `is_tp` se tratan como metadatos; no hacen parte del experimento principal para no alejarse del espacio Mel puro.
- PCA se usa como reduccion lineal e interpretable por varianza explicada.
- t-SNE se usa como visualizacion no lineal de vecindarios; no reporta varianza explicada.
- GMM y DBSCAN se usan para observar estructura no supervisada antes de la clasificacion.
- El MLP se compara contra ensambles usando `f1_macro`, porque las clases no estan perfectamente balanceadas.
- Las zonas `confianza`, `incertidumbre` y `rechazo` convierten probabilidades en una politica operativa.

Puntos que no deben venderse como resultado principal:

- `--include-metadata` es solo una opcion exploratoria; los resultados principales deben usar `mel_0` a `mel_63`.
- Streamlit es una capa de demostracion y explicacion, no un nuevo entrenamiento.
- Los ensambles son benchmark contra el MLP; no reemplazan la explicacion de la red neuronal.
- El modelo no debe presentarse como sistema productivo robusto: el rendimiento en test es moderado y debe reportarse con sus limitaciones.

Con esto, el codigo queda listo para construir el informe. Lo unico externo al codigo que falta completar es el `Contribution statement` o tabla de coevaluacion del equipo.

## Instalacion

### Windows con Git Bash

Si ya tienes `.venv` activado y quieres recrearlo, primero sal del entorno. No ejecutes `python -m venv .venv` mientras `.venv` esta activo porque Windows puede bloquear `.venv/Scripts/python.exe`.

```bash
deactivate
rm -rf .venv
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows con PowerShell

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecucion completa

```bash
python main.py
```

Por defecto se predice `species_id`, que es la variable objetivo descrita en la documentacion del dataset.

## Ejecucion rapida sin MLP

Si solo se desea validar el flujo general sin entrenar redes neuronales:

```bash
python main.py --skip-mlp
```

Si se usa fish y se quiere activar el entorno manualmente, el comando correcto es:

```fish
source .venv/bin/activate.fish
```

## Frontend en Streamlit

Primero genera los resultados del pipeline:

```bash
python main.py
```

Luego inicia la interfaz:

```bash
python -m streamlit run app.py
```

La app abre un simulador de inferencia con escenarios precargados de `eco_acoustic_test.csv`.
No reentrena modelos dentro de la interfaz; lee `outputs/test_predictions.csv` y aplica los umbrales de decision del proyecto.

La interfaz incluye:

- resumen de cumplimiento frente a `P2_ML.pdf`;
- seleccion de registros por `recording_id`;
- prediccion, especie real y confianza;
- zona operativa: `confianza`, `incertidumbre` o `rechazo`;
- grafico de probabilidades por especie;
- grafico del vector `mel_0` a `mel_63`;
- tablas de metricas con explicacion de uso e interpretacion;
- figuras ordenadas por etapa del pipeline: dataset, reduccion, clustering y clasificacion.

## Resultados generados

Los resultados se guardan en `outputs/`:

- `outputs/tables/dimensionality_metrics.csv`: tiempos y metricas de PCA vs t-SNE.
- `outputs/tables/clustering_metrics.csv`: comparacion de GMM y DBSCAN.
- `outputs/tables/classification_metrics.csv`: comparacion de MLP y ensambles.
- `outputs/figures/`: graficos para el informe.
- `outputs/test_predictions.csv`: predicciones con probabilidad y zona de decision.
- `outputs/models/`: mejor modelo entrenado y metadatos.

## Como interpretar la salida

### Resumen del dataset

Esta parte confirma que se cargaron correctamente los CSV:

- `rows`: numero de registros.
- `columns`: numero total de columnas.
- `target`: variable que se esta prediciendo. Por defecto es `species_id`.
- `features_used`: variables usadas como entrada del modelo. Debe salir `64` si se usan solo `mel_0` a `mel_63`.
- `missing_values`: valores faltantes. En este dataset sale `0`.

La tabla de distribucion del target muestra cuantas muestras hay por especie. Como las clases no tienen exactamente la misma cantidad, se usa `f1_macro` para comparar modelos de forma mas justa.

### Reduccion dimensional

Compara PCA contra t-SNE:

- `PCA`: metodo lineal. `explained_variance_ratio = 0.670951` significa que las dos primeras componentes conservan aproximadamente 67.1% de la varianza global.
- `t-SNE`: metodo no lineal para visualizacion. Es normal que `explained_variance_ratio` salga como `NaN`, porque t-SNE no reporta varianza explicada como PCA.
- `trustworthiness_10nn`: mide que tan bien se conservan vecinos locales. Mas alto es mejor.
- `seconds`: tiempo de ejecucion.

En tu corrida, t-SNE tuvo mayor `trustworthiness`, lo cual indica que preservo mejor la estructura local para visualizar grupos.

### Clustering no supervisado

Compara GMM contra DBSCAN:

- `GMM`: modelo probabilistico; se prueba con varios `n_components`.
- `DBSCAN`: modelo por densidad; detecta clusters y tambien puntos de ruido.
- `silhouette`: mas alto es mejor. Valores cercanos a 1 indican clusters mas separados.
- `calinski_harabasz`: mas alto suele ser mejor.
- `davies_bouldin`: mas bajo suele ser mejor.
- `noise_ratio`: proporcion de puntos marcados como ruido por DBSCAN.
- `bic`: criterio usado en GMM; mas bajo suele indicar mejor balance entre ajuste y complejidad.

Es normal que DBSCAN tenga `bic = NaN`, porque BIC aplica a modelos probabilisticos como GMM. Tambien es normal que algunas filas tengan `silhouette = NaN` cuando el algoritmo encontro solo un cluster valido.

### Clasificacion supervisada

Esta es la parte mas importante para elegir el modelo final:

- `accuracy`: porcentaje total de aciertos.
- `f1_macro`: promedio del F1 por clase, tratando todas las especies con la misma importancia.
- `f1_weighted`: F1 ponderado por cantidad de muestras por clase.
- `fit_seconds`: tiempo de entrenamiento.
- `predict_ms_per_sample`: tiempo promedio de inferencia por registro.
- `best_epoch`: epoca donde el MLP logro su mejor F1 macro. Sale `NaN` en modelos que no son MLP.

En tu ejecucion, el mejor modelo fue:

```text
mlp/dropout_then_batchnorm
```

Eso esta bien: significa que la red con Dropout antes de Batch Normalization tuvo el mejor `f1_macro` en validacion.

### Archivos mas utiles para el informe

- `outputs/tables/classification_metrics.csv`: tabla para comparar MLP vs ensambles.
- `outputs/figures/best_validation_confusion_matrix.png`: matriz de confusion del mejor modelo.
- `outputs/figures/mlp_loss_curves.png`: curvas de perdida para analizar regularizacion.
- `outputs/figures/mlp_f1_curves.png`: evolucion del F1 en MLP.
- `outputs/tables/dimensionality_metrics.csv`: tabla PCA vs t-SNE.
- `outputs/tables/clustering_metrics.csv`: tabla GMM vs DBSCAN.
- `outputs/test_predictions.csv`: predicciones finales y zonas de confianza.

## Umbrales de decision

El codigo separa las predicciones en tres zonas:

- `confianza`: probabilidad mayor o igual a 0.85.
- `incertidumbre`: probabilidad entre 0.40 y 0.85.
- `rechazo`: probabilidad menor a 0.40.
