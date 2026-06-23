# Proyecto 2 - Machine Learning Eco-Acustico

Este proyecto implementa un pipeline para clasificar señales eco-acusticas usando las 64 variables `mel_0` a `mel_63`.
El MLP se implementa con NumPy para incluir Dropout y Batch Normalization sin depender de frameworks pesados.

## Instalacion

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Ejecucion completa

```bash
.venv/bin/python main.py
```

Por defecto se predice `species_id`, que es la variable objetivo descrita en la documentacion del dataset.

## Ejecucion rapida sin MLP

Si solo se desea validar el flujo general sin entrenar redes neuronales:

```bash
.venv/bin/python main.py --skip-mlp
```

Si se usa fish y se quiere activar el entorno manualmente, el comando correcto es:

```fish
source .venv/bin/activate.fish
```

## Frontend en Streamlit

Primero genera los resultados del pipeline:

```bash
.venv/bin/python main.py
```

Luego inicia la interfaz:

```bash
.venv/bin/python -m streamlit run app.py
```

La app abre un simulador de inferencia con escenarios precargados de `eco_acoustic_test.csv`.
No reentrena modelos dentro de la interfaz; lee `outputs/test_predictions.csv` y aplica los umbrales de decision del proyecto.

La interfaz incluye:

- seleccion de registros por `recording_id`;
- prediccion, especie real y confianza;
- zona operativa: `confianza`, `incertidumbre` o `rechazo`;
- grafico de probabilidades por especie;
- grafico del vector `mel_0` a `mel_63`;
- tablas de metricas y figuras generadas por el pipeline.

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
