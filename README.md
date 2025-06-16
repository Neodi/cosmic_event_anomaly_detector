# PLAsTiCC anomaly Detector

Este proyecto entrena y evalua un pipeline de detección de eventos astronómicos atípicos en las curvas de luz de la competición **PLAsTiCC** de Kaggle. El núcleo del sistema es un Variational Autoencoder (VAE) que extrae representaciones latentes de forma no supervisada. Sobre dicho espacio latente se entrena un clasificador de anomalías (Isolation Forest) para identificar transientes desconocidos.

## Instalación del entorno

1. Instalar [Conda](https://docs.conda.io/en/latest/) y crear el entorno con:
   ```bash
   conda env create -f env/environment.yml
   conda activate plasticc
   pip install -r env/requirements.txt
   ```
2. Verificar que PyTorch reconoce la GPU ejecutando:
   ```bash
   python env/verificar_torch_cuda.py
   ```

## Preparación de los datos
Colocar los archivos originales de Kaggle en `data/raw`. Los siguientes scripts generan el dataset procesado:

1. `scripts/data/fast_create_tensors.py` – convierte las curvas de luz en archivos `.npz`.
2. `scripts/data/make_splits.py` – crea los splits de `train`, `val` y `test`.
3. `scripts/data/make_stats.py` – calcula medias y desviaciones para normalizar.

Los archivos resultantes se guardan en `data/processed`.

## Entrenamiento del VAE

El modelo principal se encuentra en `scripts/train/model_vae.py`. Para entrenar:
```bash
python scripts/train/train.py --latent-dim 64 --batch-size 256 --max-epochs 200
```
Los checkpoints se guardarán en `scripts/train/logs/`.

## Inferencia y evaluación

Para generar las tablas de inferencia y entrenar el clasificador de anomalías existen varios scripts en `scripts/discriminator` y `scripts/train`. El flujo completo puede ejecutarse en Windows con `scripts/run_inference.bat`. Las métricas y visualizaciones (ROC, PR y UMAP) se generan con `reports/metrics.bat`.

## Estructura del proyecto

```text
data/
├── raw/               # Datos originales descargados de Kaggle
├── processed/         # Tensores normalizados y splits
env/                   # Definición del entorno y verificación de CUDA
models/                # Pesos entrenados y parámetros de umbral
notebooks/             # Estudios y experimentos interactivos
reports/
│   ├── figures/       # ROC, PR y proyecciones UMAP
│   └── *.py, *.bat    # Scripts para generar métricas
scripts/
├── data/              # Conversión de CSV y creación de stats
├── train/             # Arquitectura del VAE y entrenamiento
├── discriminator/     # Fases de inferencia y fusión de resultados
├── models/            # Bloques de red reutilizables
└── tests/             # Conjunto de pruebas unitarias
```

En conjunto, estos directorios permiten preparar los datos, entrenar el VAE y evaluar la detección de anomalías en PLAsTiCC.
