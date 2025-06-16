@echo off
echo.
echo =================================
echo   Generando Tabla de Inferencia
echo =================================
echo.

set CHECKPOINT=C:\Users\David\___utad\4_Cuarto\TFG\proyecto_PLAsTiCC\scripts\train\logs\vae_experiment\version_18\checkpoints\vae-epoch=83.ckpt

echo [1/4] Generar chi2 y sus Percentiles
cd discriminator
python.exe compute_chi2_metrics.py test --checkpoint %CHECKPOINT%
python.exe compute_chi2_metrics.py val --checkpoint %CHECKPOINT%
python.exe compute_chi2_metrics.py train --checkpoint %CHECKPOINT%
cd ..

echo [2/4] Extraer matrices latentes
cd discriminator
echo Extrayendo matriz TRAIN
python.exe extract_latent_matrix_train.py train --checkpoint %CHECKPOINT%
echo Extrayendo matriz VAL
python.exe extract_latent_matrix_train.py val --checkpoint %CHECKPOINT%
echo Extrayendo matriz TEST
python.exe extract_latent_matrix_train.py test --checkpoint %CHECKPOINT%
cd ..

echo [3/4] Entrenar Isolation Forest y obtener Percentiles
cd train
python.exe .\train_score_isolation_forest.py
cd ..

echo [4/4] Juntar resultados y marcar anomalías
cd discriminator
python.exe merge_inference_scores.py --split val
python.exe merge_inference_scores.py --split test
cd ..