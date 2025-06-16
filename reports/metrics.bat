@echo off
echo.
echo ========================================
echo   Generando Metricas y Visualizaciones
echo ========================================
echo.


echo [1/3] Metricas de Inferencia
python.exe evaluate_metrics.py 

echo [2/3] UMAP de eventos unknown
python.exe umap_unknown.py

echo [3/3] UMAP todas las clases
python.exe umap_all.py

echo [4/4] UMAP chulo
python.exe umap_multiclass_highlight.py