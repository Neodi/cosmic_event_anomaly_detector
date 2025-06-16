@echo off
echo.
echo ================================
echo   Generando Dataset PLAsTiCC
echo ================================
echo.

echo [1/4] Cargando datos en pkl...
python.exe .\load_clean.py --obs ..\..\data\raw\plasticc_test_set_batch10.csv --meta ..\..\data\raw\plasticc_test_metadata.csv
@REM python.exe .\load_all_data.py ^
@REM     --obs ..\..\data\raw\plasticc_test_set_batch1.csv ^
@REM     ..\..\data\raw\plasticc_test_set_batch2.csv ^
@REM     ..\..\data\raw\plasticc_test_set_batch3.csv ^
@REM     ..\..\data\raw\plasticc_test_set_batch4.csv ^
@REM     ..\..\data\raw\plasticc_test_set_batch5.csv ^
@REM     --meta ..\..\data\raw\plasticc_test_metadata.csv
echo.

if %errorlevel% neq 0 (
    echo ERROR: Fallo al cargar datos
    pause
    exit /b 1
)

echo [2/4] Creando tensores...
python.exe .\fast_create_tensors.py
if %errorlevel% neq 0 (
    echo ERROR: Fallo al crear tensores
    pause
    exit /b 1
)

echo [3/4] Creando estadisticas...
python.exe .\make_stats.py
if %errorlevel% neq 0 (
    echo ERROR: Fallo al crear estadisticas
    pause
    exit /b 1
)

echo [4/4] Separando en train/val/test...
python.exe .\make_splits.py
if %errorlevel% neq 0 (
    echo ERROR: Fallo al separar datos
    pause
    exit /b 1
)

echo.
echo ================================
echo   Dataset generado exitosamente
echo ================================
pause
