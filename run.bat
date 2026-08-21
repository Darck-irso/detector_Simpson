@echo off
setlocal
cd /d "%~dp0"

title Reconocimiento de personajes de los simpsons

echo ==========================================
echo Reconocimiento de personajes de los simpsons
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: No se encontro el entorno virtual. Por favor, ejecute setup.bat primero.
    echo.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: No se encontro app.py. Por favor, asegurese de que el archivo exista en el directorio actual.
    echo.
    pause
    exit /b 1
)

if not exist "models\modelo_final_simpsons.keras" (
    echo ERROR: falta el clasificador
    echo models\modelo_final_simpsons.keras. Por favor, asegurese de que el archivo exista.
    echo.
    pause
    exit /b 1
)

if not exist "models\yolo_simpsons_best.pt" (
    echo ERROR: falta el modelo de deteccion YOLO
    echo models\yolo_simpsons_best.pt. Por favor, asegurese de que el archivo exista.
    echo.
    pause
    exit /b 1
)

if not exist "models\class_indices_simpsons.json" (
    echo ERROR: falta el archivo de indices de clases
    echo models\class_indices_simpsons.json. Por favor, asegurese de que el archivo exista.
    echo.
    pause
    exit /b 1
)

if not exist "models\pipeline_config.json" (
    echo ERROR: falta el archivo de configuracion del pipeline
    echo models\pipeline_config.json. Por favor, asegurese de que el archivo exista.
    echo.
    pause
    exit /b 1
)

echo Iniciando la aplicacion...
echo.
echo una vez termine gradio de iniciar
echo abra un navegador y copie la url que aparece en la consola
echo.
echo si desea cerrar la consola, presione Ctrl+C
echo.

".venv\Scripts\python.exe" app.py

if errorlevel 1 (
    echo ERROR: La aplicacion encontro un error.
    echo revisa la consola para mas detalles.
    echo.
    pause
    exit /b 1
)

pause