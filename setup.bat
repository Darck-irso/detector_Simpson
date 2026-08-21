@echo off
setlocal
cd /d "%~dp0"

title configuracion - reconocimiento de personajes

echo ==========================================
echo configuracion del proyecto
echo ==========================================
echo.

set "PYTHON_CMD="

py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3.12"

if not defined PYTHON_CMD (
    python --version >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD goto python_missing

if not exist "requirements.txt" (
    echo ERROR: No se encontro requirements.txt.
    goto error
)

if exist ".venv\Scripts\python.exe" (
    echo El entorno viertual ya existe.
) else (
    echo Creando entorno virtual...
    %PYTHON_CMD% -m venv .venv

    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        goto error
    )
)

echo.
echo actualizando pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

if errorlevel 1 (
    echo ERROR: No se pudo actualizar pip.
    goto error
)

echo.
echo instalando dependencias...
echo esto puede tardar un momento...
echo.

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias.
    goto error
)

echo.
echo ==========================================
echo configuracion completada con exito
echo ==========================================
echo.
echo Para ejecutar el proyecto, ejecuta run.bat
echo.
pause
exit /b 0

:python_missing
echo.
echo ERROR: No se encontro Python 3.12 en el sistema.
echo Asegurate de tener Python 3.12 instalado y agregado al PATH.
echo.
pause
exit /b 1

:error
echo.
echo la configuracion del proyecto fallo.
echo revisa los mensajes de error anteriores para mas detalles.
echo.
pause
exit /b 1
