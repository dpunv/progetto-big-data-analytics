@echo off
REM filepath: c:\Users\Hp\Desktop\appunti\big data\BigDataProg\progetto\progetto-big-data-analytics\start_cluster.bat

setlocal enabledelayedexpansion

REM Verifica argomento input
if "%1"=="" (
    echo Usage: start_cluster.bat ^<number_of_nodes^>
    echo Example: start_cluster.bat 5
    exit /b 1
)

set NUM_NODES=%1

REM Valida che sia un numero
for /f %%A in ('powershell -Command "[int]::TryParse('%NUM_NODES%', [ref]$null); $?"') do (
    if not "%%A"=="True" (
        echo Error: '%NUM_NODES%' is not a valid number
        exit /b 1
    )
)

REM Verifica che sia >= 1
if %NUM_NODES% LSS 1 (
    echo Error: Number of nodes must be at least 1
    exit /b 1
)

echo.
echo ==========================================
echo Starting Qdrant cluster with %NUM_NODES% nodes
echo ==========================================
echo.

REM Crea directory per i dati dei nodi se non esiste
if not exist "qdrant_nodes" mkdir qdrant_nodes

REM Avvia i nodi in background
for /L %%i in (1,1,%NUM_NODES%) do (
    set PORT=6333
    set /A OFFSET=%%i-1
    set /A PORT=!PORT!+!OFFSET!
    
    echo Starting node %%i on port !PORT!...
    
    start "Qdrant Node %%i" cmd /k "cd qdrant_nodes ^& qdrant --storage-path node_%%i --port !PORT!"
    
    timeout /t 2 /nobreak
)

echo.
echo ✓ Cluster with %NUM_NODES% nodes started
echo.
timeout /t 5 /nobreak

echo.
echo ==========================================
echo Running main.py
echo ==========================================
echo.

python main.py %NUM_NODES%

pause