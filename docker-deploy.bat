@echo off
REM 🚀 Script de construcción y despliegue con Docker para Windows

setlocal enabledelayedexpansion

echo 🌱 Sistema de Segmentación de Malezas - Docker Build ^& Deploy
echo ==============================================================

set IMAGE_NAME=weed-segmentation
set IMAGE_TAG=latest
set CONTAINER_NAME=weed-app

REM Verificar Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker no está instalado. Por favor instala Docker Desktop primero.
    exit /b 1
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker no está ejecutándose. Por favor inicia Docker Desktop.
    exit /b 1
)

echo [INFO] ✅ Docker está funcionando correctamente

REM Verificar archivos necesarios
if not exist "Dockerfile" (
    echo [ERROR] Dockerfile no encontrado en el directorio actual
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt no encontrado
    exit /b 1
)

if not exist "app.py" (
    echo [ERROR] app.py no encontrado
    exit /b 1
)

echo [INFO] ✅ Archivos necesarios verificados

REM Verificar modelo
if not exist "models\weed_segmenter_fpn_model_085_local.pth" (
    echo [WARN] ⚠️  Modelo no encontrado en models\weed_segmenter_fpn_model_085_local.pth
    echo [WARN]    Asegúrate de descargar el modelo antes de ejecutar la aplicación
)

REM Crear directorios necesarios
if not exist "uploads" mkdir uploads
if not exist "results" mkdir results
if not exist "models" mkdir models
if not exist "static\dist" mkdir static\dist
echo [INFO] ✅ Directorios creados

REM Función para mostrar ayuda
if "%1"=="help" goto :show_help
if "%1"=="--help" goto :show_help
if "%1"=="-h" goto :show_help

if "%1"=="build" goto :build_image
if "%1"=="run" goto :run_container
if "%1"=="stop" goto :stop_container
if "%1"=="restart" goto :restart_container
if "%1"=="logs" goto :show_logs
if "%1"=="clean" goto :clean_docker
if "%1"=="compose" goto :use_compose

echo Opción no válida: %1
goto :show_help

:show_help
echo.
echo Uso: %0 [OPCIÓN]
echo.
echo Opciones:
echo   build      Construir la imagen Docker
echo   run        Ejecutar el contenedor
echo   stop       Detener el contenedor
echo   restart    Reiniciar el contenedor
echo   logs       Mostrar logs del contenedor
echo   clean      Limpiar imágenes y contenedores
echo   compose    Usar Docker Compose
echo   help       Mostrar esta ayuda
echo.
goto :end

:build_image
echo [INFO] 🔨 Construyendo imagen Docker...
echo [INFO] Construyendo imagen: %IMAGE_NAME%:%IMAGE_TAG%

docker build -t %IMAGE_NAME%:%IMAGE_TAG% -f Dockerfile .

if %errorlevel% equ 0 (
    echo [INFO] ✅ Imagen construida exitosamente
    docker images | findstr %IMAGE_NAME%
) else (
    echo [ERROR] ❌ Error construyendo la imagen
    exit /b 1
)
goto :end

:run_container
echo [INFO] 🚀 Ejecutando contenedor...

REM Detener contenedor existente si está ejecutándose
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARN] Deteniendo contenedor existente...
    docker stop %CONTAINER_NAME%
    docker rm %CONTAINER_NAME%
)

echo [INFO] Iniciando nuevo contenedor: %CONTAINER_NAME%

docker run -d --name %CONTAINER_NAME% -p 5000:5000 -v "%cd%\models:/app/models:ro" -v "%cd%\uploads:/app/uploads" -v "%cd%\results:/app/results" --restart unless-stopped %IMAGE_NAME%:%IMAGE_TAG%

if %errorlevel% equ 0 (
    echo [INFO] ✅ Contenedor iniciado exitosamente
    echo [INFO] Aplicación disponible en: http://localhost:5000
    
    echo [INFO] Esperando a que la aplicación esté lista...
    timeout /t 10 /nobreak >nul
    
    REM Verificar salud de la aplicación
    curl -f http://localhost:5000/ >nul 2>&1
    if %errorlevel% equ 0 (
        echo [INFO] ✅ Aplicación está respondiendo correctamente
    ) else (
        echo [WARN] ⚠️  La aplicación puede estar iniciando. Verifica los logs con: %0 logs
    )
) else (
    echo [ERROR] ❌ Error iniciando el contenedor
    exit /b 1
)
goto :end

:stop_container
echo [INFO] 🛑 Deteniendo contenedor...

docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    docker stop %CONTAINER_NAME%
    docker rm %CONTAINER_NAME%
    echo [INFO] ✅ Contenedor detenido y removido
) else (
    echo [INFO] No hay contenedor ejecutándose con el nombre: %CONTAINER_NAME%
)
goto :end

:restart_container
echo [INFO] 🔄 Reiniciando contenedor...
call :stop_container
call :run_container
goto :end

:show_logs
echo [INFO] 📋 Mostrando logs del contenedor...

docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if %errorlevel% equ 0 (
    docker logs -f %CONTAINER_NAME%
) else (
    echo [ERROR] No hay contenedor ejecutándose con el nombre: %CONTAINER_NAME%
    exit /b 1
)
goto :end

:clean_docker
echo [INFO] 🧹 Limpiando imágenes y contenedores...

REM Detener y remover contenedor
call :stop_container

REM Remover imagen
docker images -q %IMAGE_NAME%:%IMAGE_TAG% >nul 2>&1
if %errorlevel% equ 0 (
    docker rmi %IMAGE_NAME%:%IMAGE_TAG%
    echo [INFO] ✅ Imagen removida
)

REM Limpiar imágenes dangling
docker image prune -f

echo [INFO] ✅ Limpieza completada
goto :end

:use_compose
echo [INFO] 🐙 Usando Docker Compose...

if not exist "docker-compose.yml" (
    echo [ERROR] docker-compose.yml no encontrado
    exit /b 1
)

if "%2"=="up" (
    docker-compose up -d
    echo [INFO] ✅ Servicios iniciados con Docker Compose
) else if "%2"=="down" (
    docker-compose down
    echo [INFO] ✅ Servicios detenidos con Docker Compose
) else if "%2"=="logs" (
    docker-compose logs -f
) else (
    echo [INFO] Uso: %0 compose [up^|down^|logs]
)
goto :end

:end
echo [INFO] 🎉 Operación completada exitosamente
endlocal
