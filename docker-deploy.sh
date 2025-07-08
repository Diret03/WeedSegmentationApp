#!/bin/bash
# 🚀 Script de construcción y despliegue con Docker

set -e  # Salir si hay errores

echo "🌱 Sistema de Segmentación de Malezas - Docker Build & Deploy"
echo "=============================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Variables
IMAGE_NAME="weed-segmentation"
IMAGE_TAG="latest"
CONTAINER_NAME="weed-app"

# Verificar Docker
if ! command -v docker &> /dev/null; then
    error "Docker no está instalado. Por favor instala Docker primero."
    exit 1
fi

if ! docker info &> /dev/null; then
    error "Docker no está ejecutándose. Por favor inicia Docker."
    exit 1
fi

log "✅ Docker está funcionando correctamente"

# Verificar archivos necesarios
if [ ! -f "Dockerfile" ]; then
    error "Dockerfile no encontrado en el directorio actual"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    error "requirements.txt no encontrado"
    exit 1
fi

if [ ! -f "app.py" ]; then
    error "app.py no encontrado"
    exit 1
fi

log "✅ Archivos necesarios verificados"

# Verificar modelo
if [ ! -f "models/weed_segmenter_fpn_model_085_local.pth" ]; then
    warn "⚠️  Modelo no encontrado en models/weed_segmenter_fpn_model_085_local.pth"
    warn "   Asegúrate de descargar el modelo antes de ejecutar la aplicación"
fi

# Crear directorios necesarios
mkdir -p uploads results models static/dist
log "✅ Directorios creados"

# Función para mostrar ayuda
show_help() {
    echo "Uso: $0 [OPCIÓN]"
    echo ""
    echo "Opciones:"
    echo "  build      Construir la imagen Docker"
    echo "  run        Ejecutar el contenedor"
    echo "  stop       Detener el contenedor"
    echo "  restart    Reiniciar el contenedor"
    echo "  logs       Mostrar logs del contenedor"
    echo "  clean      Limpiar imágenes y contenedores"
    echo "  compose    Usar Docker Compose"
    echo "  help       Mostrar esta ayuda"
    echo ""
}

# Función para construir la imagen
build_image() {
    log "🔨 Construyendo imagen Docker..."
    
    info "Construyendo imagen: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    docker build \
        --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
        --build-arg VERSION="1.0.0" \
        -t ${IMAGE_NAME}:${IMAGE_TAG} \
        -f Dockerfile \
        .
    
    if [ $? -eq 0 ]; then
        log "✅ Imagen construida exitosamente"
        docker images | grep ${IMAGE_NAME}
    else
        error "❌ Error construyendo la imagen"
        exit 1
    fi
}

# Función para ejecutar el contenedor
run_container() {
    log "🚀 Ejecutando contenedor..."
    
    # Detener contenedor existente si está ejecutándose
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        warn "Deteniendo contenedor existente..."
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
    fi
    
    info "Iniciando nuevo contenedor: ${CONTAINER_NAME}"
    
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p 5000:5000 \
        -v $(pwd)/models:/app/models:ro \
        -v $(pwd)/uploads:/app/uploads \
        -v $(pwd)/results:/app/results \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        log "✅ Contenedor iniciado exitosamente"
        info "Aplicación disponible en: http://localhost:5000"
        
        # Esperar a que la aplicación esté lista
        info "Esperando a que la aplicación esté lista..."
        sleep 10
        
        # Verificar salud de la aplicación
        if curl -f http://localhost:5000/ > /dev/null 2>&1; then
            log "✅ Aplicación está respondiendo correctamente"
        else
            warn "⚠️  La aplicación puede estar iniciando. Verifica los logs con: $0 logs"
        fi
    else
        error "❌ Error iniciando el contenedor"
        exit 1
    fi
}

# Función para detener el contenedor
stop_container() {
    log "🛑 Deteniendo contenedor..."
    
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker stop ${CONTAINER_NAME}
        docker rm ${CONTAINER_NAME}
        log "✅ Contenedor detenido y removido"
    else
        info "No hay contenedor ejecutándose con el nombre: ${CONTAINER_NAME}"
    fi
}

# Función para reiniciar
restart_container() {
    log "🔄 Reiniciando contenedor..."
    stop_container
    run_container
}

# Función para mostrar logs
show_logs() {
    log "📋 Mostrando logs del contenedor..."
    
    if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker logs -f ${CONTAINER_NAME}
    else
        error "No hay contenedor ejecutándose con el nombre: ${CONTAINER_NAME}"
        exit 1
    fi
}

# Función para limpiar
clean_docker() {
    log "🧹 Limpiando imágenes y contenedores..."
    
    # Detener y remover contenedor
    stop_container
    
    # Remover imagen
    if docker images -q ${IMAGE_NAME}:${IMAGE_TAG} | grep -q .; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
        log "✅ Imagen removida"
    fi
    
    # Limpiar imágenes dangling
    docker image prune -f
    
    log "✅ Limpieza completada"
}

# Función para usar Docker Compose
use_compose() {
    log "🐙 Usando Docker Compose..."
    
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml no encontrado"
        exit 1
    fi
    
    case "$2" in
        up)
            docker-compose up -d
            log "✅ Servicios iniciados con Docker Compose"
            ;;
        down)
            docker-compose down
            log "✅ Servicios detenidos con Docker Compose"
            ;;
        logs)
            docker-compose logs -f
            ;;
        *)
            info "Uso: $0 compose [up|down|logs]"
            ;;
    esac
}

# Procesamiento de argumentos
case "$1" in
    build)
        build_image
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        restart_container
        ;;
    logs)
        show_logs
        ;;
    clean)
        clean_docker
        ;;
    compose)
        use_compose "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Opción no válida: $1"
        show_help
        exit 1
        ;;
esac

log "🎉 Operación completada exitosamente"
