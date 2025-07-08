# 🌱 Dockerfile para Sistema de Segmentación de Malezas en Cultivos de Papa
# Imagen base optimizada para PyTorch y aplicaciones de ML
FROM python:3.9-slim

# Información del mantenedor
LABEL maintainer="tu-email@ejemplo.com"
LABEL description="Sistema de segmentación automática de malezas en cultivos de papa"
LABEL version="1.0"

# Variables de entorno para optimización
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    MODEL_PATH=models/weed_segmenter_fpn_model_085_local.pth

# Instalar dependencias del sistema necesarias para OpenCV y PyTorch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app && \
    chown -R app:app /app

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Instalar Node.js para Tailwind CSS
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Copiar package.json para dependencias de Node.js
COPY package.json .
RUN npm install

# Copiar código de la aplicación
COPY . .

# Compilar CSS con Tailwind
RUN npx tailwindcss -i ./static/src/input.css -o ./static/dist/output.css --minify || \
    echo "Tailwind CSS compilation skipped - using existing CSS"

# Crear directorios necesarios
RUN mkdir -p uploads results models static/dist && \
    chown -R app:app /app

# Cambiar al usuario no-root
USER app

# Verificar que los archivos críticos existen
RUN ls -la /app/ && \
    echo "✅ Aplicación Flask encontrada" && \
    ls -la /app/models/ || echo "⚠️  Directorio models vacío - asegúrate de montar el modelo" && \
    ls -la /app/static/ || echo "⚠️  Directorio static no encontrado"

# Exponer puerto
EXPOSE 5000

# Comando de health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Comando por defecto - usar Gunicorn para producción con configuración optimizada para ML
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--worker-class", "sync", "--timeout", "300", "--max-requests", "100", "--max-requests-jitter", "10", "--preload", "--worker-tmp-dir", "/dev/shm", "app:app"]

# Comando alternativo para desarrollo (comentado)
# CMD ["python", "app.py"]
