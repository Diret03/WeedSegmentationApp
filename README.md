# 🌱 Sistema de Segmentación Automática de Malezas en Cultivos de Papa

## 📋 Descripción

Sistema de inteligencia artificial especializado en la identificación y segmentación automática de malezas en cultivos agrícolas de papa. Utiliza tecnología de aprendizaje profundo con arquitectura Feature Pyramid Network (FPN) para detectar y clasificar diferentes tipos de malezas con alta precisión.

![Captura de pantalla de la aplicación](screenshots/app-preview.png)

## 🎯 Características Principales

- **Detección Automática Multi-Clase**: Identifica 6 clases diferentes incluyendo papa y 4 tipos de malezas
- **Interfaz Web Intuitiva**: Aplicación Flask con diseño moderno usando Tailwind CSS
- **Procesamiento en Tiempo Real**: Análisis rápido de imágenes con visualización instantánea
- **Animación de Resultados**: Visualización progresiva de la segmentación con efectos animados
- **Estadísticas Detalladas**: Análisis cuantitativo de la distribución de clases
- **Descarga de Resultados**: Exportación de imágenes procesadas

## 🔬 Clases de Detección

El sistema puede identificar las siguientes clases:

| Clase | Descripción | Color de Visualización |
|-------|-------------|----------------------|
| 🟫 **Fondo** | Background/Suelo | Negro |
| 🔴 **Lengua de Vaca** | Maleza común en cultivos | Rojo |
| 🟡 **Diente de León** | Maleza perenne | Naranja/Amarillo |
| 🟣 **Kikuyo** | Pasto invasivo | Púrpura |
| 🩷 **Otras Malezas** | Otras especies de malezas | Rosa/Magenta |
| 🟢 **Papa** | Cultivo principal | Verde esmeralda |

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Node.js (para compilación de CSS con Tailwind)
- GPU compatible con CUDA (recomendado para mejor rendimiento)

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/WeedSegmentationApp.git
cd WeedSegmentationApp
```

### 2. Configurar entorno virtual de Python

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias de Python

```bash
pip install -r requirements.txt
```

### 4. Configurar Tailwind CSS

```bash
# Instalar dependencias de Node.js
npm install

# Compilar CSS (si es necesario)
npx tailwindcss -i ./static/src/input.css -o ./static/dist/output.css --watch
```

### 5. Descargar el modelo entrenado

Coloca el modelo pre-entrenado en la carpeta `models/`:
```
models/
└── weed_segmenter_fpn_model_085_local.pth
```


## 🏃‍♂️ Ejecución


```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`


## 📁 Estructura del Proyecto

```
WeedSegmentationApp/
├── app.py                      # Aplicación principal Flask
├── weed_predictor.py          # Lógica de predicción y modelo
├── requirements.txt           # Dependencias Python
├── package.json              # Dependencias Node.js
├── README.md                 # Documentación
├── models/                   # Modelos entrenados
│   └── weed_segmenter_fpn_model_085_local.pth
├── static/                   # Archivos estáticos
│   ├── style.css
│   ├── potato.svg
│   ├── src/
│   └── dist/
├── templates/                # Templates HTML
│   └── index.html
├── uploads/                  # Imágenes subidas (temporal)
├── results/                  # Resultados procesados
└── __pycache__/             # Archivos Python compilados
```

## 🧠 Arquitectura del Modelo

### Feature Pyramid Network (FPN)

El sistema utiliza una arquitectura FPN personalizada con las siguientes características:

- **Backbone**: EfficientNet o ResNet como extractor de características
- **Módulos de Atención**: Channel Attention y Spatial Attention
- **Decoder Piramidal**: Múltiples escalas de resolución
- **6 Clases de Salida**: Segmentación semántica multi-clase

### Componentes Técnicos

1. **Attention Mechanisms**:
   - Channel Attention Module (CAM)
   - Spatial Attention Module (SAM)

2. **Multi-Scale Feature Fusion**:
   - Pyramid levels: P2, P3, P4, P5
   - Feature upsampling y lateral connections

3. **Decoder**:
   - Progressive feature refinement
   - Skip connections para preservar detalles


## 🎨 Interfaz de Usuario

### Tecnologías Frontend

- **HTML5**: Estructura semántica
- **Tailwind CSS**: Framework de utilidades CSS
- **JavaScript Vanilla**: Interactividad sin dependencias
- **Font Awesome**: Iconografía

## 📊 Métricas y Estadísticas

El sistema proporciona:

- **Porcentaje por clase**: Distribución de píxeles por categoría
- **Conteo de malezas**: Número de tipos de malezas detectadas
- **Área de cultivo**: Porcentaje de plantas de papa
- **Tiempo de procesamiento**: Duración del análisis


## 🚀 Despliegue

### 🐳 Docker

El proyecto incluye una configuración completa de Docker optimizada para aplicaciones de ML.

#### Archivos Docker incluidos:
- `Dockerfile` - Imagen optimizada para PyTorch y OpenCV
- `docker-compose.yml` - Configuración multi-servicio con Nginx
- `.dockerignore` - Archivos excluidos del contexto
- `nginx.conf` - Configuración de proxy reverso
- `docker-deploy.sh` / `docker-deploy.bat` - Scripts de automatización

#### 🏗️ Construcción Manual

```bash
# Construir la imagen
docker build -t weed-segmentation:latest .

# Ejecutar el contenedor
docker run -d \
  --name weed-app \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  weed-segmentation:latest
```

#### 🚀 Scripts de Automatización

**Linux/Mac:**
```bash
# Dar permisos de ejecución
chmod +x docker-deploy.sh

# Construir imagen
./docker-deploy.sh build

# Ejecutar aplicación
./docker-deploy.sh run

# Ver logs
./docker-deploy.sh logs

# Detener aplicación
./docker-deploy.sh stop
```

**Windows:**
```cmd
# Construir imagen
docker-deploy.bat build

# Ejecutar aplicación
docker-deploy.bat run

# Ver logs
docker-deploy.bat logs

# Detener aplicación
docker-deploy.bat stop
```

#### 🐙 Docker Compose

Para un despliegue completo con Nginx:

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver logs de todos los servicios
docker-compose logs -f

# Detener todos los servicios
docker-compose down
```

## 👥 Autores

- Desarrollo principal - [@Diret03](https://github.com/Diret03)

