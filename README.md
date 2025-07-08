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

> **Nota**: El modelo debe ser entrenado previamente o descargado desde el repositorio de modelos.

## 🏃‍♂️ Ejecución

### Modo de Desarrollo

```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

### Modo de Producción

```bash
# Usando Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

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

## 🔧 API Endpoints

### POST `/upload`

Procesa una imagen y retorna la segmentación.

**Request:**
- Método: `POST`
- Content-Type: `multipart/form-data`
- Body: archivo de imagen (JPG, PNG, GIF, BMP)

**Response:**
```json
{
  "success": true,
  "original_image": "data:image/jpeg;base64,...",
  "segmented_image": "data:image/jpeg;base64,...",
  "animation_frames": ["data:image/jpeg;base64,..."],
  "filename": "uuid-filename.jpg",
  "class_stats": {
    "background": 45.2,
    "lengua_vaca": 12.3,
    "diente_leon": 8.7,
    "kikuyo": 15.1,
    "otras_malezas": 6.2,
    "papa": 12.5
  },
  "detected_classes": ["Papa", "Lengua de Vaca", "Kikuyo"]
}
```

### GET `/download/<filename>`

Descarga la imagen procesada.

## 🎨 Interfaz de Usuario

### Características de la UI

- **Drag & Drop**: Arrastra imágenes directamente a la zona de carga
- **Animaciones Suaves**: Transiciones CSS y efectos visuales
- **Responsive Design**: Compatible con dispositivos móviles
- **Visualización Progresiva**: Animación frame-by-frame de la segmentación
- **Estadísticas en Tiempo Real**: Métricas detalladas de detección

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

## 🔧 Configuración Avanzada

### Personalización del Modelo

Para usar un modelo diferente, modifica en `weed_predictor.py`:

```python
# Cambiar ruta del modelo
model_path = 'models/tu_modelo_personalizado.pth'

# Ajustar número de clases
num_classes = 6  # Cambiar según tu modelo
```

### Configuración de la Aplicación

Variables de entorno importantes:

```bash
export FLASK_ENV=development  # o production
export FLASK_DEBUG=1          # para modo debug
export MODEL_PATH=models/weed_segmenter_fpn_model_085_local.pth
```

## 🧪 Testing

### Ejecutar Pruebas

```bash
# Instalar dependencias de testing
pip install pytest

# Ejecutar pruebas
pytest tests/
```

### Pruebas de Imagen

Para probar con imágenes de ejemplo:

```bash
python script_de_segmentación_con_métricas_y_gráficos_mejorados.py
```

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

**Servicios incluidos:**
- **weed-segmentation**: Aplicación Flask principal
- **nginx**: Proxy reverso con balanceeo de carga

#### ⚙️ Variables de Entorno

```bash
# Variables principales
FLASK_ENV=production
FLASK_APP=app.py
MODEL_PATH=models/weed_segmenter_fpn_model_085_local.pth
```

#### 🔍 Health Checks

El contenedor incluye health checks automáticos:
- Endpoint: `http://localhost:5000/`
- Intervalo: 30 segundos
- Timeout: 30 segundos
- Reintentos: 3

### ☁️ Heroku

```bash
# Instalar Heroku CLI y login
heroku login

# Crear aplicación
heroku create tu-app-weed-segmentation

# Configurar variables de entorno
heroku config:set FLASK_ENV=production

# Crear Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Escalar dynos
heroku ps:scale web=1
```

### ☁️ AWS ECS

```bash
# Construir y taggar para ECR
docker build -t your-account.dkr.ecr.region.amazonaws.com/weed-segmentation:latest .

# Push a ECR
docker push your-account.dkr.ecr.region.amazonaws.com/weed-segmentation:latest

# Crear task definition y service en ECS
```

### 🌐 DigitalOcean App Platform

```yaml
# app.yaml
name: weed-segmentation
services:
- name: web
  source_dir: /
  github:
    repo: tu-usuario/WeedSegmentationApp
    branch: main
  run_command: gunicorn --worker-tmp-dir /dev/shm app:app
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
  envs:
  - key: FLASK_ENV
    value: production
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Tu Nombre** - Desarrollo principal - [@tu-usuario](https://github.com/tu-usuario)

## 🙏 Agradecimientos

- Equipo de investigación en agricultura de precisión
- Comunidad de PyTorch y computer vision
- Proveedores de datasets de cultivos agrícolas

## 📞 Soporte

Para soporte técnico o preguntas:

- 📧 Email: tu-email@ejemplo.com
- 🐛 Issues: [GitHub Issues](https://github.com/tu-usuario/WeedSegmentationApp/issues)
- 📖 Documentación: [Wiki del proyecto](https://github.com/tu-usuario/WeedSegmentationApp/wiki)

---

⭐ **¡Si este proyecto te fue útil, considera darle una estrella en GitHub!** ⭐
