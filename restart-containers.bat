@echo off
echo 🛑 Deteniendo contenedores existentes...
docker-compose down

echo 🗑️ Limpiando imágenes antiguas...
docker system prune -f

echo 🔨 Reconstruyendo contenedores...
docker-compose build --no-cache

echo 🚀 Iniciando contenedores...
docker-compose up -d

echo ✅ Contenedores reiniciados exitosamente
echo 🌐 Aplicación disponible en: http://localhost

echo 📋 Ver logs:
echo docker-compose logs -f weed-segmentation

pause
