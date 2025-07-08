from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid

# Importar la lógica de predicción desde el script separado
from weed_predictor import WeedSegmentationPredictor, CLASS_COLORS, CLASS_NAMES_ES

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize model using the separated predictor with the new improved model
print("🚀 Inicializando modelo de segmentación de malezas WeedSegmenter mejorado...")
segmentation_model = WeedSegmentationPredictor(model_path='models/weed_segmenter_fpn_model_085_local.pth')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print(f"📁 Archivo guardado: {filepath}")
            print(f"📊 Tamaño del archivo: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
            
            # Run segmentation
            print("🧠 Iniciando segmentación...")
            mask = segmentation_model.predict(filepath)
            print("✅ Segmentación completada")

            # Calculate class statistics
            print("📊 Calculando estadísticas...")
            class_stats = segmentation_model.calculate_class_statistics(mask)

            # Create overlay visualization with animation frames
            print("🎨 Generando visualización...")
            result_path, animation_frames = create_weed_overlay(filepath, mask, filename)

            # Convert images to base64 for frontend
            print("🔄 Convirtiendo imágenes a base64...")
            original_b64 = image_to_base64(filepath)
            result_b64 = image_to_base64(result_path)

            print("✅ Procesamiento completado exitosamente")
            
            return jsonify({
                'success': True,
                'original_image': original_b64,
                'segmented_image': result_b64,
                'animation_frames': animation_frames,  # NUEVO: Frames para animación
                'filename': filename,
                'class_stats': class_stats,
                'detected_classes': [CLASS_NAMES_ES[cls] for cls, pct in class_stats.items()
                                   if cls in CLASS_NAMES_ES and pct > 0.1]
            })

        except Exception as e:
            print(f"❌ Error en el procesamiento: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error en el procesamiento: {str(e)}'}), 500

    return jsonify({'error': 'Tipo de archivo no válido'}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_weed_overlay(image_path, mask, filename):
    """
    Crea visualización overlay de resultados de segmentación de malezas con transparencia mejorada
    y genera frames para animación progresiva
    """
    # Load original image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    print(f"📏 Procesando imagen: {width}x{height}")

    # Create colored overlay based on segmentation classes
    overlay = np.zeros_like(image)

    # Aplicar colores a TODAS las clases (incluyendo background)
    for class_id, color in CLASS_COLORS.items():
        class_mask = mask == class_id
        overlay[class_mask] = color

    # CORREGIDO: Aumentar transparencia del overlay para que el background sea más visible
    alpha = 0.6  # Aumentado de 0.3 a 0.6 para mayor visibilidad
    beta = 0.4   # Reducido de 0.7 a 0.4 - menos peso a la imagen original

    # Aplicar overlay a toda la imagen
    result = cv2.addWeighted(image, beta, overlay, alpha, 0)

    # ======= OPTIMIZADO: Generar frames para animación con menos memoria =======
    num_frames = 10  # Reducido de 20 a 10 frames para ahorrar memoria
    animation_frames = []

    print(f"🎬 Generando {num_frames} frames de animación...")
    
    # Crear máscara de revelación progresiva (barrido de izquierda a derecha)
    for frame_idx in range(num_frames + 1):
        # Calcular el porcentaje de revelación (0 a 1)
        reveal_progress = frame_idx / num_frames

        # Crear máscara de revelación circular que se expande desde el centro
        center_x, center_y = width // 2, height // 2
        max_radius = np.sqrt((width/2)**2 + (height/2)**2)
        current_radius = reveal_progress * max_radius * 1.2  # Factor 1.2 para asegurar cobertura completa

        # Crear máscara circular
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        reveal_mask = distances <= current_radius

        # Crear frame progresivo
        frame = image.copy()
        frame[reveal_mask] = result[reveal_mask]

        # Guardar frame
        frame_filename = f"frame_{frame_idx:02d}_{filename}"
        frame_path = os.path.join(app.config['RESULTS_FOLDER'], frame_filename)
        cv2.imwrite(frame_path, frame)

        # Convertir a base64 para envío al frontend
        frame_b64 = image_to_base64(frame_path)
        animation_frames.append(frame_b64)
        
        # Limpiar el frame de memoria
        del frame
        
        if frame_idx % 3 == 0:  # Log cada 3 frames
            print(f"  📄 Frame {frame_idx}/{num_frames} generado")

    # Save final result
    result_filename = f"result_{filename}"
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    cv2.imwrite(result_path, result)
    
    print(f"✅ Animación completada: {len(animation_frames)} frames generados")

    return result_path, animation_frames

def image_to_base64(image_path):
    """Convert image to base64 string for frontend display"""
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        ext = image_path.split('.')[-1].lower()
        return f"data:image/{ext};base64,{img_base64}"

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    print("🌱 Iniciando aplicación de segmentación de malezas en cultivos de papa")
    print("📊 Clases detectadas:", list(CLASS_NAMES_ES.values()))
    app.run(debug=True, host='0.0.0.0', port=5000)
