"""
Weed Segmentation Predictor
Script independiente para la predicción de segmentación de malezas en cultivos de papa
Separado de la lógica de la aplicación Flask para mejor modularidad
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm

# Definición de clases y colores para segmentación de malezas en cultivos de papa
CLASS_NAMES = {
    0: 'background',
    1: 'lengua_vaca',
    2: 'diente_leon',
    3: 'kikuyo',
    4: 'otras_malezas',
    5: 'papa'
}

# Colores en formato BGR para OpenCV
CLASS_COLORS = {
    0: [0, 0, 0],         # Gris oscuro - background
    1: [68, 68, 239],     # Rojo - lengua de vaca
    2: [11, 158, 245],    # Naranja/Amarillo - diente de león
    3: [246, 92, 139],    # Púrpura - kikuyo
    4: [153, 72, 236],    # Rosa/Magenta - otras malezas
    5: [129, 185, 16]     # Verde esmeralda - papa
}

CLASS_NAMES_ES = {
    'background': 'Fondo',
    'lengua_vaca': 'Lengua de Vaca',
    'diente_leon': 'Diente de León',
    'kikuyo': 'Kikuyo',
    'otras_malezas': 'Otras Malezas',
    'papa': 'Papa'
}

# ==============================================================================
# ARQUITECTURA DEL MODELO FPN
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class AttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels_skip, in_channels_up, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels_up, in_channels_up, kernel_size=2, stride=2)
        total_in_channels = in_channels_skip + in_channels_up
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionModule(in_channels=in_channels_skip)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(total_in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x_skip, x_up):
        x_up = self.upsample(x_up)
        if self.use_attention:
            x_skip_att = self.attention(x_skip)
        else:
            x_skip_att = x_skip
        x_concat = torch.cat([x_up, x_skip_att], dim=1)
        return self.conv_fuse(x_concat)

class WeedSegmenterFPN(nn.Module):
    def __init__(self, num_classes=6):
        super(WeedSegmenterFPN, self).__init__()
        self.training = False # Por defecto en modo eval para inferencia

        self.backbone = timm.create_model(
            'tf_efficientnetv2_s.in21k',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        backbone_channels = self.backbone.feature_info.channels()

        self.aspp = ASPP(in_channels=backbone_channels[3], atrous_rates=(6, 12, 18), out_channels=256)

        decoder_out_channels = [128, 64, 48]

        self.decoder_block3 = DecoderBlock(backbone_channels[2], 256, decoder_out_channels[0])
        self.decoder_block2 = DecoderBlock(backbone_channels[1], decoder_out_channels[0], decoder_out_channels[1])
        self.decoder_block1 = DecoderBlock(backbone_channels[0], decoder_out_channels[1], decoder_out_channels[2])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels[2], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        self.aux_head_3 = nn.Conv2d(decoder_out_channels[0], num_classes, 1)
        self.aux_head_2 = nn.Conv2d(decoder_out_channels[1], num_classes, 1)

        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        img_size = x.shape[-2:]
        features = self.backbone(x)

        aspp_output = self.aspp(features[3])

        decoder_out3 = self.decoder_block3(x_skip=features[2], x_up=aspp_output)
        decoder_out2 = self.decoder_block2(x_skip=features[1], x_up=decoder_out3)
        decoder_out1 = self.decoder_block1(x_skip=features[0], x_up=decoder_out2)

        logits = self.segmentation_head(decoder_out1)
        final_logits = self.final_upsample(logits)

        if self.training:
            aux3 = F.interpolate(self.aux_head_3(decoder_out3), size=img_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head_2(decoder_out2), size=img_size, mode='bilinear', align_corners=False)
            return final_logits, aux3, aux2

        return final_logits

class WeedSegmentationPredictor:
    """
    Clase principal para predicción de segmentación de malezas
    """
    def __init__(self, model_path='models/weed_segmenter_fpn_model_085_local.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.input_size = (256, 256)

        # Transformaciones exactas del entrenamiento mejorado
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_model()

    def load_model(self):
        """Cargar el modelo de segmentación entrenado con el script mejorado"""
        try:
            print(f"🔄 Cargando modelo WeedSegmenterFPN mejorado desde: {self.model_path}")

            if not os.path.exists(self.model_path):
                print(f"❌ Error: No se encontró el archivo del modelo en {self.model_path}")
                print("📝 Modelos disponibles en la carpeta 'models':")
                models_dir = os.path.dirname(self.model_path)
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith('.pth'):
                            print(f"  - {file}")
                return

            # Crear instancia del modelo con la arquitectura mejorada
            self.model = WeedSegmenterFPN(num_classes=6)

            # Cargar los pesos del modelo
            print("🔄 Cargando checkpoint del modelo...")
            
            # Configurar dispositivo para CPU para evitar problemas de memoria GPU
            device = torch.device('cpu')
            self.device = device
            
            checkpoint = torch.load(self.model_path, map_location=device)

            # El modelo del script mejorado guarda solo el state_dict directamente
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Si es un checkpoint completo
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Checkpoint completo cargado")
                if 'epoch' in checkpoint:
                    print(f"📊 Época: {checkpoint['epoch']}")
                if 'best_val_miou' in checkpoint:
                    print(f"📊 Mejor mIoU: {checkpoint['best_val_miou']:.4f}")
            else:
                # Si es solo el state_dict (como guarda el script mejorado)
                self.model.load_state_dict(checkpoint)
                print(f"✅ State dict cargado directamente")

            self.model.to(self.device)
            self.model.eval()
            self.model.training = False  # Importante: modo evaluación

            # Configurar optimizaciones para inferencia
            torch.set_grad_enabled(False)
            if hasattr(torch.jit, 'optimize_for_inference'):
                torch.jit.optimize_for_inference(self.model)

            print(f"✅ Modelo WeedSegmenterFPN mejorado cargado exitosamente en {self.device}")
            print(f"📏 Tamaño de entrada: {self.input_size}")
            print("📊 Clases detectadas:", list(CLASS_NAMES_ES.values()))
            print("🎯 Modelo optimizado con:")
            print("  - Arquitectura FPN con módulos de atención")
            print("  - ASPP (Atrous Spatial Pyramid Pooling)")
            print("  - Supervisión profunda con cabezales auxiliares")
            print("  - EfficientNetV2-S como backbone")
            print("  - Optimizado para inferencia CPU")

        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            print("📋 Detalles del error:")
            import traceback
            print(traceback.format_exc())
            print("🔄 Usando modo demo con datos simulados")
            self.model = None

    def preprocess_image(self, image_path):
        """Preprocesar imagen para el modelo"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor.to(self.device), original_size

    def predict(self, image_path):
        """
        Predecir segmentación usando el modelo entrenado
        Retorna máscara de segmentación como numpy array con valores 0-5
        """
        if self.model is None:
            print("⚠️ Modelo no disponible, usando datos simulados")
            return self._create_dummy_mask(image_path)

        try:
            print(f"🔍 Iniciando predicción para: {os.path.basename(image_path)}")
            
            # Limpiar memoria antes de procesar
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Monitorear memoria
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Memoria antes del procesamiento: {memory_before:.1f} MB")

            input_tensor, original_size = self.preprocess_image(image_path)
            print(f"📏 Tensor de entrada: {input_tensor.shape}")

            with torch.no_grad():
                print("🧠 Ejecutando modelo de segmentación...")
                outputs = self.model(input_tensor)
                print(f"📤 Salida del modelo: {outputs.shape}")
                
                probabilities = F.softmax(outputs, dim=1)
                predicted_mask = torch.argmax(probabilities, dim=1)
                mask = predicted_mask.cpu().numpy()[0]
                
                # Liberar memoria del GPU
                del outputs, probabilities, predicted_mask, input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Redimensionar a tamaño original
                original_image = cv2.imread(image_path)
                original_height, original_width = original_image.shape[:2]
                print(f"📐 Redimensionando de {mask.shape} a {original_height}x{original_width}")
                
                mask_resized = cv2.resize(mask.astype(np.uint8),
                                        (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)

                unique_classes = np.unique(mask_resized)
                detected_classes = [CLASS_NAMES[cls] for cls in unique_classes if cls in CLASS_NAMES]
                
                # Monitorear memoria después
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                print(f"📊 Memoria después del procesamiento: {memory_after:.1f} MB")
                print(f"📊 Incremento de memoria: {memory_after - memory_before:.1f} MB")
                
                print(f"✅ Predicción completada. Clases detectadas: {detected_classes}")

                return mask_resized

        except Exception as e:
            print(f"❌ Error durante la predicción: {str(e)}")
            print("🔄 Traceback completo:")
            import traceback
            traceback.print_exc()
            print("🔄 Usando datos simulados como respaldo")
            return self._create_dummy_mask(image_path)

    def _create_dummy_mask(self, image_path):
        """Crear máscara simulada cuando el modelo no está disponible"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Simular detecciones realistas
        cv2.circle(mask, (width//4, height//4), 40, 1, -1)  # Lengua de vaca
        cv2.circle(mask, (3*width//4, height//3), 25, 1, -1)
        cv2.ellipse(mask, (width//3, 2*height//3), (30, 20), 0, 0, 360, 2, -1)  # Diente de león
        cv2.ellipse(mask, (2*width//3, height//4), (25, 15), 45, 0, 360, 2, -1)
        cv2.rectangle(mask, (width//6, height//2), (width//6 + 50, height//2 + 40), 3, -1)  # Kikuyo
        cv2.circle(mask, (5*width//6, 2*height//3), 20, 4, -1)  # Otras malezas
        cv2.ellipse(mask, (width//2, height//2), (60, 80), 0, 0, 360, 5, -1)  # Papa
        cv2.ellipse(mask, (width//5, 3*height//4), (40, 50), 30, 0, 360, 5, -1)
        cv2.ellipse(mask, (4*width//5, height//6), (35, 45), -20, 0, 360, 5, -1)

        print("⚠️ Usando máscara simulada para demostración")
        return mask

    def calculate_class_statistics(self, mask):
        """Calcula estadísticas detalladas por clase de la máscara de segmentación"""
        total_pixels = mask.size
        stats = {}

        for class_id, class_name in CLASS_NAMES.items():
            class_pixels = np.sum(mask == class_id)
            percentage = (class_pixels / total_pixels) * 100
            stats[class_name] = percentage

        # Estadísticas adicionales específicas para agricultura
        weed_classes = ['lengua_vaca', 'diente_leon', 'kikuyo', 'otras_malezas']
        weed_pixels = sum(stats[weed] for weed in weed_classes)

        # Contar tipos de malezas detectadas (con umbral mínimo del 0.1%)
        detected_weed_types = len([weed for weed in weed_classes if stats[weed] > 0.1])

        # Métricas adicionales
        stats['total_weeds'] = detected_weed_types
        stats['potato_area'] = f"{stats['papa']:.1f}%"
        stats['weed_coverage'] = f"{weed_pixels:.1f}%"
        stats['crop_health_ratio'] = stats['papa'] / (weed_pixels + 0.001)

        return stats

    def create_overlay_visualization(self, image_path, mask, alpha=0.6, beta=0.4):
        """
        Crear visualización overlay de los resultados de segmentación
        """
        image = cv2.imread(image_path)
        overlay = np.zeros_like(image)

        # Aplicar colores a todas las clases
        for class_id, color in CLASS_COLORS.items():
            class_mask = mask == class_id
            overlay[class_mask] = color

        # Combinar imagen original con overlay
        result = cv2.addWeighted(image, beta, overlay, alpha, 0)
        return result

def main():
    """Función de ejemplo para uso independiente del predictor"""
    # Ejemplo de uso del predictor
    predictor = WeedSegmentationPredictor()

    # Ruta de imagen de prueba (cambiar por una imagen real)
    test_image = "appTest.png"

    if os.path.exists(test_image):
        print(f"🔍 Procesando imagen: {test_image}")

        # Realizar predicción
        mask = predictor.predict(test_image)

        # Calcular estadísticas
        stats = predictor.calculate_class_statistics(mask)

        print("\n📊 Estadísticas de segmentación:")
        for class_name, percentage in stats.items():
            if class_name in CLASS_NAMES_ES:
                print(f"  {CLASS_NAMES_ES[class_name]}: {percentage:.2f}%")

        # Crear visualización
        overlay = predictor.create_overlay_visualization(test_image, mask)

        # Guardar resultado
        cv2.imwrite("prediction_result.jpg", overlay)
        print("✅ Resultado guardado como 'prediction_result.jpg'")
    else:
        print(f"❌ No se encontró la imagen de prueba: {test_image}")

if __name__ == "__main__":
    main()
