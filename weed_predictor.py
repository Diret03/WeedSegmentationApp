"""
Weed Segmentation Predictor
Standalone script for predicting weed segmentation in potato crops.
Separated from the Flask application logic for better modularity.
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

# Class definitions and colors for weed segmentation in potato crops
CLASS_NAMES = {
    0: 'background',
    1: 'cow_tongue',
    2: 'dandelion',
    3: 'kikuyo',
    4: 'other_weeds',
    5: 'potato'
}

# Colors in BGR format for OpenCV
CLASS_COLORS = {
    0: [0, 0, 0],           # Black - background
    1: [255, 0, 0],         # Blue - Cow-tongue
    2: [0, 165, 255],       # Orange - Dandelion
    3: [0, 255, 255],       # Yellow - Kikuyo
    4: [128, 0, 128],       # Purple - Other Weeds
    5: [0, 128, 0]          # Green - Potato
}

CLASS_NAMES_EN = {
    'background': 'Background',
    'cow_tongue': 'Cow-tongue',
    'dandelion': 'Dandelion',
    'kikuyo': 'Kikuyo',
    'other_weeds': 'Other Weeds',
    'potato': 'Potato'
}

# ==============================================================================
# FPN MODEL ARCHITECTURE
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
        self.training = False # Default to eval mode for inference

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
    Main class for weed segmentation prediction
    """
    def __init__(self, model_path='models/weed_segmentation_087_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.input_size = (256, 256)

        # Exact transformations from the improved training
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_model()

    def load_model(self):
        """Load the trained segmentation model with the improved script"""
        try:
            print(f"🔄 Loading improved WeedSegmenterFPN model from: {self.model_path}")

            if not os.path.exists(self.model_path):
                print(f"❌ Error: Model file not found at {self.model_path}")
                print("📝 Available models in the 'models' folder:")
                models_dir = os.path.dirname(self.model_path)
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith('.pth'):
                            print(f"  - {file}")
                return

            # Create an instance of the model with the improved architecture
            self.model = WeedSegmenterFPN(num_classes=6)

            # Load the model weights
            print("🔄 Loading model checkpoint...")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # The model from the improved script saves only the state_dict directly
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # If it's a full checkpoint
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Full checkpoint loaded")
                if 'epoch' in checkpoint:
                    print(f"📊 Epoch: {checkpoint['epoch']}")
                if 'best_val_miou' in checkpoint:
                    print(f"📊 Best mIoU: {checkpoint['best_val_miou']:.4f}")
            else:
                # If it's just the state_dict (as saved by the improved script)
                self.model.load_state_dict(checkpoint)
                print(f"✅ State dict loaded directly")

            self.model.to(self.device)
            self.model.eval()
            self.model.training = False  # Important: evaluation mode

            print(f"✅ Improved WeedSegmenterFPN model loaded successfully on {self.device}")
            print(f"📏 Input size: {self.input_size}")
            print("📊 Detected classes:", list(CLASS_NAMES_EN.values()))
            print("🎯 Model optimized with:")
            print("  - FPN architecture with attention modules")
            print("  - ASPP (Atrous Spatial Pyramid Pooling)")
            print("  - Deep supervision with auxiliary heads")
            print("  - EfficientNetV2-S as backbone")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("📋 Error details:")
            import traceback
            print(traceback.format_exc())
            print("🔄 Using demo mode with simulated data")
            self.model = None

    def preprocess_image(self, image_path):
        """Preprocess image for the model"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor.to(self.device), original_size

    def predict(self, image_path):
        """
        Predict segmentation using the trained model
        Returns segmentation mask as a numpy array with values 0-5
        """
        if self.model is None:
            print("⚠️ Model not available, using simulated data")
            return self._create_dummy_mask(image_path)

        try:
            print(f"🔍 Starting prediction for: {os.path.basename(image_path)}")

            input_tensor, original_size = self.preprocess_image(image_path)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_mask = torch.argmax(probabilities, dim=1)
                mask = predicted_mask.cpu().numpy()[0]

                # Resize to original size
                original_image = cv2.imread(image_path)
                original_height, original_width = original_image.shape[:2]
                mask_resized = cv2.resize(mask.astype(np.uint8),
                                        (original_width, original_height),
                                        interpolation=cv2.INTER_NEAREST)

                unique_classes = np.unique(mask_resized)
                detected_classes = [CLASS_NAMES[cls] for cls in unique_classes if cls in CLASS_NAMES]
                print(f"✅ Prediction completed. Detected classes: {detected_classes}")

                return mask_resized

        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            print("🔄 Using simulated data as fallback")
            return self._create_dummy_mask(image_path)

    def _create_dummy_mask(self, image_path):
        """Create a simulated mask when the model is not available"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Simulate realistic detections
        cv2.circle(mask, (width//4, height//4), 40, 1, -1)  # Cow-tongue
        cv2.circle(mask, (3*width//4, height//3), 25, 1, -1)
        cv2.ellipse(mask, (width//3, 2*height//3), (30, 20), 0, 0, 360, 2, -1)  # Dandelion
        cv2.ellipse(mask, (2*width//3, height//4), (25, 15), 45, 0, 360, 2, -1)
        cv2.rectangle(mask, (width//6, height//2), (width//6 + 50, height//2 + 40), 3, -1)  # Kikuyo
        cv2.circle(mask, (5*width//6, 2*height//3), 20, 4, -1)  # Other weeds
        cv2.ellipse(mask, (width//2, height//2), (60, 80), 0, 0, 360, 5, -1)  # Potato
        cv2.ellipse(mask, (width//5, 3*height//4), (40, 50), 30, 0, 360, 5, -1)
        cv2.ellipse(mask, (4*width//5, height//6), (35, 45), -20, 0, 360, 5, -1)

        print("⚠️ Using simulated mask for demonstration")
        return mask

    def calculate_class_statistics(self, mask):
        """Calculates detailed statistics per class from the segmentation mask"""
        total_pixels = mask.size
        stats = {}

        for class_id, class_name in CLASS_NAMES.items():
            class_pixels = np.sum(mask == class_id)
            percentage = (class_pixels / total_pixels) * 100
            stats[class_name] = percentage

        # Additional agriculture-specific statistics
        weed_classes = ['cow_tongue', 'dandelion', 'kikuyo', 'other_weeds']
        weed_pixels = sum(stats[weed] for weed in weed_classes)

        # Count detected weed types (with a minimum threshold of 0.1%)
        detected_weed_types = len([weed for weed in weed_classes if stats[weed] > 0.1])

        # Additional metrics
        stats['total_weeds'] = detected_weed_types
        stats['potato_area'] = f"{stats['potato']:.1f}%"
        stats['weed_coverage'] = f"{weed_pixels:.1f}%"
        stats['crop_health_ratio'] = stats['potato'] / (weed_pixels + 0.001)

        return stats

    def create_overlay_visualization(self, image_path, mask, alpha=0.6, beta=0.4):
        """
        Create an overlay visualization of the segmentation results
        """
        image = cv2.imread(image_path)
        overlay = np.zeros_like(image)

        # Apply colors to all classes
        for class_id, color in CLASS_COLORS.items():
            class_mask = mask == class_id
            overlay[class_mask] = color

        # Combine original image with overlay
        result = cv2.addWeighted(image, beta, overlay, alpha, 0)
        return result

def main():
    """Example function for standalone use of the predictor"""
    # Example of using the predictor
    predictor = WeedSegmentationPredictor()

    # Test image path (change to a real image)
    test_image = "appTest.png"

    if os.path.exists(test_image):
        print(f"🔍 Processing image: {test_image}")

        # Perform prediction
        mask = predictor.predict(test_image)

        # Calculate statistics
        stats = predictor.calculate_class_statistics(mask)

        print("\n📊 Segmentation Statistics:")
        for class_name, percentage in stats.items():
            if class_name in CLASS_NAMES_EN:
                print(f"  {CLASS_NAMES_EN[class_name]}: {percentage:.2f}%")

        # Create visualization
        overlay = predictor.create_overlay_visualization(test_image, mask)

        # Save result
        cv2.imwrite("prediction_result.jpg", overlay)
        print("✅ Result saved as 'prediction_result.jpg'")
    else:
        print(f"❌ Test image not found: {test_image}")

if __name__ == "__main__":
    main()
