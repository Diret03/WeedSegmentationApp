from flask import Flask, render_template, request, jsonify, send_from_directory, g
import os
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid
import traceback

# Import prediction logic
from weed_predictor import WeedSegmentationPredictor, CLASS_COLORS, CLASS_NAMES

# Import professional logging
from logger_config import (
    setup_logging, log_request, log_performance, 
    ErrorCodes, ValidationError, ProcessingError, ModelError,
    create_error_response, create_success_response
)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Setup professional logging
logger = setup_logging(
    app_name="weed_segmentation_app",
    log_file="logs/app.log"
)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize model
logger.info("Initializing weed segmentation model with TTA enhancement")
try:
    segmentation_model = WeedSegmentationPredictor(model_path='models/weed_segmentation_S-TTA.pth')
    logger.info("Model initialization completed successfully")
except Exception as e:
    logger.error("Failed to initialize segmentation model", exc_info=True)
    segmentation_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@log_request
@log_performance
def upload_file():
    """Handle file upload and weed segmentation processing"""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Validate file presence
        if 'file' not in request.files:
            logger.warning("File upload attempted without file", extra={'request_id': request_id})
            return jsonify(create_error_response(
                ErrorCodes.FILE_NOT_PROVIDED,
                "No file was provided in the request",
                request_id=request_id
            )), 400

        file = request.files['file']
        if file.filename == '':
            logger.warning("File upload attempted with empty filename", extra={'request_id': request_id})
            return jsonify(create_error_response(
                ErrorCodes.FILE_NOT_SELECTED,
                "No file was selected",
                request_id=request_id
            )), 400

        # Validate file type
        if not file or not allowed_file(file.filename):
            logger.warning(
                "Invalid file type uploaded",
                extra={
                    'request_id': request_id,
                    'filename': file.filename,
                    'content_type': file.content_type
                }
            )
            return jsonify(create_error_response(
                ErrorCodes.FILE_TYPE_INVALID,
                "Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP",
                {'supported_formats': ['PNG', 'JPG', 'JPEG', 'GIF', 'BMP']},
                request_id=request_id
            )), 400

        # Generate unique filename and save
        original_filename = file.filename
        unique_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        logger.info(
            "Processing file upload",
            extra={
                'request_id': request_id,
                'original_filename': original_filename,
                'unique_filename': unique_filename,
                'file_size': len(file.read())
            }
        )
        
        # Reset file pointer after reading for size
        file.seek(0)
        file.save(filepath)

        # Validate image dimensions
        try:
            is_valid, error_message, image_info = validate_image_dimensions_enhanced(filepath)
            if not is_valid:
                # Clean up uploaded file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                logger.warning(
                    "Image validation failed",
                    extra={
                        'request_id': request_id,
                        'error': error_message,
                        'image_info': image_info
                    }
                )
                
                return jsonify(create_error_response(
                    ErrorCodes.FILE_DIMENSIONS_INVALID,
                    error_message,
                    image_info,
                    request_id=request_id
                )), 400
                
        except Exception as e:
            logger.error(
                "Image validation error",
                extra={'request_id': request_id},
                exc_info=True
            )
            return jsonify(create_error_response(
                ErrorCodes.FILE_CORRUPTED,
                "Unable to process image file",
                {'error': str(e)},
                request_id=request_id
            )), 400

        # Process with model
        if segmentation_model is None:
            logger.error("Segmentation model not available", extra={'request_id': request_id})
            return jsonify(create_error_response(
                ErrorCodes.MODEL_NOT_FOUND,
                "Segmentation model is not available",
                request_id=request_id
            )), 500

        try:
            # Run segmentation with TTA
            logger.info("Starting TTA segmentation", extra={'request_id': request_id})
            mask = segmentation_model.predict(filepath)

            # Calculate class statistics
            class_stats = segmentation_model.calculate_class_statistics(mask)

            # Create overlay visualization with animation frames
            result_path, animation_frames = create_weed_overlay(filepath, mask, unique_filename)

            # Convert images to base64 for frontend
            original_b64 = image_to_base64(filepath)
            result_b64 = image_to_base64(result_path)

            # Prepare detected classes info
            detected_classes = [CLASS_NAMES[cls] for cls, pct in class_stats.items()
                              if cls in CLASS_NAMES and pct > 0.1]
            
            logger.info(
                "Segmentation completed successfully",
                extra={
                    'request_id': request_id,
                    'detected_classes': detected_classes,
                    'image_dimensions': f"{image_info['width']}x{image_info['height']}",
                    'animation_frames_count': len(animation_frames)
                }
            )

            response_data = {
                'original_image': original_b64,
                'segmented_image': result_b64,
                'animation_frames': animation_frames,
                'filename': unique_filename,
                'class_stats': class_stats,
                'detected_classes': detected_classes,
                'processing_info': {
                    'tta_enabled': True,
                    'model_type': 'WeedSegmenterFPN',
                    'input_size': segmentation_model.input_size
                }
            }

            return jsonify(create_success_response(
                response_data,
                "Weed segmentation completed successfully",
                request_id=request_id
            ))

        except Exception as e:
            logger.error(
                "Segmentation processing failed",
                extra={'request_id': request_id},
                exc_info=True
            )
            return jsonify(create_error_response(
                ErrorCodes.PREDICTION_FAILED,
                "Error during image processing",
                {'error': str(e)},
                request_id=request_id
            )), 500

    except Exception as e:
        logger.error(
            "Unexpected error in upload handler",
            extra={'request_id': request_id},
            exc_info=True
        )
        return jsonify(create_error_response(
            "INTERNAL_ERROR",
            "An unexpected error occurred",
            {'error': str(e)},
            request_id=request_id
        )), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_dimensions_enhanced(filepath, max_width=256, max_height=256):
    """
    Enhanced image validation with detailed error reporting
    
    Returns:
        tuple: (is_valid, error_message, image_info)
    """
    try:
        with Image.open(filepath) as img:
            width, height = img.size
            file_size = os.path.getsize(filepath)
            
            image_info = {
                'width': width,
                'height': height,
                'format': img.format,
                'mode': img.mode,
                'file_size': file_size
            }
            
            if width > max_width or height > max_height:
                error_msg = f'Image dimensions ({width}x{height}) exceed maximum allowed size of {max_width}x{max_height} pixels'
                return False, error_msg, image_info
            
            # Additional validation for minimum size
            if width < 32 or height < 32:
                error_msg = f'Image dimensions ({width}x{height}) are too small. Minimum size is 32x32 pixels'
                return False, error_msg, image_info
            
            return True, None, image_info
            
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}", exc_info=True)
        return False, f'Unable to read image file: {str(e)}', {'error': str(e)}

@log_performance
def create_weed_overlay(image_path, mask, filename):
    """
    Enhanced overlay creation with proper error handling
    """
    try:
        logger.info(f"Creating overlay visualization for {filename}")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ProcessingError(
                "Unable to load image for overlay creation",
                ErrorCodes.IMAGE_PROCESSING_FAILED
            )
            
        height, width = image.shape[:2]
        logger.debug(f"Image dimensions: {width}x{height}")

        # Create colored overlay based on segmentation classes
        overlay = np.zeros_like(image)

        # Apply colors to all classes
        for class_id, color in CLASS_COLORS.items():
            class_mask = mask == class_id
            overlay[class_mask] = color

        # Apply overlay parameters
        alpha = 0.6  # Overlay transparency
        beta = 0.4   # Original image weight

        # Create final result
        result = cv2.addWeighted(image, beta, overlay, alpha, 0)

        # Generate animation frames
        num_frames = 20
        animation_frames = []

        logger.debug(f"Generating {num_frames} animation frames")

        for frame_idx in range(num_frames + 1):
            # Calculate reveal progress (0 to 1)
            reveal_progress = frame_idx / num_frames

            # Create circular reveal mask from center
            center_x, center_y = width // 2, height // 2
            max_radius = np.sqrt((width/2)**2 + (height/2)**2)
            current_radius = reveal_progress * max_radius * 1.2

            # Create circular mask
            y_coords, x_coords = np.ogrid[:height, :width]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            reveal_mask = distances <= current_radius

            # Create progressive frame
            frame = image.copy()
            frame[reveal_mask] = result[reveal_mask]

            # Save frame
            frame_filename = f"frame_{frame_idx:02d}_{filename}"
            frame_path = os.path.join(app.config['RESULTS_FOLDER'], frame_filename)
            
            if not cv2.imwrite(frame_path, frame):
                logger.warning(f"Failed to save animation frame {frame_idx}")
                continue

            # Convert to base64
            try:
                frame_b64 = image_to_base64(frame_path)
                animation_frames.append(frame_b64)
            except Exception as e:
                logger.warning(f"Failed to convert frame {frame_idx} to base64: {str(e)}")

        # Save final result
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        if not cv2.imwrite(result_path, result):
            raise ProcessingError(
                "Failed to save final overlay result",
                ErrorCodes.OVERLAY_CREATION_FAILED
            )

        logger.info(
            f"Overlay creation completed",
            extra={
                'file_name': filename,  # Changed from 'filename' to 'file_name'
                'animation_frames_generated': len(animation_frames),
                'result_path': result_path
            }
        )

        return result_path, animation_frames
        
    except Exception as e:
        logger.error(f"Error creating overlay for {filename}", exc_info=True)
        raise ProcessingError(
            f"Failed to create overlay visualization: {str(e)}",
            ErrorCodes.OVERLAY_CREATION_FAILED
        )

def image_to_base64(image_path):
    """Convert image to base64 string with error handling"""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            ext = image_path.split('.')[-1].lower()
            return f"data:image/{ext};base64,{img_base64}"
            
    except Exception as e:
        logger.error(f"Error converting image to base64: {image_path}", exc_info=True)
        raise ProcessingError(
            f"Failed to convert image to base64: {str(e)}",
            ErrorCodes.IMAGE_PROCESSING_FAILED
        )

@app.route('/download/<filename>')
@log_request
def download_file(filename):
    """Handle file download with security validation"""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Validate filename to prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(
                "Suspicious filename in download request",
                extra={'request_id': request_id, 'filename': filename}
            )
            return jsonify(create_error_response(
                "SECURITY_ERROR",
                "Invalid filename",
                request_id=request_id
            )), 400
        
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.warning(
                "Download requested for non-existent file",
                extra={'request_id': request_id, 'filename': filename}
            )
            return jsonify(create_error_response(
                "FILE_NOT_FOUND",
                "Requested file does not exist",
                request_id=request_id
            )), 404
        
        logger.info(
            "File download initiated",
            extra={'request_id': request_id, 'filename': filename}
        )
        
        return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)
        
    except Exception as e:
        logger.error(
            "Error during file download",
            extra={'request_id': request_id, 'filename': filename},
            exc_info=True
        )
        return jsonify(create_error_response(
            "DOWNLOAD_ERROR",
            "Error downloading file",
            {'error': str(e)},
            request_id=request_id
        )), 500

@app.route('/get_model_info')
@log_request
def get_model_info():
    """Get current model information with enhanced details"""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        if segmentation_model is None:
            logger.warning("Model info requested but model not available", extra={'request_id': request_id})
            return jsonify(create_error_response(
                ErrorCodes.MODEL_NOT_FOUND,
                "Segmentation model is not available",
                request_id=request_id
            )), 503
        
        model_info = {
            'model_path': segmentation_model.model_path,
            'tta_enabled': True,
            'device': str(segmentation_model.device),
            'input_size': segmentation_model.input_size,
            'num_tta_augmentations': len(segmentation_model.tta_transforms),
            'classes': list(CLASS_NAMES.values()),
            'status': 'ready',
            'version': '1.0.0'
        }
        
        logger.info("Model information retrieved", extra={'request_id': request_id})
        
        return jsonify(create_success_response(
            model_info,
            "Model information retrieved successfully",
            request_id=request_id
        ))
        
    except Exception as e:
        logger.error(
            "Error retrieving model information",
            extra={'request_id': request_id},
            exc_info=True
        )
        return jsonify(create_error_response(
            "MODEL_INFO_ERROR",
            "Error retrieving model information",
            {'error': str(e)},
            request_id=request_id
        )), 500

@app.errorhandler(413)
def file_too_large(error):
    """Handle file size exceeded error"""
    logger.warning(f"File upload rejected - size exceeds limit: {app.config['MAX_CONTENT_LENGTH']} bytes")
    return jsonify(create_error_response(
        ErrorCodes.FILE_SIZE_EXCEEDED,
        f"File size exceeds maximum limit of {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"
    )), 413

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    logger.warning(f"Bad request: {str(error)}")
    return jsonify(create_error_response(
        "BAD_REQUEST",
        "Invalid request format"
    )), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify(create_error_response(
        "INTERNAL_ERROR",
        "An internal server error occurred"
    )), 500

if __name__ == '__main__':
    logger.info("Starting Weed Segmentation Application")
    logger.info(f"Detected classes: {list(CLASS_NAMES.values())}")
    logger.info("Application configuration:")
    logger.info(f"  - Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"  - Results folder: {app.config['RESULTS_FOLDER']}")
    logger.info(f"  - Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    logger.info(f"  - Model available: {segmentation_model is not None}")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical("Failed to start application", exc_info=True)
        raise
