from flask import Flask, render_template, request, jsonify, send_from_directory, g
import os
import time
import cv2
import base64
import uuid
from PIL import Image

# Import prediction logic
from weed_predictor import WeedSegmentationPredictor, CLASS_NAMES

# Import professional logging
from logger_config import (
    setup_logging, log_request, log_performance,
    ErrorCodes, ProcessingError,
    create_error_response, create_success_response
)


def env_int(name, default):
    """Read an integer setting from the environment, falling back to a default"""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_flag(name, default=False):
    """Read a boolean setting from the environment"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['RESULTS_FOLDER'] = os.getenv('RESULTS_FOLDER', 'results')
app.config['MAX_CONTENT_LENGTH'] = env_int('MAX_UPLOAD_MB', 8) * 1024 * 1024
app.config['MAX_IMAGE_DIM'] = env_int('MAX_IMAGE_DIM', 256)
app.config['MIN_IMAGE_DIM'] = env_int('MIN_IMAGE_DIM', 32)
app.config['RESULT_TTL_MINUTES'] = env_int('RESULT_TTL_MINUTES', 60)

MODEL_PATH = os.getenv('MODEL_PATH', 'models/weed_segmentation_S-TTA.pth')
DEMO_MODE = env_flag('DEMO_MODE', False)
LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Extension to MIME subtype; jpg is not a registered type, jpeg is
MIME_SUBTYPES = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'gif': 'gif', 'bmp': 'bmp'}

# Setup professional logging
logger = setup_logging(
    app_name="weed_segmentation_app",
    log_file=LOG_FILE
)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Initialize model. A failure here leaves segmentation_model as None and every
# prediction endpoint answers 503 - it must never fall back to fake output.
logger.info("Initializing weed segmentation model with TTA enhancement",
            extra={'model_path': MODEL_PATH, 'demo_mode': DEMO_MODE})
try:
    segmentation_model = WeedSegmentationPredictor(model_path=MODEL_PATH, demo_mode=DEMO_MODE)
    logger.info("Model initialization completed successfully")
except Exception:
    logger.critical(
        "Failed to initialize segmentation model - prediction endpoints will return 503",
        exc_info=True
    )
    segmentation_model = None


@app.route('/')
def index():
    return render_template(
        'index.html',
        max_upload_mb=app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024),
        max_image_dim=app.config['MAX_IMAGE_DIM']
    )


@app.route('/health')
def health():
    """Liveness and readiness probe. Deliberately unlogged - polled every 30s."""
    ready = segmentation_model is not None and segmentation_model.model is not None
    payload = {
        'status': 'ready' if ready else 'degraded',
        'model_loaded': ready,
        'demo_mode': DEMO_MODE
    }
    if ready:
        payload['device'] = str(segmentation_model.device)
    return jsonify(payload), (200 if ready else 503)


@app.route('/upload', methods=['POST'])
@log_request
@log_performance
def upload_file():
    """Handle file upload and weed segmentation processing"""
    request_id = getattr(g, 'request_id', 'unknown')
    filepath = None

    try:
        # Drop stale artifacts before adding new ones
        purge_old_results()

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
        if not allowed_file(file.filename):
            logger.warning(
                "Invalid file type uploaded",
                extra={
                    'request_id': request_id,
                    'original_filename': file.filename,
                    'content_type': file.content_type
                }
            )
            return jsonify(create_error_response(
                ErrorCodes.FILE_TYPE_INVALID,
                "Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP",
                {'supported_formats': sorted(ext.upper() for ext in ALLOWED_EXTENSIONS)},
                request_id=request_id
            )), 400

        # Generate unique filename and save
        extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        logger.info(
            "Processing file upload",
            extra={
                'request_id': request_id,
                'original_filename': file.filename,
                'unique_filename': unique_filename,
                'file_size': os.path.getsize(filepath)
            }
        )

        # Validate image dimensions
        try:
            is_valid, error_message, image_info = validate_image_dimensions(filepath)
        except Exception:
            logger.error("Image validation error", extra={'request_id': request_id}, exc_info=True)
            return jsonify(create_error_response(
                ErrorCodes.FILE_CORRUPTED,
                "Unable to process image file",
                request_id=request_id
            )), 400

        if not is_valid:
            logger.warning(
                "Image validation failed",
                extra={'request_id': request_id, 'error': error_message, 'image_info': image_info}
            )
            return jsonify(create_error_response(
                ErrorCodes.FILE_DIMENSIONS_INVALID,
                error_message,
                image_info,
                request_id=request_id
            )), 400

        # Process with model
        if segmentation_model is None or segmentation_model.model is None:
            logger.error("Segmentation model not available", extra={'request_id': request_id})
            return jsonify(create_error_response(
                ErrorCodes.MODEL_NOT_FOUND,
                "Segmentation model is not available",
                request_id=request_id
            )), 503

        try:
            logger.info("Starting TTA segmentation", extra={'request_id': request_id})
            mask = segmentation_model.predict(filepath)

            class_stats = segmentation_model.calculate_class_statistics(mask)

            result_path = create_result_image(filepath, mask, unique_filename)

            original_b64 = image_to_base64(filepath)
            result_b64 = image_to_base64(result_path)

            # class_stats is keyed by class name, so compare against the names
            detected_classes = [
                name for name in CLASS_NAMES.values()
                if name != 'background' and class_stats.get(name, 0) > 0.1
            ]

            logger.info(
                "Segmentation completed successfully",
                extra={
                    'request_id': request_id,
                    'detected_classes': detected_classes,
                    'image_dimensions': f"{image_info['width']}x{image_info['height']}"
                }
            )

            response_data = {
                'original_image': original_b64,
                'segmented_image': result_b64,
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

        except Exception:
            logger.error("Segmentation processing failed", extra={'request_id': request_id}, exc_info=True)
            return jsonify(create_error_response(
                ErrorCodes.PREDICTION_FAILED,
                "Error during image processing",
                request_id=request_id
            )), 500

    except Exception:
        logger.error("Unexpected error in upload handler", extra={'request_id': request_id}, exc_info=True)
        return jsonify(create_error_response(
            "INTERNAL_ERROR",
            "An unexpected error occurred",
            request_id=request_id
        )), 500

    finally:
        # The upload is only needed while the request is being served
        if filepath:
            remove_quietly(filepath)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_quietly(path):
    """Delete a file, logging but not raising if it cannot be removed"""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"Could not remove temporary file {path}: {e}")


def purge_old_results():
    """Delete result images older than the configured TTL"""
    ttl_seconds = app.config['RESULT_TTL_MINUTES'] * 60
    if ttl_seconds <= 0:
        return

    cutoff = time.time() - ttl_seconds
    removed = 0
    try:
        entries = os.scandir(app.config['RESULTS_FOLDER'])
    except OSError:
        return

    with entries:
        for entry in entries:
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
                    removed += 1
            except OSError:
                continue

    if removed:
        logger.info(f"Purged {removed} expired result files")


def validate_image_dimensions(filepath):
    """
    Validate image dimensions with detailed error reporting

    Returns:
        tuple: (is_valid, error_message, image_info)
    """
    max_dim = app.config['MAX_IMAGE_DIM']
    min_dim = app.config['MIN_IMAGE_DIM']

    with Image.open(filepath) as img:
        width, height = img.size
        image_info = {
            'width': width,
            'height': height,
            'format': img.format,
            'mode': img.mode,
            'file_size': os.path.getsize(filepath)
        }

        if width > max_dim or height > max_dim:
            return False, (
                f'Image dimensions ({width}x{height}) exceed maximum allowed size '
                f'of {max_dim}x{max_dim} pixels'
            ), image_info

        if width < min_dim or height < min_dim:
            return False, (
                f'Image dimensions ({width}x{height}) are too small. '
                f'Minimum size is {min_dim}x{min_dim} pixels'
            ), image_info

        return True, None, image_info


@log_performance
def create_result_image(image_path, mask, filename):
    """
    Render the segmentation overlay and persist it for download.

    The progressive reveal used to be rendered here as 21 separate frames and
    shipped as base64; the browser now animates a single image instead.
    """
    result = segmentation_model.create_overlay_visualization(image_path, mask)

    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
    if not cv2.imwrite(result_path, result):
        raise ProcessingError(
            "Failed to save overlay result",
            ErrorCodes.OVERLAY_CREATION_FAILED
        )

    logger.debug(f"Overlay written to {result_path}")
    return result_path


def image_to_base64(image_path):
    """Convert image to a data URI"""
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    except OSError as e:
        logger.error(f"Error reading image for base64 conversion: {image_path}", exc_info=True)
        raise ProcessingError(
            f"Failed to convert image to base64: {e}",
            ErrorCodes.IMAGE_PROCESSING_FAILED
        )

    extension = image_path.rsplit('.', 1)[-1].lower()
    subtype = MIME_SUBTYPES.get(extension, 'png')
    return f"data:image/{subtype};base64,{img_base64}"


@app.route('/download/<filename>')
@log_request
def download_file(filename):
    """Handle file download with security validation"""
    request_id = getattr(g, 'request_id', 'unknown')

    try:
        # Reject anything that could escape the results folder
        if filename != os.path.basename(filename) or '..' in filename:
            logger.warning(
                "Suspicious filename in download request",
                extra={'request_id': request_id, 'requested_file': filename}
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
                extra={'request_id': request_id, 'requested_file': filename}
            )
            return jsonify(create_error_response(
                "FILE_NOT_FOUND",
                "Requested file does not exist",
                request_id=request_id
            )), 404

        logger.info("File download initiated",
                    extra={'request_id': request_id, 'requested_file': filename})

        return send_from_directory(app.config['RESULTS_FOLDER'], filename, as_attachment=True)

    except Exception:
        logger.error(
            "Error during file download",
            extra={'request_id': request_id, 'requested_file': filename},
            exc_info=True
        )
        return jsonify(create_error_response(
            "DOWNLOAD_ERROR",
            "Error downloading file",
            request_id=request_id
        )), 500


@app.route('/get_model_info')
@log_request
def get_model_info():
    """Get current model information with enhanced details"""
    request_id = getattr(g, 'request_id', 'unknown')

    try:
        if segmentation_model is None or segmentation_model.model is None:
            logger.warning("Model info requested but model not available",
                           extra={'request_id': request_id})
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
            'version': '1.1.0'
        }

        logger.info("Model information retrieved", extra={'request_id': request_id})

        return jsonify(create_success_response(
            model_info,
            "Model information retrieved successfully",
            request_id=request_id
        ))

    except Exception:
        logger.error("Error retrieving model information",
                     extra={'request_id': request_id}, exc_info=True)
        return jsonify(create_error_response(
            "MODEL_INFO_ERROR",
            "Error retrieving model information",
            request_id=request_id
        )), 500


@app.errorhandler(413)
def file_too_large(error):
    """Handle file size exceeded error"""
    limit_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    logger.warning(f"File upload rejected - size exceeds limit of {limit_mb}MB")
    return jsonify(create_error_response(
        ErrorCodes.FILE_SIZE_EXCEEDED,
        f"File size exceeds maximum limit of {limit_mb}MB"
    )), 413


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    logger.warning(f"Bad request: {error}")
    return jsonify(create_error_response(
        "BAD_REQUEST",
        "Invalid request format"
    )), 400


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify(create_error_response(
        "INTERNAL_ERROR",
        "An internal server error occurred"
    )), 500


if __name__ == '__main__':
    port = env_int('PORT', 5000)
    logger.info("Starting Weed Segmentation Application")
    logger.info(f"Detected classes: {list(CLASS_NAMES.values())}")
    logger.info("Application configuration:")
    logger.info(f"  - Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"  - Results folder: {app.config['RESULTS_FOLDER']}")
    logger.info(f"  - Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)}MB")
    logger.info(f"  - Max image size: {app.config['MAX_IMAGE_DIM']}px")
    logger.info(f"  - Result TTL: {app.config['RESULT_TTL_MINUTES']} min")
    logger.info(f"  - Model available: {segmentation_model is not None}")

    try:
        app.run(debug=False, host=os.getenv('HOST', '0.0.0.0'), port=port)
    except Exception:
        logger.critical("Failed to start application", exc_info=True)
        raise
