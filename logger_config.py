"""
Professional logging configuration for Weed Segmentation Application
"""
import logging
import sys
import os
from datetime import datetime
import json
import traceback
from functools import wraps
import time
import uuid

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        if hasattr(record, 'file_size'):
            log_entry['file_size_bytes'] = record.file_size
        if hasattr(record, 'image_dimensions'):
            log_entry['image_dimensions'] = record.image_dimensions
        if hasattr(record, 'detected_classes'):
            log_entry['detected_classes'] = record.detected_classes
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        return json.dumps(log_entry)

def setup_logging(app_name="weed_segmentation", log_level=logging.INFO, log_file=None):
    """
    Setup professional logging configuration
    
    Args:
        app_name: Name of the application
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Use JSON formatter for production, simple formatter for development
    if os.getenv('FLASK_ENV') == 'production':
        console_formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for persistent logging
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # Reduce Flask noise
    logging.getLogger('PIL').setLevel(logging.WARNING)       # Reduce Pillow noise
    
    return logging.getLogger(app_name)

class ErrorCodes:
    """Standardized error codes for the application"""
    
    # File-related errors
    FILE_NOT_PROVIDED = "FILE_001"
    FILE_NOT_SELECTED = "FILE_002"
    FILE_TYPE_INVALID = "FILE_003"
    FILE_SIZE_EXCEEDED = "FILE_004"
    FILE_DIMENSIONS_INVALID = "FILE_005"
    FILE_CORRUPTED = "FILE_006"
    FILE_SAVE_FAILED = "FILE_007"
    
    # Model-related errors
    MODEL_LOAD_FAILED = "MODEL_001"
    MODEL_NOT_FOUND = "MODEL_002"
    PREDICTION_FAILED = "MODEL_003"
    TTA_FAILED = "MODEL_004"
    
    # Processing errors
    IMAGE_PROCESSING_FAILED = "PROC_001"
    OVERLAY_CREATION_FAILED = "PROC_002"
    STATISTICS_CALCULATION_FAILED = "PROC_003"
    ANIMATION_GENERATION_FAILED = "PROC_004"
    
    # System errors
    MEMORY_ERROR = "SYS_001"
    DISK_SPACE_ERROR = "SYS_002"
    PERMISSION_ERROR = "SYS_003"
    TIMEOUT_ERROR = "SYS_004"

class CustomException(Exception):
    """Base exception class with error codes and context"""
    
    def __init__(self, message, error_code=None, context=None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()

class ValidationError(CustomException):
    """Exception for validation failures"""
    pass

class ProcessingError(CustomException):
    """Exception for processing failures"""
    pass

class ModelError(CustomException):
    """Exception for model-related failures"""
    pass

def log_performance(func):
    """Decorator to log function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger('performance')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.info(
                f"Function {func.__name__} completed successfully",
                extra={
                    'function': func.__name__,
                    'duration': duration,
                    'success': True
                }
            )
            
            return result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    'function': func.__name__,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    
    return wrapper

def generate_request_id():
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def log_request(func):
    """Decorator to log HTTP requests"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        from flask import request, g
        
        # Generate request ID
        request_id = generate_request_id()
        g.request_id = request_id
        
        logger = logging.getLogger('requests')
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.path}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }
        )
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            # Determine status code
            status_code = 200
            if hasattr(result, 'status_code'):
                status_code = result.status_code
            elif isinstance(result, tuple) and len(result) > 1:
                status_code = result[1]
                
            logger.info(
                f"Request completed: {request.method} {request.path}",
                extra={
                    'request_id': request_id,
                    'status_code': status_code,
                    'duration': duration
                }
            )
            
            return result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            logger.error(
                f"Request failed: {request.method} {request.path}",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'duration': duration
                },
                exc_info=True
            )
            raise
    
    return wrapper

def create_error_response(error_code, message, details=None, request_id=None):
    """Create standardized error response"""
    return {
        'success': False,
        'error': {
            'code': error_code,
            'message': message,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'request_id': request_id
        }
    }

def create_success_response(data, message=None, request_id=None):
    """Create standardized success response"""
    response = {
        'success': True,
        'data': data,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    if message:
        response['message'] = message
    if request_id:
        response['request_id'] = request_id
        
    return response
