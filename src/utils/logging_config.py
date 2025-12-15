"""
Logging Configuration for Arabic Attendance System

Comprehensive logging setup with file rotation and structured output.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ProcessingLogger:
    """Specialized logger for processing operations."""
    
    def __init__(self, name: str = "processing"):
        """Initialize processing logger."""
        self.logger = logging.getLogger(name)
        self.processing_stats = {
            'files_processed': 0,
            'images_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_processing_time': 0.0
        }
    
    def log_file_start(self, file_path: str, file_size: int = None):
        """Log start of file processing."""
        self.logger.info("Starting file processing", extra={
            'event_type': 'file_start',
            'file_path': file_path,
            'file_size': file_size
        })
    
    def log_file_complete(self, file_path: str, processing_time: float, 
                         images_count: int, success_count: int):
        """Log completion of file processing."""
        self.processing_stats['files_processed'] += 1
        self.processing_stats['images_processed'] += images_count
        self.processing_stats['successful_extractions'] += success_count
        self.processing_stats['failed_extractions'] += (images_count - success_count)
        self.processing_stats['total_processing_time'] += processing_time
        
        self.logger.info("File processing completed", extra={
            'event_type': 'file_complete',
            'file_path': file_path,
            'processing_time': processing_time,
            'images_count': images_count,
            'success_count': success_count,
            'success_rate': success_count / images_count if images_count > 0 else 0.0
        })
    
    def log_image_processing(self, image_id: str, result: Dict[str, Any]):
        """Log image processing result."""
        self.logger.info("Image processed", extra={
            'event_type': 'image_processed',
            'image_id': image_id,
            'success': result.get('success', False),
            'extracted_rows': len(result.get('extracted_data', [])),
            'confidence': result.get('confidence_scores', {}).get('average_confidence', 0.0)
        })
    
    def log_extraction_details(self, image_id: str, extraction_type: str, 
                              value: str, confidence: float):
        """Log detailed extraction information."""
        self.logger.debug("Extraction detail", extra={
            'event_type': 'extraction_detail',
            'image_id': image_id,
            'extraction_type': extraction_type,
            'value': value,
            'confidence': confidence
        })
    
    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log processing error."""
        extra_data = {
            'event_type': 'processing_error',
            'error_type': error_type,
            'error_message': error_message
        }
        if context:
            extra_data.update(context)
        
        self.logger.error("Processing error occurred", extra=extra_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'files_processed': 0,
            'images_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_processing_time': 0.0
        }


class AuditLogger:
    """Logger for audit trail and compliance."""
    
    def __init__(self, name: str = "audit"):
        """Initialize audit logger."""
        self.logger = logging.getLogger(name)
    
    def log_api_request(self, endpoint: str, method: str, user_id: str = None,
                       file_info: Dict = None):
        """Log API request."""
        self.logger.info("API request", extra={
            'event_type': 'api_request',
            'endpoint': endpoint,
            'method': method,
            'user_id': user_id,
            'file_info': file_info,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_data_access(self, data_type: str, action: str, record_count: int = None):
        """Log data access."""
        self.logger.info("Data access", extra={
            'event_type': 'data_access',
            'data_type': data_type,
            'action': action,
            'record_count': record_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_system_event(self, event_type: str, description: str, 
                        severity: str = "info"):
        """Log system event."""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method("System event", extra={
            'event_type': 'system_event',
            'system_event_type': event_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })


def setup_logging(log_dir: str = "logs", log_level: str = "INFO",
                 enable_json_logging: bool = True, enable_file_logging: bool = True):
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        enable_json_logging: Whether to use JSON formatting
        enable_file_logging: Whether to log to files
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_json_logging:
        console_formatter = JSONFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / "application.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(JSONFormatter() if enable_json_logging else console_formatter)
        root_logger.addHandler(app_handler)
        
        # Error log
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter() if enable_json_logging else console_formatter)
        root_logger.addHandler(error_handler)
        
        # Processing log
        processing_handler = logging.handlers.RotatingFileHandler(
            log_path / "processing.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        processing_handler.setLevel(logging.DEBUG)
        processing_handler.setFormatter(JSONFormatter() if enable_json_logging else console_formatter)
        
        # Add filter for processing logger
        processing_logger = logging.getLogger("processing")
        processing_logger.addHandler(processing_handler)
        processing_logger.setLevel(logging.DEBUG)
        
        # Audit log
        audit_handler = logging.handlers.RotatingFileHandler(
            log_path / "audit.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=20  # Keep more audit logs
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(JSONFormatter() if enable_json_logging else console_formatter)
        
        # Add filter for audit logger
        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    logging.info("Logging configuration completed")


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, name: str = "performance"):
        """Initialize performance logger."""
        self.logger = logging.getLogger(name)
    
    def log_timing(self, operation: str, duration: float, context: Dict = None):
        """Log operation timing."""
        extra_data = {
            'event_type': 'performance_timing',
            'operation': operation,
            'duration_seconds': duration
        }
        if context:
            extra_data.update(context)
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s", 
                        extra=extra_data)
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage."""
        self.logger.info(f"Memory usage for '{operation}': {memory_mb:.1f}MB", extra={
            'event_type': 'memory_usage',
            'operation': operation,
            'memory_mb': memory_mb
        })
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float, 
                          disk_usage_mb: float = None):
        """Log system resource usage."""
        extra_data = {
            'event_type': 'resource_usage',
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb
        }
        if disk_usage_mb is not None:
            extra_data['disk_usage_mb'] = disk_usage_mb
        
        self.logger.info(f"Resource usage - CPU: {cpu_percent:.1f}%, "
                        f"Memory: {memory_mb:.1f}MB", extra=extra_data)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def get_processing_logger() -> ProcessingLogger:
    """Get processing logger instance."""
    return ProcessingLogger()


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return AuditLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return PerformanceLogger()


# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[PerformanceLogger] = None):
        """Initialize timing context."""
        self.operation_name = operation_name
        self.logger = logger or get_performance_logger()
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.log_timing(self.operation_name, duration)


# Decorator for timing functions
def timed_operation(operation_name: str = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with TimingContext(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator