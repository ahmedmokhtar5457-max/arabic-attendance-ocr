#!/usr/bin/env python3
"""
Server startup script for Arabic Attendance Extraction System

Provides easy server startup with configuration options.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging


def main():
    """Main server startup function."""
    parser = argparse.ArgumentParser(
        description="Arabic Attendance Data Extraction Server"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration if available"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Arabic Attendance Data Extraction Server")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Workers: {args.workers}, Log Level: {args.log_level}")
    logger.info(f"GPU Enabled: {args.use_gpu}")
    
    # Set environment variables for configuration
    if args.use_gpu:
        os.environ["USE_GPU"] = "true"
    
    if args.config:
        os.environ["CONFIG_FILE"] = args.config
    
    try:
        import uvicorn
        
        # Run the server
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
    except ImportError:
        logger.error("uvicorn not installed. Please install with: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()