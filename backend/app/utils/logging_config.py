"""
Centralised Logging Configuration for SourceUp
-----------------------------------------------
Configure logging once at application entry point.
All modules should use logging.getLogger(__name__).
"""

import logging
import sys
from typing import Optional


def setup_logging ( level: str = "INFO", log_file: Optional[str] = None ) :
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    log_level = getattr( logging, level.upper(), logging.INFO )

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel( log_level )

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:] :
        root_logger.removeHandler( handler )

    # Console handler
    console_handler = logging.StreamHandler( sys.stdout )
    console_handler.setFormatter( formatter )
    root_logger.addHandler( console_handler )

    # File handler (if specified)
    if log_file :
        file_handler = logging.FileHandler( log_file, encoding='utf-8' )
        file_handler.setFormatter( formatter )
        root_logger.addHandler( file_handler )

    # Set levels for noisy third-party loggers
    logging.getLogger( "urllib3" ).setLevel( logging.WARNING )
    logging.getLogger( "requests" ).setLevel( logging.WARNING )
    logging.getLogger( "httpx" ).setLevel( logging.WARNING )
    logging.getLogger( "sentence_transformers" ).setLevel( logging.WARNING )

    logging.info( f"Logging configured: level={level}, file={log_file}" )
    return root_logger


# Default logger for modules that don't have their own
default_logger = logging.getLogger( __name__ )


def get_logger ( name: str ) -> logging.Logger :
    """Get a logger with the specified name."""
    return logging.getLogger( name )