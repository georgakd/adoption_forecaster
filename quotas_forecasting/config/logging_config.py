"""
Centralized logging configuration for quotas_forecasting.

This module provides consistent logging setup across all modules with:
- Timestamps
- Function/method names  
- Module names
- Consistent formatting
"""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(level: int = logging.INFO, enable_colors: bool = True, log_to_file: Optional[str] = None) -> logging.Logger:
    """
    Set up consistent logging configuration across the application.

    Args:
        level: Logging level (default: INFO)
        enable_colors: Whether to enable colored output for console
        log_to_file: Optional file path to also log to file

    Returns:
        Configured logger instance
    """

    # Create formatter
    log_format = "%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if enable_colors:
        console_formatter = ColoredFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        console_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Usually __name__ of the calling module

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging when this module is imported
# This ensures consistent logging across all modules
setup_logging()
