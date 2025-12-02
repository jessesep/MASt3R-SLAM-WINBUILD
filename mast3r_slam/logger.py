"""
MASt3R-SLAM Logging System
Provides structured logging with levels, file output, and console formatting
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import threading

# Global logger instance
_logger = None
_lock = threading.Lock()


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to levelname
        if sys.stdout.isatty():  # Only add colors if outputting to terminal
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name="mast3r_slam",
    level=logging.INFO,
    log_file=None,
    console=True,
    file_level=logging.DEBUG,
    quiet_mode=False
):
    """
    Setup the global logger

    Args:
        name: Logger name
        level: Console logging level
        log_file: Optional log file path
        console: Enable console output
        file_level: File logging level
        quiet_mode: If True, only show WARNING and above

    Returns:
        Configured logger instance
    """
    global _logger

    with _lock:
        if _logger is not None:
            return _logger

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture everything, filter at handler level
        logger.handlers.clear()  # Remove any existing handlers

        # Adjust level for quiet mode
        if quiet_mode:
            level = logging.WARNING

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = ColoredFormatter(
                '[%(levelname)s] %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, mode='a')
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        _logger = logger
        return logger


def get_logger():
    """Get the global logger instance"""
    global _logger

    if _logger is None:
        # Create default logger if not initialized
        return setup_logger()

    return _logger


# Convenience functions for direct logging
def debug(msg, *args, **kwargs):
    """Log debug message"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Log info message"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log warning message"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log error message"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log critical message"""
    get_logger().critical(msg, *args, **kwargs)


# Specialized loggers for different components
class ComponentLogger:
    """Logger for specific SLAM components"""

    def __init__(self, component_name):
        self.component = component_name
        self.logger = get_logger()

    def _format_msg(self, msg):
        return f"[{self.component}] {msg}"

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self._format_msg(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(self._format_msg(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self._format_msg(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(self._format_msg(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(self._format_msg(msg), *args, **kwargs)


# Pre-configured component loggers
def get_component_logger(component):
    """Get a logger for a specific component"""
    return ComponentLogger(component)


# Example usage
if __name__ == "__main__":
    # Test logging system
    setup_logger(level=logging.DEBUG, log_file="logs/test.log")

    debug("This is a debug message")
    info("This is an info message")
    warning("This is a warning message")
    error("This is an error message")
    critical("This is a critical message")

    # Component logger
    tracker_log = get_component_logger("Tracker")
    tracker_log.info("Frame tracked successfully")
    tracker_log.warning("Low confidence: 0.45")

    viz_log = get_component_logger("Visualization")
    viz_log.info("Rendering 10 keyframes")
    viz_log.error("Failed to update texture")
